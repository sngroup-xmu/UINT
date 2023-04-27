from functools import partial

from src import NeurouNetwork
from src.utils.activation import *
from src.utils.linearalgebra import *

from z3 import Int, Real, RealVal, And, Or

class ConstraintEncoder():

    def __init__(self, model_path: str, system_name: str, model_in_name, model_out_len: int, multi=None, K=1, hierarchy=False):
        '''
        :param model_path:
        :param system_name:
        :param model_in_name: 模型输入特征的名字
        :param model_out_len: 输出层神经元的个数
        :param multi: 模型放大的倍数（在强制将模型输入变量转换为Int型时需要用到）
        :param K: 模型复制K个（考虑验证连续K个状态需要把模型复制K份）
        :param hierarchy: （是否开启分层编码：加入中间变量）
        '''
        nn = NeurouNetwork(model_path, system_name)
        self.K = K
        self.multi = multi
        if multi == 1.0 or multi is None:
            self._to_int = False
        else:
            self._to_int = True

        self._hierarchy = hierarchy

        # Z3 variables
        self._in_vars, self._nn_vars, self._out_vars = [], [], []

        # Z3 constraints
        self._in_out_constrs, self._nn_constrs = [], []

        # Z3 tracks（仅在使用unsat core时会用到）
        self._tracks = {}

        weights, biases, activ = nn.parameters()
        if multi is not None:
            # scaling bias
            new_biases = []
            for bias in biases:
                new_bias = []
                for bia in bias:
                    if isinstance(bia, list):
                        new_bias.append([b * multi for b in bia])
                    else:
                        new_bias.append(bia * multi)
                new_biases.append(new_bias)
            biases = new_biases

        for i in range(K):
            self._encode_nn(model_in_name, weights, biases, activ, model_out_len, state=i)

    def _encode_nn(self, model_in_name, weights, biases, activ, model_out_len, state=0):

        H = []
        # 创建输入变量
        if self._to_int:
            self._in_vars.append([Int(f'{name}${state}') for name in model_in_name])
        else:
            self._in_vars.append([Real(f'{name}${state}') for name in model_in_name])
        # 创建输出变量
        self._out_vars.append([Real(f'out_{i}${state}') for i in range(model_out_len)])

        H.append(self._in_vars[-1])
        # 分层编码（加入中间变量）
        if self._hierarchy:
            layer = 1
            for idx, (weight, bias) in enumerate(zip(weights, biases)):
                I = H[-1]
                # Pensieve系统结构的特殊处理，即第一层有3个全连接和3个卷积，然后堆叠
                # 如果有其他特殊nn结构，在此增加分支即可
                if isinstance(weight, list):
                    num_hidden_vars = 0
                    for w_v in weight:
                        num_hidden_vars += w_v.shape[1] ####
                    # 中间变量
                    hidden_vars = [Real(f'hidden{layer}_{i}${state}') for i in range(num_hidden_vars)]

                    start_pos = 0
                    O = []
                    for i in range(len(weight)):
                        w_v = weight[i]
                        W = map(partial(map, RealVal), w_v) ####
                        B = map(RealVal, bias[i])
                        O.extend(vecadd(vecmatprod(I[start_pos: start_pos + w_v.shape[0]], W), B)) ####

                        start_pos += weight[i].shape[0]
                else:
                    num_hidden_vars = weight.shape[1]
                    hidden_vars = [Real(f'hidden{layer}_{i}${state}') for i in range(num_hidden_vars)]
                    W = map(partial(map, RealVal), weight)
                    B = map(RealVal, bias)
                    O = vecadd(vecmatprod(I, W), B)

                if idx < len(weights) - 1:
                    O = activation(activ, O)
                    hidden_constrs = [None for _ in range(num_hidden_vars)]
                    for idx, hidden_expr in enumerate(O):
                        hidden_constrs[idx] = hidden_vars[idx] == hidden_expr
                    H.append(hidden_vars)
                    layer += 1
                    self._nn_constrs.extend(hidden_constrs)
                else:
                    H.append(O)
            self._nn_constrs.extend([out_var == expr for out_var, expr in zip(self._out_vars[-1], H.pop())])
        else:
            for idx, (weight, bias) in enumerate(zip(weights, biases)):
                I = H[-1]

                if isinstance(weight, list):
                    start_pos = 0
                    O = []
                    for i in range(len(weight)):
                        w_v = weight[i]
                        W = map(partial(map, RealVal), w_v)
                        B = map(RealVal, bias[i])
                        O.extend(vecadd(vecmatprod(I[start_pos: start_pos + w_v.shape[0]], W), B))
                        start_pos += weight[i].shape[0]
                else:
                    W = map(partial(map, RealVal), weight)
                    B = map(RealVal, bias)
                    O = vecadd(vecmatprod(I, W), B)

                if idx < len(weights) - 1:
                    O = activation(activ, O)
                H.append(O)

            self._nn_constrs.extend([out_var == expr for out_var, expr in zip(self._out_vars[-1], H.pop())])

    def argmax_eq_constr(self, vars: list, idx: int):
        # 最大值下标相等约束
        exprs = []
        for i in range(len(vars)):
            if i == idx:
                continue
            exprs.append(vars[idx] >= vars[i])
        self._in_out_constrs.append(And(exprs))

    def argmax_neq_constr(self, vars, idx):
        # 最大值下标不等约束
        exprs = []
        for i in range(len(vars)):
            if i == idx:
                continue
            exprs.append(vars[idx] < vars[i])
        self._in_out_constrs.append(Or(exprs))

    def eq_constr(self, vars1, vars2):
        # 严格相等约束
        self._in_out_constrs.append([x1 == x2 for x1, x2 in zip(vars1, vars2)])

    def approximate_eq_constr(self, vars, values, float_=0.5): ####
        # 近似相等约束
        # float_ 是近似的程度，根据变量取值范围定
        if self.multi is None:
            self._in_out_constrs.append(And([
                And([x1 > x2 - float_, x1 < x2 + float_])
                for x1, x2 in zip(vars, values)]))
        else:
            self._in_out_constrs.append(And([
                And([x1 > x2 - float_ * self.multi, x1 < x2 + float_ * self.multi])
                for x1, x2 in zip(vars, values)]))

    def approximate_neq_constr(self, vars, values, float_=0.5):
        or_constraints = []
        for var, value in zip(vars, values):
            if self.multi is None:
                or_constraints.extend([var < value - float_, var > value + float_])
            else:
                or_constraints.extend([var < value - float_ * self.multi, var > value + float_ * self.multi])
        self._in_out_constrs.append(Or(or_constraints))

    def assign_constr(self, vars, values):
        # 变量赋值约束
        constr = And([var == value for var, value in zip(vars, values)])
        self._in_out_constrs.append(constr)

        return constr

    def boundary_constr(self, vars, bounds):
        # 变量设置边界约束
        # bounds是左开右闭区间
        constrs = []
        for var, bound in zip(vars, bounds):
            if bound[0] is None:
                constrs.append(var <= bound[1])
            elif bound[1] is None:
                constrs.append(var >= bound[0])
            else:
                constrs.extend([var >= bound[0], var < bound[1]])
        self._in_out_constrs.append(And(constrs))

        return And(constrs)

    def discrete_constr(self, vars, possible_values):
        # 变量在一组离散值中间取值约束
        constr = And([Or([var == value for value in values]) 
                                         for var, values in zip(vars, possible_values)])
        self._in_out_constrs.append(constr)

        return constr

    def assign_track(self, vars, values):

        for var, value in zip(vars, values):
            self._tracks[f'{str(var)}_assign'] = var == value

    def eq_track(self, vars1, vars2):
        for var1, var2 in zip(vars1, vars2):
            self._tracks[f'{str(var1)}_eq'] = var1 == var2

    def boundary_track(self, vars, bounds):
        for var, bound in zip(vars, bounds):
            track_name = f'{str(var)}$_bound'
            if bound[0] is None:
                self._tracks[track_name] = var <= bound[1]
            elif bound[1] is None:
                self._tracks[track_name] = var >= bound[0]
            else:
                self._tracks[track_name] = And([var >= bound[0], var <= bound[1]])

    def discrete_track(self, vars, possible_values):
        for var, values in zip(vars, possible_values):
            track_name = f'{str(var)}$_discrete'
            self._tracks[track_name] = Or([var == value for value in values])

    def get_vars(self, idx=None):
        if idx is None:
            return self._in_vars, self._out_vars
        else:
            return self._in_vars[idx], self._out_vars[idx]

    def get_nn_constrs(self):
        return self._nn_constrs

    def get_in_out_constrs(self):
        return self._in_out_constrs

    def get_tracks(self):
        return self._tracks

    def reset(self):
        self._in_out_constrs = []
        self._tracks = {}

    






