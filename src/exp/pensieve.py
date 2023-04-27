import csv
from hmac import new
import json
from re import M
import time
import argparse

import numpy as np
from z3 import sat, unsat, And, Or, RealVal, If, SolverFor
from z3 import *
import sys
sys.path.append("../..") 
from src import ConstraintEncoder, split_list_by_idxs, find_solution

parser = argparse.ArgumentParser()
parser.add_argument('--system-precision', type=float, default=0.01) #sa: change to 0.00001
parser.add_argument('--is-multi', type=bool, default=True)
args = parser.parse_args()

BASE_DIR = '../..'
SYSTEM_NAME = 'pensieve'
MODEL_PATH = f'{BASE_DIR}/model_file/pensieve/{SYSTEM_NAME}.pb'
dataset = json.load(open(f'{BASE_DIR}/model_file/pensieve/{SYSTEM_NAME}_dataset.json', 'r'))

IS_MULTI = args.is_multi
PRECISION = args.system_precision
if IS_MULTI:
    MULTI = int(1 / PRECISION)
else:
    MULTI = 1.0

def data_round(value):
    if IS_MULTI:
        return round(value) #四舍五入
    else:
        return value

def data_multi(value, reverse=False):
    if IS_MULTI:
        if reverse:
            return value / MULTI
        else:
            return round(value * MULTI)
    else:
        return value

OUT_LENGTH = 6
HISTORY_LENS = dataset['historys']
FEATURE_TYPES = int(dataset['types'])
FEATURE_NAMES = dataset['names']
FEATURE_BOUNDS = [(data_multi(lb), data_multi(ub))for lb, ub in dataset['bounds']]
FEATURE_RANGES = [bound[1] - bound[0] for bound in FEATURE_BOUNDS]

PRE_BIT_POS = 0
REMAIN_THUNKS_POS = 24
DISCRETE_VALUES = {
    PRE_BIT_POS: [data_multi(v / 4300) for v in [300, 750, 1200, 1850, 2850, 4300]],
    REMAIN_THUNKS_POS: [data_multi(v / 48) for v in range(48)],
}

FEATURE_START_POS = [0, 1, 2, 10, 18, 24, 25] ###

data_list = dataset['data']

def generate_idxs(features, historys, pos='R'):

    if pos == 'R':
        if isinstance(features, list):
            return [FEATURE_START_POS[f + 1] - i - 1 for f, history in zip(features, historys) for i in range(history)]
        else:
            return [FEATURE_START_POS[features + 1] - i - 1 for i in range(historys)]
    else:
        if isinstance(features, list):
            return [FEATURE_START_POS[f] + i for f, history in zip(features, historys) for i in range(history)]
        else:
            return [FEATURE_START_POS[features] + i for i in range(historys)]

def map_to_closest(value, options):
    dist = float('inf')
    selected = None
    for option in options:
        if abs(value - option) < dist:
            selected = option
            dist = abs(option - value)
    return selected

############################### Verification Problem ######################################
def adversarial_perturbation(pfs, historys, epsilon):

    print('*********** Verify Adversarial Perturbation *********')
    print(f'perturbation features: {pfs}, history: {historys}')

    solver = SolverFor('LRA')

    x_, y_ = data_list[0]['x'], data_list[0]['y']
    x_ = [data_multi(v) for v in x_]
    y_ = [v * MULTI for v in y_]

    encoder = ConstraintEncoder(MODEL_PATH, SYSTEM_NAME, FEATURE_NAMES, OUT_LENGTH, multi=MULTI)
    in_vars, out_vars = encoder.get_vars(0)

    # Discrete value contraints
    discrete_vars, discrete_values = [in_vars[f] for f in DISCRETE_VALUES.keys()], list(DISCRETE_VALUES.values())

    # Generate perturbation features, values, bounds and ranges
    p_idxs = generate_idxs(pfs, historys)
    perturb_vars, normal_vars = split_list_by_idxs(in_vars, p_idxs)
    perturb_values, normal_values = split_list_by_idxs(x_, p_idxs)
    perturb_ranges = [FEATURE_RANGES[idx] for idx in p_idxs]

    perturb_bounds = []
    for p_idx, v, k in zip(p_idxs, perturb_values, perturb_ranges):
        if p_idx in DISCRETE_VALUES.keys():
            tmp = (map_to_closest(data_round(v - k * epsilon), DISCRETE_VALUES[p_idx]),
                   map_to_closest(data_round(v + k * epsilon), DISCRETE_VALUES[p_idx]))
            perturb_bounds.append(tmp)
        else:
            perturb_bounds.append((data_round(v - k * epsilon), data_round(v + k * epsilon)))

    encoder.discrete_constr(discrete_vars, discrete_values)  # 变量在一组离散值中间取值约束
    encoder.boundary_constr(perturb_vars, perturb_bounds) # 变量设置边界约束
    encoder.assign_constr(normal_vars, normal_values) #对未扰动变量设置赋值约束
    encoder.argmax_neq_constr(out_vars, np.argmax(y_))  #output Q

    # Add constraints to Z3
    solver.add(encoder.get_in_out_constrs())
    solver.add(encoder.get_nn_constrs())
   
    # with open('assert_adversarial_perturbation.txt','w',encoding='utf-8') as f:
    #     for c in solver.assertions():
    #         f.write(str(c))

    start_t = time.time()
    res = "able" if solver.check() == unsat else "unable"
    total_t = time.time() - start_t

    print(f'Epsilon: {epsilon} | Result: {res} | Time: {total_t} \n')

    return res, total_t

def verify_missing_features(mf, history):

    print('******* Verify Missing Features *******')

    solver = SolverFor('LRA')

    x_, y_ = data_list[0]['x'], data_list[0]['y']
    x_ = [data_multi(v) for v in x_]
    y_ = [v * MULTI for v in y_]

    encoder = ConstraintEncoder(MODEL_PATH, SYSTEM_NAME, FEATURE_NAMES, OUT_LENGTH, multi=MULTI)
    in_vars, out_vars = encoder.get_vars(0)

    m_idxs = generate_idxs(mf, history)
    missing_vars, normal_vars = split_list_by_idxs(in_vars, m_idxs)
    _, normal_values = split_list_by_idxs(x_, m_idxs)

    missing_bounds = [FEATURE_BOUNDS[idx] for idx in m_idxs]

    # Discrete value contraints
    discrete_vars, discrete_values = [in_vars[f] for f in DISCRETE_VALUES.keys()], list(DISCRETE_VALUES.values())

    encoder.discrete_constr(discrete_vars, discrete_values)
    encoder.boundary_constr(missing_vars, missing_bounds)
    encoder.assign_constr(normal_vars, normal_values)
    encoder.argmax_neq_constr(out_vars, np.argmax(y_))

    solver.add(encoder.get_in_out_constrs())
    solver.add(encoder.get_nn_constrs())

    start_t = time.time()
    res = "keep" if solver.check() == unsat else "don't keep"
    total_t = time.time() - start_t

    return res, total_t

def verify_extreme_values(history=8):

    print('****** Verify Extreme Values *******')

    special_idxs = generate_idxs([0, 2], [1, history])

    solver = SolverFor('LRA')

    encoder = ConstraintEncoder(MODEL_PATH, SYSTEM_NAME, FEATURE_NAMES, OUT_LENGTH, multi=MULTI)
    in_vars, out_vars = encoder.get_vars(0)

    bounds = [bound for bound in FEATURE_BOUNDS]
    for idx in special_idxs:
        if idx == 0:
            tmp_bound = (bounds[idx][0], data_round(1850 * MULTI / 4300))
        elif idx >= FEATURE_START_POS[2] and idx < FEATURE_START_POS[3]:
            tmp_bound = (data_round(0.54 * MULTI), bounds[idx][1])
        else:
            tmp_bound = bounds[idx]
        bounds[idx] = tmp_bound

    # Discrete value contraints
    discrete_vars, discrete_values = [in_vars[f] for f in DISCRETE_VALUES.keys()], list(DISCRETE_VALUES.values())

    encoder.discrete_constr(discrete_vars, discrete_values)
    encoder.boundary_constr(in_vars, bounds)
    encoder.assign_constr(in_vars[1:2], [data_multi(0.4)])
    encoder.argmax_neq_constr(out_vars, 3)

    solver.add(encoder.get_in_out_constrs())
    solver.add(encoder.get_nn_constrs())

    start_t = time.time()
    res = 'unsat' if solver.check() == sat else 'sat'
    return  res, time.time() - start_t

def verify_decision_boundary(log_path):
    print('****** Verify Decision Boundary *******')

    file = open(log_path, 'w', newline='')
    writer = csv.writer(file)
    writer.writerow(['Features', 'Result', 'Time'])
    file.flush()

    # Initialize
    solver = SolverFor('LRA')

    x_, y_ = data_list[0]['x'], data_list[0]['y']
    x_ = [data_multi(v) for v in x_]
    y_ = [v * MULTI for v in y_]

    encoder = ConstraintEncoder(MODEL_PATH, SYSTEM_NAME, FEATURE_NAMES, OUT_LENGTH)
    in_vars, out_vars = encoder.get_vars(0)

    pairs = []
    for i in range(FEATURE_TYPES - 1):
        for j in range(HISTORY_LENS[i]):
            f1_idx = FEATURE_START_POS[i] + j
            for f2_idx in range(FEATURE_START_POS[i + 1], FEATURE_START_POS[-1]):
                pairs.append([f1_idx, f2_idx])

    # Discrete value contraints
    discrete_vars, discrete_values = [in_vars[f] for f in DISCRETE_VALUES.keys()], list(DISCRETE_VALUES.values())

    for pair in pairs:

        bounds = [FEATURE_BOUNDS[idx] for idx in pair]
        select_vars, normal_vars = split_list_by_idxs(in_vars, pair)
        _, normal_values = split_list_by_idxs(x_, pair)

        encoder.reset()
        encoder.boundary_constr(select_vars, bounds)
        encoder.discrete_constr(discrete_vars, discrete_values)
        encoder.assign_constr(normal_vars, normal_values)
        encoder.argmax_neq_constr(out_vars, np.argmax(y_))

        solver.reset()
        solver.add(encoder.get_in_out_constrs())
        solver.add(encoder.get_nn_constrs())

        start_t = time.time()
        res = "don't keep" if solver.check() == sat else "keep"
        total_t = time.time() - start_t

        print(f'Features: {pair}, Result: {res}, Time: {total_t}')
        writer.writerow([pair, res, total_t])
        file.flush()
    file.close()


# ############################### Interpretability Problem ####################################

def qualitatively_anchor_t(x, y):
    print(f'******** Qualitatively Anchor *******')

    x = [data_multi(v) for v in x]
    y = [v * MULTI for v in y]
    print(y)
    solver = SolverFor('LRA')
    solver.set(unsat_core=True)
    solver.set("smt.core.minimize", True)

    encoder = ConstraintEncoder(MODEL_PATH, SYSTEM_NAME, FEATURE_NAMES, OUT_LENGTH,multi=MULTI)
    in_vars, out_vars = encoder.get_vars(0)

    solver.add_soft(encoder.assign_constr(in_vars,x))
    # encoder.assign_track(in_vars, x)
    # encoder.approximate_neq_constr(out_vars, y)
    solver.add(encoder.argmax_neq_constr(out_vars, np.argmax(y)))
    
    # tracks = encoder.get_tracks()
    # for name, expr in tracks.items():
    #     solver.assert_and_track(expr, name)

    # solver.add(encoder.get_in_out_constrs())
    solver.add(encoder.get_nn_constrs())
    # i=0
    # with open('expr_qualitatively_anchor_t.txt','w',encoding='utf-8') as f:
    #     f.write(solver.sexpr())
    start_t = time.time()
    if solver.check() == unsat:
        print('unsat')
        CoMSS = solver.unsat_core()
        features = [str(x).split('$')[0] for x in CoMSS]
        print(CoMSS)
    else:
        print('sat')
        features = []

    return features, time.time() - start_t

def qualitatively_anchor(x, y):
    print(f'******** Qualitatively Anchor *******')

    x = [data_multi(v) for v in x]
    y = [v * MULTI for v in y]
    print(y)
    solver = SolverFor('LRA')
    g=Goal()
    t = Tactic('tseitin-cnf')
    solver.set(unsat_core=True)
    solver.set("smt.core.minimize", True)

    encoder = ConstraintEncoder(MODEL_PATH, SYSTEM_NAME, FEATURE_NAMES, OUT_LENGTH,multi=MULTI)
    in_vars, out_vars = encoder.get_vars(0)

    encoder.assign_track(in_vars, x)
    # encoder.approximate_neq_constr(out_vars, y)
    encoder.argmax_neq_constr(out_vars, np.argmax(y))
    
    tracks = encoder.get_tracks()
    for name, expr in tracks.items():
        solver.assert_and_track(expr, name)
        g.assert_and_track(expr,name)

    solver.add(encoder.get_in_out_constrs())
    solver.add(encoder.get_nn_constrs())
    g.add(encoder.get_in_out_constrs())
    g.add(encoder.get_nn_constrs())

    # i=0
    # with open('expr_qualitatively_anchor.txt','w',encoding='utf-8') as f:#使用with open()新建对象f
    #     f.write(solver.sexpr())
    # clauses = t(g)
    # i=0
    # with open('clause_qualitatively_anchor.txt','w',encoding='utf-8') as f:#使用with open()新建对象f
    #     for clause in clauses[0]:
    #         i=i+1
    #         f.write(str(i)+":"+str(clause)+"\n")
    
    start_t = time.time()
    if solver.check() == unsat:
        print('unsat')
        CoMSS = solver.unsat_core()
        features = [str(x).split('$')[0] for x in CoMSS]
        print(CoMSS)
    else:
        print('sat')
        features = []

    return features, time.time() - start_t

def find_minimal_attack_power(pfs, historys, precision=1e-3):

    print('***** Minimal Attack Power ******')
    print(f'Perturbation Features: {pfs} | History: {historys}')

    solver = SolverFor('LRA')
    g=Goal()
    t = Tactic('tseitin-cnf')
    x_, y_ = data_list[0]['x'], data_list[0]['y']
    x_ = [data_multi(v) for v in x_]
    y_ = [v * MULTI for v in y_]

    encoder = ConstraintEncoder(MODEL_PATH, SYSTEM_NAME, FEATURE_NAMES, OUT_LENGTH, multi=MULTI)
    in_vars, out_vars = encoder.get_vars(0)

    # Discrete value contraints
    discrete_vars, discrete_values = [in_vars[f] for f in DISCRETE_VALUES.keys()], list(
        DISCRETE_VALUES.values())

    # generate perturbation features' set
    p_idxs = generate_idxs(pfs, historys)
    perturb_vars, normal_vars = split_list_by_idxs(in_vars, p_idxs)
    perturb_values, normal_values = split_list_by_idxs(x_, p_idxs)
    perturb_ranges = [FEATURE_RANGES[idx] for idx in p_idxs]

    def reset_constraints(epsilon):
        # Get perturbation boundary for special features
        perturb_bounds = []
        for p_idx, v, k in zip(p_idxs, perturb_values, perturb_ranges):
            if p_idx in DISCRETE_VALUES.keys():
                tmp = (map_to_closest(data_round(v - k * epsilon), DISCRETE_VALUES[p_idx]),
                       map_to_closest(data_round(v + k * epsilon), DISCRETE_VALUES[p_idx]))
                perturb_bounds.append(tmp)
            else:
                perturb_bounds.append((data_round(v - k * epsilon), data_round(v + k * epsilon)))

        encoder.reset()
        encoder.discrete_constr(discrete_vars, discrete_values)
        encoder.boundary_constr(perturb_vars, perturb_bounds)
        encoder.assign_constr(normal_vars, normal_values)
        encoder.argmax_neq_constr(out_vars, np.argmax(y_))

        solver.reset()
        solver.add(encoder.get_in_out_constrs())
        solver.add(encoder.get_nn_constrs())
        g.add(encoder.get_in_out_constrs())
        g.add(encoder.get_nn_constrs())

    # multi-add searching
    lb, epsilon = 0, 0.05

    start_t = time.time()

    reset_constraints(epsilon)

    # i=0
    # with open('assert_minimal_attack_power.txt','w',encoding='utf-8') as f:#使用with open()新建对象f
    #     for c in solver.assertions():
    #         i=i+1
    #         f.write(str(i)+":"+str(c)+"\n ##############################################\n")
 
    # print('Searching minimal attack power...')
    # clauses = t(g)
    # i=0
    # with open('clause_minimal_attack_power.txt','w',encoding='utf-8') as f:#使用with open()新建对象f
    #     for clause in clauses[0]:
    #         i=i+1
    #         f.write(str(i)+":"+str(clause)+"\n")
    
    it_st = time.time()
    count = 0
    while solver.check() == unsat:
        count += 1
        lb = epsilon
        epsilon *= 2
        print(f'Count: {count} | Epsilon: {epsilon: .8f} | Iteration time: {time.time() - it_st: .2f}s')

        reset_constraints(epsilon)

        it_st = time.time()
    ub = epsilon

    # binary search
    while ub - lb > precision:
        epsilon = (ub + lb) / 2

        reset_constraints(epsilon)

        if solver.check() == unsat:
            lb = epsilon
        else:
            ub = epsilon
        count += 1
        print(f'Refined attack power bounds: ({lb: .8f}, {ub: .8f})')

    total_t = time.time() - start_t
    print(f'The minimal attack power is {epsilon: .8f}, time: {total_t: .2f}s')

    return epsilon, total_t

def decision_boundaries(x_, expect_label, feature1, feature2, log_path, total=50):
    print('******* Searching Decision Boundaries *******')

    solver = SolverFor('LRA')
    g = Goal()
    t = Tactic('tseitin-cnf')
    x_ = [data_multi(v) for v in x_]

    encoder = ConstraintEncoder(MODEL_PATH, SYSTEM_NAME, FEATURE_NAMES, OUT_LENGTH,multi=MULTI)
    in_vars, out_vars = encoder.get_vars(0)

    name1, name2 = str(in_vars[feature1]).split('$')[0], str(in_vars[feature2]).split('$')[0]
    file = open(log_path, 'w', newline='')
    writer = csv.writer(file)
    writer.writerow([name1, name2, 'Time'])

    bound_vars, assign_vars = split_list_by_idxs(in_vars, [feature1, feature2])
    _, assign_values = split_list_by_idxs(x_, [feature1, feature2])
    bounds = [FEATURE_BOUNDS[feature1], FEATURE_BOUNDS[feature2]]

    # Discrete value contraints
    discrete_vars, discrete_values = [in_vars[f] for f in DISCRETE_VALUES.keys()], list(
        DISCRETE_VALUES.values())
    encoder.discrete_constr(discrete_vars, discrete_values)
    encoder.assign_constr(assign_vars, assign_values)
    encoder.boundary_constr(bound_vars, bounds)
    encoder.argmax_eq_constr(out_vars, expect_label)

    solver.add(encoder.get_in_out_constrs())
    solver.add(encoder.get_nn_constrs())
    g.add(encoder.get_in_out_constrs())
    g.add(encoder.get_nn_constrs())
    count = 0
    start_t = time.time()
    ################
    # i=0
    # with open('assert_decision_boundaries.txt','w',encoding='utf-8') as f:
    #     for c in solver.assertions():
    #         i=i+1
    #         f.write(str(i)+":"+str(c)+"\n ##############################################\n")
    # clauses = t(g)
    # i=0
    # with open('clause_decision_boundaries.txt','w',encoding='utf-8') as f:
    #     for clause in clauses[0]:
    #         i=i+1
    #         f.write(str(i)+":"+str(clause)+"\n")
    ###############

    while count < total and solver.check() == sat:
        print("sat")
        values = find_solution(solver, bound_vars)
        values = [data_multi(v, reverse=True) for v in values]
        total_t = time.time() - start_t
        writer.writerow([values[0], values[1], total_t])
        print(f'Solution: [{name1} = {values[0]}, {name2} = {values[1]}] | Time: {total_t: .2}s')
        file.flush()

        start_t = time.time()
        count += 1
    file.close()

def counterfactual_example(x_, expect_label, precision=10):

    print(f'****** Search counterfactual example, expect label = {expect_label} ******')

    solver = SolverFor('LRA')
    g=Goal()
    t = Tactic('tseitin-cnf')
    x_ = [data_multi(v) for v in x_]

    encoder = ConstraintEncoder(MODEL_PATH, SYSTEM_NAME, FEATURE_NAMES, OUT_LENGTH,multi=MULTI)
    in_vars, out_vars = encoder.get_vars(0)
    print(out_vars)

    lb, ub = 0., 0.
    for v, bound, r in zip(x_, FEATURE_BOUNDS, FEATURE_RANGES):
        if v - bound[0] > bound[1] - v:
            ub += (v - bound[0]) / r
        else:
            ub += (bound[1] - v) / r

    constr = RealVal(0)
    for x, v, r in zip(in_vars, x_, FEATURE_RANGES):
        constr = constr + If(x > v, (x - v) / r, (v - x) / r)

    # Discrete value contraints
    discrete_vars, discrete_values = [in_vars[f] for f in DISCRETE_VALUES.keys()], list(
        DISCRETE_VALUES.values())
    encoder.discrete_constr(discrete_vars, discrete_values)
    encoder.boundary_constr(in_vars, FEATURE_BOUNDS)
    encoder.argmax_eq_constr(out_vars, expect_label)
    solver.add(encoder.get_in_out_constrs())
    solver.add(encoder.get_nn_constrs())
    g.add(encoder.get_in_out_constrs())
    g.add(encoder.get_nn_constrs())
    solver.push()
    solver.add(constr < ub)
    g.add(constr<ub)
    # i=0
    # with open('assert_counterfactual_example.txt','w',encoding='utf-8') as f:#使用with open()新建对象f
    #     for c in solver.assertions():
    #         i=i+1
    #         f.write(str(i)+":"+str(c)+"\n ##############################################\n")
    # clauses = t(g)
    # i=0
    # with open('clause_counterfactual_example.txt','w',encoding='utf-8') as f:#使用with open()新建对象f
    #     for clause in clauses[0]:
    #         i=i+1
    #         f.write(str(i)+":"+str(clause)+"\n")
    start_t = time.time()
    while ub - lb > precision:
        print(f'bound: ({lb: .6f}, {ub: .6f})')
        mid = (lb + ub) / 2

        solver.pop()
        solver.push()
        solver.add(constr < mid)
       
        if solver.check() == sat:
            ub = mid
        else:
            lb = mid

    names = [str(x).split('$')[0] for x in in_vars]

    solver.pop()
    solver.add(constr < ub)
    solver.check()
    values = find_solution(solver, in_vars, all=False)
    values = [data_multi(v, reverse=True) for v in values]
    total_t = time.time() - start_t
    return names, values, total_t

def sensitivity_analysis(pf,history):
    
    solver=SolverFor('LRA')
    x_, y_ = data_list[0]['x'], data_list[0]['y']
    x_ = [data_multi(v) for v in x_]
    y_ = [v * MULTI for v in y_]

    encoder = ConstraintEncoder(MODEL_PATH, SYSTEM_NAME, FEATURE_NAMES, OUT_LENGTH,multi=MULTI)
    in_vars, out_vars = encoder.get_vars(0)

    p_idxs = generate_idxs(pf, history)
    perturb_vars, normal_vars = split_list_by_idxs(in_vars, p_idxs)
    perturb_values, normal_values = split_list_by_idxs(x_, p_idxs)
    perturb_ranges = [FEATURE_RANGES[idx] for idx in p_idxs]
    epsilon = 5e-5
    perturb_bounds = [(data_round(v - k * epsilon), data_round(v + k * epsilon))
                          for v, k in zip(perturb_values, perturb_ranges)]

    encoder.boundary_constr(perturb_vars, perturb_bounds)
    encoder.assign_constr(normal_vars, normal_values)
    solver.add(encoder.get_in_out_constrs())
    solver.add(encoder.get_nn_constrs())
    
    # 多次sat得到output值
    start_t = time.time()
    count = 0
    my_list = [] #保留仅pfi这一特征干扰下多组赋值
    #保存output3的概率结果
    output3 = []
    select = [] #干扰特征值

    while count < 300 and solver.check()== sat:
        values = [] #记录不可以重复的values值
        new_constraints = []
        solution = solver.model()
        select.append(eval(str(solution[perturb_vars[0]]))) #####
        output3.append(eval(str(solution[out_vars[3]])))
     
        for x in perturb_vars:
            values.append(eval(str(solution[x])))
            new_constraints.append(x != solution[x])
            #values = [data_multi(v, reverse=True) for v in values]
       
        solver.add(And(new_constraints))
        count+=1
        my_list.append(solution)
 
    total_t = time.time()-start_t
    ####### 保存sa的数据
    with open(f'sa1/sa_pensieve_output3_{perturb_vars[0]}.txt', 'w') as f:
        for i in range(len(output3)):
            #print(float(output3[i] - y_[3]))
            f.writelines(str(select[i])+', '+str(output3[i])+', '+str(float(output3[i] - y_[3]))+'\n')
    f.close()

    return my_list,total_t


if __name__ == '__main__':

 
    ###################  Check Pensieve encoding #############################
    # 验证SMT model encodig
    solver = SolverFor('LRA')
    
    x_, y_ = data_list[0]['x'], data_list[0]['y']
    x_ = [data_multi(v) for v in x_]
    y_ = [v * MULTI for v in y_]
    print(np.argmax(y_))
    encoder = ConstraintEncoder(MODEL_PATH, SYSTEM_NAME, FEATURE_NAMES, OUT_LENGTH, multi=MULTI)
    in_vars, out_vars = encoder.get_vars(0)
    print(y_)
    encoder.assign_constr(in_vars, x_)
    # encoder.argmax_eq_constr(out_vars, np.argmax(y_))
    encoder.approximate_eq_constr(out_vars,y_)

    solver.add(encoder.get_in_out_constrs())
    solver.add(encoder.get_nn_constrs())
    print(solver.assertions())
    print(solver.check())

    ######################  Adversarial Pertubation ##############################
    log_path = f'{BASE_DIR}/logs/adversarial_pensieve.csv'
    file = open(log_path, 'w', newline='')
    writer = csv.writer(file)
    writer.writerow(['Features', 'Historys', 'Epsilon', 'Result', 'Time'])
    # perturbation feature's index
    p_features = [
        0,
        1,
        2,
        3,
        4,
        5,
        [0, 1, 2, 3, 4, 5]
    ]
    
    historys_list = [
        1,
        1,
        1,
        1,
        1,
        1,
        [1, 1, 1, 1, 1, 1],
    ]
    
    #epsilons = [0.016 * (i + 1) for i in range(20)]
    epsilons = [0.016 * (i + 1) for i in range(5)]
    
    for features, historys in zip(p_features, historys_list):
        for epsilon in epsilons:
            res, total_t = adversarial_perturbation(features, historys, epsilon)
            writer.writerow([features, historys, epsilon, res, total_t])
            file.flush()
    file.close()

    ##################### Missing features ######################################
    log_path = f'{BASE_DIR}/logs/missing_features_pensieve.csv'
    file = open(log_path, 'w', newline='')
    writer = csv.writer(file)
    writer.writerow(['Features', 'Historys', 'Result', 'Time'])
    file.flush()
    
    # missing feature's index
    m_features = [i for i in range(FEATURE_TYPES)]
    
    historys_list = [
        1,  # for pensieve, each feature's history is not equal
        1,
        1,
        1,
        1,
        1
    ]
    
    for feature, historys in zip(m_features, historys_list):
        res, total_t = verify_missing_features(feature, historys)
        writer.writerow([feature, historys, res, total_t])
        file.flush()
    file.close()

    ########################### Verify extreme_values ##############################
    log_path = f'{BASE_DIR}/logs/extreme_values_pensieve.csv'
    file = open(log_path, 'w', newline='')
    writer = csv.writer(file)
    writer.writerow(['Historys', 'Result', 'Time'])
    file.flush()
    historys = [8-i for i in range(8)]
    for i in range(1, 8):
        res, t = verify_extreme_values()
        writer.writerow([i, res, t])

    ########################## Verify decision boundary ############################
    verify_decision_boundary(f'{BASE_DIR}/logs/decision_boundary_pensieve.csv')

    # ########################### Find minimal attack power ##########################
    # log_path = f'{BASE_DIR}/logs/minimal_epsilon_pensieve.csv'
    # file = open(log_path, 'w', newline='')
    # writer = csv.writer(file)
    # writer.writerow(['Features', 'Historys', 'Epsilon', 'Time'])
    # file.flush()
    # pfs = [
    #     0,
    #     1,
    #     2,
    #     3,
    #     4,
    #     5,
    #     [0, 1, 2, 3, 4, 5]
    # ]
    # historys_list = [
    #     1,
    #     1,
    #     1,
    #     1,
    #     1,
    #     1,
    #     [1, 1, 1, 1, 1, 1]
    # ]
    # for i in range(len(pfs)):
    #     epsilon,total_t = find_minimal_attack_power(pfs[i], historys_list[i])
    #     writer.writerow([pfs[i], historys_list[i], epsilon, total_t])
    #     file.flush()
    
    # file.close()

    # ################# Decision boundary interpretability ############################
    # x_, expect_label = data_list[0]['x'], 3
    # # print(x_)
    # decision_boundaries(x_, expect_label, 1, 2, f'{BASE_DIR}/logs/decision_boundaries_pensieve.csv')

    # ################ Qualitatively Anchor interpretability ############################
    # file = open(f'{BASE_DIR}/logs/qualitatively_anchor_{SYSTEM_NAME}_t.csv', 'a', newline='')
    # writer = csv.writer(file)
    # x_, expect_label = data_list[0]['x'], 3
    # y=data_list[0]["y"]
    # features, total_t = qualitatively_anchor_t(x_, y)
    # features.append(total_t)
    # writer.writerow(features)
    # file.flush()
    
    # file.close()

    # ################# Counterfactual Example interpretability ##########################
    # file = open(f'{BASE_DIR}/logs/counterfactual_pensieve.csv', 'w', newline='')

    # x_, expect_label = data_list[0]['x'], 1
    # writer = csv.writer(file)
    # names, values, total_t = counterfactual_example(x_, 2)
    # names.append('Time')
    # values.append(total_t)
    # writer.writerow([names])
    # writer.writerow([values])
    # file.close()
    # #################################Sensitivity Analysis##############################
    # file = open(f'{BASE_DIR}/logs/sensitivity_{SYSTEM_NAME}_origin_new.csv', 'w', newline='')
    # writer = csv.writer(file)
    # #x_, y_ = data_list[0]['x'], data_list[0]['y']
    # pfs = [
    # #    0
    # #    1,
    # #    2,
    # #    3,
    #     4,
    #     5
    # ]
    # historys_list = [
    # #    1,
    # #    1,
    # #    1,
    # #    1,
    #     1,
    #     1
    # ]
    # for i in range(len(pfs)):
    #     res,total_t = sensitivity_analysis(pfs[i], historys_list[i])
    #     #writer.writerow([pfs[i], historys_list[i], res, total_t])
    #     #file.flush()
    # file.close()

    


