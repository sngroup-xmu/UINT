import csv
import json
import time
import argparse

from z3 import sat, unsat, And, Or, If, RealVal, SolverFor
from z3 import *
import sys
sys.path.append("../..") 
from src import ConstraintEncoder, split_list_by_idxs, find_solution
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--aurora-version', type=int, default='0')
parser.add_argument('--system-precision', type=float, default=0.000001) #0.01 0.00001 sa #0.000001
#parser.add_argument('--system-precision', type=float, default=0.01) 
parser.add_argument('--is-multi', type=bool, default=True)
args = parser.parse_args()

BASE_DIR = '../..'
SYSTEM_NAME=f'aurora{args.aurora_version}'
MODEL_PATH = f'{BASE_DIR}/model_file/aurora/{SYSTEM_NAME}.pb'
dataset = json.load(open(f'{BASE_DIR}/model_file/aurora/{SYSTEM_NAME}_dataset.json', 'r'))

IS_MULTI = args.is_multi
PRECISION = args.system_precision
if IS_MULTI:
    MULTI = int(1 / PRECISION)
else:
    MULTI = 1.0

def data_round(value):
    if IS_MULTI:
        return round(value)
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

OUT_LENGTH = 1
HISTORY_LEN = 10
FEATURE_TYPES = int(dataset['types'])
FEATURE_NAMES = dataset['names']
FEATURE_BOUNDS = [(data_multi(lb), data_multi(ub))for lb, ub in dataset['bounds']]
FEATURE_RANGES = [bound[1] - bound[0] for bound in FEATURE_BOUNDS]


data_list = dataset['data']

#def generate_idxs(features, history=HISTORY_LEN, pos='R'):
def generate_idxs(features, history=HISTORY_LEN, pos='L'):

    if pos == 'R':
        return [i * FEATURE_TYPES + f for i in range(HISTORY_LEN - history, HISTORY_LEN) for f in features]
    else:
        return [i * FEATURE_TYPES + f for i in range(history) for f in features]

############################### Verification Problem ######################################
def adversarial_perturbation(pfs, history, epsilon):
    print('*********** Verify Adversarial Perturbation *********')
    print(f'perturbation features: {pfs}, history: {history}')

    solver = SolverFor('LRA')

    x_, y_ = data_list[0]['x'], data_list[0]['y']
    x_ = [data_multi(v) for v in x_]
    y_ = [v * MULTI for v in y_]

    encoder = ConstraintEncoder(MODEL_PATH, SYSTEM_NAME, FEATURE_NAMES, OUT_LENGTH, multi=MULTI)

    input_vars, out_vars = encoder.get_vars(0)

    # Generate perturbation features, values, bounds and ranges
    p_idxs = generate_idxs(pfs, history)
    perturb_vars, normal_vars = split_list_by_idxs(input_vars, p_idxs)
    perturb_values, normal_values = split_list_by_idxs(x_, p_idxs)
    perturb_ranges = [FEATURE_RANGES[idx] for idx in p_idxs]

    perturb_bounds = [(data_round(v - k * epsilon), data_round(v + k * epsilon))
                      for v, k in zip(perturb_values, perturb_ranges)]
    encoder.boundary_constr(perturb_vars, perturb_bounds)
    encoder.assign_constr(normal_vars, normal_values)
    encoder.approximate_neq_constr(out_vars, y_)

    # Add constraints to Z3
    solver.add(encoder.get_in_out_constrs())
    solver.add(encoder.get_nn_constrs())

    start_t = time.time()
    res = "able" if solver.check() == unsat else "unable"
    total_t = time.time() - start_t

    print(f'Epsilon: {epsilon} | Result: {res} | Time: {total_t} \n')

    return res, total_t

def verify_extreme_values(history=10):

    print('*********** Verify Extreme Values *********')

    solver = SolverFor('LRA')

    encoder = ConstraintEncoder(MODEL_PATH, SYSTEM_NAME, FEATURE_NAMES, OUT_LENGTH, multi=MULTI)

    in_vars, out_vars = encoder.get_vars(0)

    # Generate extreme features and bounds
    e_idxs = generate_idxs([1], history)
    e_bounds = []
    for idx in range(len(in_vars)):
        if idx in e_idxs:
            e_bounds.append((data_multi(0.99), data_multi(1.0)))
        else:
            e_bounds.append(FEATURE_BOUNDS[idx])

    encoder.boundary_constr(in_vars, e_bounds)
    encoder.approximate_eq_constr(out_vars, [0])

    solver.add(encoder.get_in_out_constrs())
    solver.add(encoder.get_nn_constrs())


    start_t = time.time()
    res = 'unsat' if solver.check() == sat else 'sat'
    total_t = time.time() - start_t

    print(f'History: {history} | Result: {res} | Time: {total_t} \n')
    return res, total_t

def verify_missing_features(mfs, history):

    print('******* Verify Missing Features *******')

    solver = SolverFor('LRA')

    x_, y_ = data_list[0]['x'], data_list[0]['y']
    x_ = [data_multi(v) for v in x_]
    y_ = [v * MULTI for v in y_]

    encoder = ConstraintEncoder(MODEL_PATH, SYSTEM_NAME, FEATURE_NAMES, OUT_LENGTH, multi=MULTI)
    in_vars, out_vars = encoder.get_vars(0)

    m_idxs = generate_idxs(mfs, history)
    missing_vars, normal_vars = split_list_by_idxs(in_vars, m_idxs)
    _, normal_values = split_list_by_idxs(x_, m_idxs)
    missing_bounds = [FEATURE_BOUNDS[idx] for idx in m_idxs]

    encoder.boundary_constr(missing_vars, missing_bounds)
    encoder.assign_constr(normal_vars, normal_values)
    encoder.approximate_neq_constr(out_vars, y_)

    solver.add(encoder.get_in_out_constrs())
    solver.add(encoder.get_nn_constrs())

    start_t = time.time()
    res = "keep" if solver.check() == unsat else "don't keep"
    total_t = time.time() - start_t
    print(f'Missing feature: {mfs} | history: {history} | Result: {res} | Time: {total_t}')

    return res, total_t

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

    encoder = ConstraintEncoder(MODEL_PATH, SYSTEM_NAME, FEATURE_NAMES, OUT_LENGTH, multi=MULTI)
    in_vars, out_vars = encoder.get_vars(0)

    # Generate feature pairs
    pairs = []
    for i1 in range(FEATURE_TYPES - 1):
        for i2 in range(HISTORY_LEN):
            f1_idx = i2 * FEATURE_TYPES + i1
            for j1 in range(i1 + 1, FEATURE_TYPES):
                for j2 in range(HISTORY_LEN):
                    f2_idx = j2 * FEATURE_TYPES + j1
                    pairs.append([f1_idx, f2_idx])

    for pair in pairs:
        bounds = [FEATURE_BOUNDS[idx] for idx in pair]

        missing_vars, normal_vars = split_list_by_idxs(in_vars, pair)
        _, normal_values = split_list_by_idxs(x_, pair)

        encoder.reset()
        encoder.boundary_constr(missing_vars, bounds)
        encoder.assign_constr(normal_vars, normal_values)
        encoder.approximate_neq_constr(out_vars, y_)

        solver.reset()
        solver.add(encoder.get_in_out_constrs())
        solver.add(encoder.get_nn_constrs())
        start_t = time.time()
        res = 'keep' if solver.check() == unsat else "don't keep"
        total_t = time.time() - start_t

        print(f'Features: {pair}, Result: {res}, Time: {total_t}')
        writer.writerow([pair, res, total_t])
        file.flush()

    file.close()


################################# Interpretability Problem ####################################

def find_minimal_attack_power(pf, history, precision=1e-6):

    print('***** Minimal Attack Power ******')
    print(f'Perturbation Features: {pf} | History: {history}')

    solver = SolverFor('LRA')
    g=Goal()
    t = Tactic('tseitin-cnf')
    x_, y_ = data_list[0]['x'], data_list[0]['y']
    x_ = [data_multi(v) for v in x_]
    y_ = [v * MULTI for v in y_]

    encoder = ConstraintEncoder(MODEL_PATH, SYSTEM_NAME, FEATURE_NAMES, OUT_LENGTH, multi=MULTI)
    in_vars, out_vars = encoder.get_vars(0)

    # Generate perturbation features' set
    p_idxs = generate_idxs(pf, history)
    perturb_vars, normal_vars = split_list_by_idxs(in_vars, p_idxs)
    perturb_values, normal_values = split_list_by_idxs(x_, p_idxs)
    perturb_ranges = [FEATURE_RANGES[idx] for idx in p_idxs]

    def reset_constraints(epsilon):
        perturb_bounds = [(data_round(v - k * epsilon), data_round(v + k * epsilon))
                          for v, k in zip(perturb_values, perturb_ranges)]

        encoder.reset()
        encoder.boundary_constr(perturb_vars, perturb_bounds)
        encoder.assign_constr(normal_vars, normal_values)
        encoder.approximate_neq_constr(out_vars, y_)

        solver.reset()
        solver.add(encoder.get_in_out_constrs())
        solver.add(encoder.get_nn_constrs())
        g.add(encoder.get_in_out_constrs())
        g.add(encoder.get_nn_constrs())
    lb, epsilon = 0, 5e-5
    start_t = time.time()
    reset_constraints(epsilon)

    # 保存SMT求解器的assertion
    #i=0
    # with open('assert_aurora_minimal_attack_power.txt','w',encoding='utf-8') as f:
    #     for c in solver.assertions():
    #         i=i+1
    #         f.write(str(i)+":"+str(c)+"\n ##############################################\n")
    # clauses = t(g)
    # 保存SMT求解器的子句
    # i=0
    # with open('clause_aurora_minimal_attack_power.txt','w',encoding='utf-8') as f:
    #     for clause in clauses[0]:
    #         i=i+1
    #         f.write(str(i)+":"+str(clause)+"\n")
    
    print('Searching minimal attack power...')
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
    print(f'The perturbed feature: {epsilon: .8f}')
    print(f'The minimal attack power is {epsilon: .8f}, time: {total_t: .2f}s')

    return epsilon, total_t

def qualitatively_anchor(x, y):
    print(f'******** Qualitatively Anchor *******')

    x = [data_multi(v) for v in x]
    y = [v * MULTI for v in y]
    print(y)
    solver = SolverFor('LRA')
    solver.set(unsat_core=True)
    solver.set("smt.core.minimize", True)

    encoder = ConstraintEncoder(MODEL_PATH, SYSTEM_NAME, FEATURE_NAMES, OUT_LENGTH,multi=MULTI)
    in_vars, out_vars = encoder.get_vars(0)

    encoder.assign_track(in_vars, x)
    encoder.approximate_neq_constr(out_vars, y)

    tracks = encoder.get_tracks()
    for name, expr in tracks.items():
        solver.assert_and_track(expr, name)

    solver.add(encoder.get_in_out_constrs())
    solver.add(encoder.get_nn_constrs())
    i=0
    with open('expr_aurora_qualitatively_anchor.txt','w',encoding='utf-8') as f:#使用with open()新建对象f
        f.write(solver.sexpr())
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

def decision_boundaries(x_, y_, feature1, feature2, log_path, total=10000):

    print('******* Searching Decision Boundaries *******')

    solver = SolverFor('LRA')
    g=Goal()
    t = Tactic('tseitin-cnf')
    x_ = [data_multi(v) for v in x_]
    y_ = [v * MULTI for v in y_]

    encoder = ConstraintEncoder(MODEL_PATH, SYSTEM_NAME, FEATURE_NAMES, OUT_LENGTH,multi=MULTI)
    in_vars, out_vars = encoder.get_vars(0)

    name1, name2 = str(in_vars[feature1]).split('$')[0], str(in_vars[feature2]).split('$')[0]
    file = open(log_path, 'w', newline='')
    writer = csv.writer(file)
    writer.writerow([name1, name2, 'Time'])

    bound_vars, assign_vars = split_list_by_idxs(in_vars, [feature1, feature2])
    _, assign_values = split_list_by_idxs(x_, [feature1, feature2])
    bounds = [FEATURE_BOUNDS[feature1], FEATURE_BOUNDS[feature2]]

    encoder.assign_constr(assign_vars, assign_values)
    encoder.boundary_constr(bound_vars, bounds)
    encoder.approximate_eq_constr(out_vars, y_)

    solver.add(encoder.get_in_out_constrs())
    solver.add(encoder.get_nn_constrs())
    g.add(encoder.get_in_out_constrs())
    g.add(encoder.get_nn_constrs())
    count = 0

    # i=0
    # with open('assert_aurora_decision_boundaries.txt','w',encoding='utf-8') as f:
    #     for c in solver.assertions():
    #         i=i+1
    #         f.write(str(i)+":"+str(c)+"\n ##############################################\n")
    # clauses = t(g)
    # i=0
    # with open('clause_aurora_decision_boundaries.txt','w',encoding='utf-8') as f:
    #     for clause in clauses[0]:
    #         i=i+1
    #         f.write(str(i)+":"+str(clause)+"\n")
    
    start_t = time.time()
    while count < total and solver.check() == sat:
        values = find_solution(solver, bound_vars)
        values = [data_multi(v, reverse=True) for v in values]
        total_t = time.time() - start_t
        writer.writerow([values[0], values[1], total_t])
        print(f'Solution: [{name1} = {values[0]}, {name2} = {values[1]}] | Time: {total_t: .2}s')
        file.flush()

        start_t = time.time()
        count += 1
    file.close()

def counterfactual_example(x_, ye, precision=0.1):

    print(f'****** Search counterfactual example, ye = {ye[0]} ******')

    solver = SolverFor('LRA')
    g=Goal()
    t = Tactic('tseitin-cnf')
    x_ = [data_multi(v) for v in x_]
    ye = [v * MULTI for v in ye]

    encoder = ConstraintEncoder(MODEL_PATH, SYSTEM_NAME, FEATURE_NAMES, OUT_LENGTH,multi=MULTI)
    in_vars, out_vars = encoder.get_vars(0)

    lb, ub = 0., 0.
    for v, bound, r in zip(x_, FEATURE_BOUNDS, FEATURE_RANGES):
        if v - bound[0] > bound[1] - v:
            ub += (v - bound[0]) / r
        else:
            ub += (bound[1] - v) / r


    constr = RealVal(0)
    for x, v, r in zip(in_vars, x_, FEATURE_RANGES):
        constr = constr + If(x > v, (x - v) / r, (v - x) / r)

    encoder.boundary_constr(in_vars, FEATURE_BOUNDS)
    encoder.approximate_eq_constr(out_vars, ye)
    solver.add(encoder.get_in_out_constrs())
    solver.add(encoder.get_nn_constrs())
    g.add(encoder.get_in_out_constrs())
    g.add(encoder.get_nn_constrs())
    solver.push()
    solver.add(constr < ub)
    g.add(constr<ub)

    # i=0
    # with open('assert_aurora_counterfactual.txt','w',encoding='utf-8') as f:#使用with open()新建对象f
    #     for c in solver.assertions():
    #         i=i+1
    #         f.write(str(i)+":"+str(c)+"\n ##############################################\n")
    # clauses = t(g)
    # i=0
    # with open('clause_aurora_counterfactual.txt','w',encoding='utf-8') as f:#使用with open()新建对象f
    #     for clause in clauses[0]:
    #         i=i+1
    #         f.write(str(i)+":"+str(clause)+"\n")
    
    start_t = time.time()
    precision *= MULTI
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

# def minimal_change():
#     solver = Z3Solver(FEATURE_NAMES, weights, biases, activ, 1, unsat_core=True, k=2)
#     X1, ye1, X2, ye2 = solver.in_vars[0], solver.pre_out_vars[0], solver.in_vars[1], solver.pre_out_vars[1]
#     solver.add_track_assign_constraints(X1, x_)
#     solver.add_track_assign_constraints(X2, x_hat)
#     solver.add_track_eq(X1, X2)
#     solver.add(solver.continue_eq(ye1, y_hat))
#
#     start_t = time.time()
#     if solver.check() == unsat:
#         print('unsat')
#         CoMSS = solver.solver.unsat_core()
#         features = [str(x).split('$')[0] for x in CoMSS]
#         print(CoMSS)
#     else:
#         print('sat')
#         features = []

#metric:单次扰动后的输出与原值的偏离程度
def sensitivity_analysis(pf,history):
    solver=SolverFor('LRA')
    x_, y_ = data_list[0]['x'], data_list[0]['y']
    x_ = [data_multi(v) for v in x_]
    y_ = [v * MULTI for v in y_]

    encoder = ConstraintEncoder(MODEL_PATH, SYSTEM_NAME, FEATURE_NAMES, OUT_LENGTH,multi=MULTI)
    in_vars, out_vars = encoder.get_vars(0)
    # name1, name2 = str(in_vars[feature1]).split('$')[0], str(in_vars[feature2]).split('$')[0]
    # file = open(log_path, 'w', newline='')
    # writer = csv.writer(file)
    # writer.writerow([name1, name2, 'Time'])
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

    start_t = time.time()

    #使用方法sexpr()提取z3的表达式的内部表示
    # with open('assert_aurora_sensitivity_analysis.txt','w',encoding='utf-8') as f:
    #     f.write(solver.sexpr())

    print(solver.check())
    my_list=solver.model()
    total_t = time.time()-start_t
    return my_list,total_t

#metric:多次重复下输出与原值偏离的波动范围
def sensitivity_analysis_new(pf,history):
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
    
    # 多次sat得到output值取avg
    start_t = time.time()
    count = 0
    output = [] #存储多次重复求解时的不同输出
    sum = 1200 #重复次数
    select = [] #存储已经求解得到的solution内的扰动特征赋值
    while count < sum and solver.check()== sat:
        new_constraints = []
        solution = solver.model()
        output.append(eval(str(solution[out_vars[0]])))
        #增加限制，保证每次solver返回的解不同
        for x in perturb_vars:
            select.append(eval(str(solution[x])))
            new_constraints.append(x != solution[x])
        solver.add(And(new_constraints))
        count+=1
        
    total_t = time.time()-start_t
    
    with open(f'SA_new/auraro_{perturb_vars[0]}.txt', 'w') as f:
        for i in range(len(output)):
            f.writelines(str(select[i])+', '+str(output[i])+', '+str(float(output[i] - y_[0]))+'\n')
    f.close()
  
    return output,total_t


if __name__ == '__main__':

    #################################  Adversarial Perturbation ####################################
    # file = open(f'{BASE_DIR}/logs/adversarial_{SYSTEM_NAME}.csv', 'w', newline='')
    # writer = csv.writer(file)
    # writer.writerow(['Features', 'History', 'Epsilon', 'Result', 'Time'])
    # file.flush()
    # m_features = [
    #     [0],
    #     [1],
    #     [2],
    #     [0, 1, 2],
    # ]

    # historys = [10, 5, 3]

    # epsilons = [0.00025 * (i + 1) for i in range(20)]
    # for epsilon in epsilons:
    #     for  history in historys:
    #         for features in m_features:
    #             res, total_t = adversarial_perturbation(features, history, epsilon)
    #             writer.writerow([features, history, epsilon, res, total_t])
    #             file.flush()
    # file.close()

    ###################################  Missing features ####################################
    # m_features = [
    #     [0],
    #     [1],
    #     [2]
    # ]
    # historys = [1, 3, 5, 10]

    # file = open(f'{BASE_DIR}/logs/missing_features_{SYSTEM_NAME}.csv', 'w', newline='')
    # writer = csv.writer(file)
    # writer.writerow(['Features', 'History', 'Result', 'Time'])
    # file.flush()

    # for history in historys:
    #     for mf in m_features:
    #         res, total_t = verify_missing_features(mf, history)
    #         writer.writerow([mf, history, res, total_t])
    #         file.flush()
    # file.close()

    ###############################  Extreme values #############################################
    file = open(f'{BASE_DIR}/logs/extreme_values_{SYSTEM_NAME}.csv', 'w', newline='')
    writer = csv.writer(file)
    writer.writerow(['History', 'Result', 'Time'])
    file.flush()

    for history in [1, 3, 5, 10]:
        res, total_t = verify_extreme_values(history)
        writer.writerow([history, res, total_t])
        file.flush()
    file.close()

    ####################### Verify decision boundary ######################################
    verify_decision_boundary(f'{BASE_DIR}/logs/decision_boundary_{SYSTEM_NAME}.csv')

    
    # # ################### Feature Importance —— Find Minimal Attack Power ###########################
    # p_features = [
    #     [0]
    # #    [1],
    # #    [2],
    # #    [0, 1, 2]
    # ]

    # historys = [
    # #    5, 
    #     10]

    # file = open(f'{BASE_DIR}/logs/minimal_attack_{SYSTEM_NAME}.csv', 'w', newline='')
    # writer = csv.writer(file)
    # writer.writerow(['Features', 'History', 'Epsilon', 'Time'])
    # file.flush()

    # for history in historys:
    #     for pf in p_features:
    #         epsilon, total_t = find_minimal_attack_power(pf, history)
    #         writer.writerow([pf, history, epsilon, total_t])
    #         file.flush()
    # file.close()

    # # # # ################## Decision boundary interpretability ############################
    # x_, y_ = data_list[0]['x'], data_list[0]['y']
    # decision_boundaries(x_, y_, 0, 1, f'{BASE_DIR}/logs/decision_boundaries_{SYSTEM_NAME}_1.csv')
    # decision_boundaries(x_, [2.0], 0, 1, f'{BASE_DIR}/logs/decision_boundaries_{SYSTEM_NAME}_2.csv')

    # ############################## Qualitatively Anchor Interpretability #################
    # file = open(f'{BASE_DIR}/logs/qualitatively_anchor_{SYSTEM_NAME}.csv', 'a', newline='')
    # writer = csv.writer(file)
    # for i in range(len(data_list)):
    #     x_, y_ = data_list[i]['x'], data_list[i]['y']
    #     features, total_t = qualitatively_anchor(x_, y_)
    #     features.append(total_t)
    #     writer.writerow(features)
    #     file.flush()

    # ################# Conterfactual Example ###########################
    # file = open(f'{BASE_DIR}/logs/counterfactual_{SYSTEM_NAME}_origin.csv', 'w', newline='')
    # writer = csv.writer(file)
    # print(data_list[1]['x'])
    # x_ = data_list[1]['x']
    # names, values, total_t = counterfactual_example(x_, [0])
    # names.append('Time')
    # values.append(total_t)
    # writer.writerow([names])
    # writer.writerow([values])
    # file.close()

    # #################################Sensitivity Analysis##############################
    # # file = open(f'{BASE_DIR}/logs/sensitivity_{SYSTEM_NAME}.csv', 'w', newline='')
    # # writer = csv.writer(file)
    # pfs = [
    #     [0],
    #     [1],
    #     [2]
    # ]
    # historys_list = [
    #     5, #10
    #     5,
    #     5
    # ]
    # for i in range(len(pfs)):
    #     mylist,total_t = sensitivity_analysis_new(pfs[i], historys_list[i])
    #     #writer.writerow([pfs[i], historys_list[i], mylist, total_t])
    #     #file.flush()
    # #file.close()

    
