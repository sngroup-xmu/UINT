import csv
import json
import time
import argparse

import numpy as np
from z3 import sat, unsat, And, Or, RealVal, If, SolverFor
from z3 import *
import sys
sys.path.append("../..") 
from src import ConstraintEncoder, split_list_by_idxs, find_solution

parser = argparse.ArgumentParser()
parser.add_argument('--system-precision', type=float, default=0.001)
parser.add_argument('--is-multi', type=bool, default=True)
args = parser.parse_args()

BASE_DIR = '../..'
SYSTEM_NAME = 'decima'
MODEL_PATH = f'{BASE_DIR}/model_file/decima/{SYSTEM_NAME}.pb'
dataset = json.load(open(f'{BASE_DIR}/model_file/decima/{SYSTEM_NAME}1_dataset.json', 'r'))

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
HISTORY_LENS = dataset['historys']
FEATURE_TYPES = int(dataset['types'])
FEATURE_NAMES = dataset['names']
FEATURE_BOUNDS = [(data_multi(lb), data_multi(ub))for lb, ub in dataset['bounds']]
FEATURE_RANGES = [bound[1] - bound[0] for bound in FEATURE_BOUNDS]

PRE_BIT_POS = 0
REMAIN_THUNKS_POS = 24
# DISCRETE_VALUES = {
#     PRE_BIT_POS: [data_multi(v / 4300) for v in [300, 750, 1200, 1850, 2850, 4300]],
#     REMAIN_THUNKS_POS: [data_multi(v / 48) for v in range(48)],
# }

FEATURE_START_POS = [0,5,13,21,29] ###

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
def adversarial_perturbation(pfs, history, epsilon):
    print('*********** Verify Adversarial Perturbation Problem *********')
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

def verify_extreme_values(history=1):

    print('*********** Verify Extreme Values *********')

    solver = SolverFor('LRA')
    x_, y_ = data_list[0]['x'], data_list[0]['y']
    x_ = [data_multi(v) for v in x_]
    y_ = [v * MULTI for v in y_]
    encoder = ConstraintEncoder(MODEL_PATH, SYSTEM_NAME, FEATURE_NAMES, OUT_LENGTH, multi=MULTI)

    in_vars, out_vars = encoder.get_vars(0)
    e_idxs = generate_idxs([1], history)
    extreme_vars, normal_vars = split_list_by_idxs(in_vars, e_idxs)
    _, normal_values = split_list_by_idxs(x_, e_idxs)
    extreme_bounds = [(data_multi(0.4), data_multi(0.5))]

    encoder.boundary_constr(extreme_vars, extreme_bounds)
    encoder.assign_constr(normal_vars, normal_values)
    encoder.approximate_eq_constr(out_vars, [1])

    solver.add(encoder.get_in_out_constrs())
    solver.add(encoder.get_nn_constrs())


    start_t = time.time()
    res = 'error' if solver.check() == sat else 'right'
    total_t = time.time() - start_t

    print(f'History: {history} | Result: {res} | Time: {total_t} \n')
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
        missing_vars, normal_vars = split_list_by_idxs(in_vars, pair)
        _, normal_values = split_list_by_idxs(x_, pair)

        encoder.reset()
        encoder.boundary_constr(missing_vars, bounds)
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

def verify_liveness_property(k=2):

    DOWNLOAD_TIME = 0.4
    print('****** Verify Liveness Property ******')

    # Initialize
    solver = SolverFor('LRA')
    encoder = ConstraintEncoder(MODEL_PATH, SYSTEM_NAME, FEATURE_NAMES, OUT_LENGTH, K=k)
    in_vars_list, out_vars_list = encoder.get_vars()

    ###################### Discrete value contraints ##################
    discrete_values = list(DISCRETE_VALUES.values())
    for in_vars in in_vars_list:

        discrete_vars = [in_vars[f] for f in DISCRETE_VALUES.keys()]

        encoder.discrete_constr(discrete_vars, discrete_values)

    ################# Encode initial state ######################
    solver.add([in_vars_list[i][0] == data_multi(750 / 4300) for i in range(k)])
    [encoder.boundary_constr(in_vars, FEATURE_BOUNDS) for in_vars in in_vars_list]

    ############### Encode inappropriate states ###############
    and_constraints = []
    solver.add([in_vars_list[i][1] == data_multi(0.4 + ((4 - DOWNLOAD_TIME * 10) * (i)) / 10) for i in range(k)])
    for i in range(k):
        and_constraints.extend([in_vars_list[i][FEATURE_START_POS[3] + j] < DOWNLOAD_TIME for j in range(HISTORY_LENS[2])])
    solver.add(And(and_constraints))

    ############## Encode consective states ####################
    # Next throughtput equal to pre througntput shift by one
    and_constraints = []
    for i in range(k - 1):
        and_constraints.extend([in_vars_list[i][j+FEATURE_START_POS[2]+1] == in_vars_list[i + 1][j+FEATURE_START_POS[2]] for j in range(7)])
        and_constraints.extend([in_vars_list[i][j+FEATURE_START_POS[3]+1] == in_vars_list[i + 1][j+FEATURE_START_POS[3]] for j in range(7)])
    # Pre_bit_rate = pre NN(x)
    and_constraints.extend([in_vars_list[i][0] == data_multi(300 / 4300) for i in range(1, k)])
    for i in range(k - 1):
        and_constraints.extend([out_vars_list[i][0] >= out_vars_list[i][j] for j in range(1, OUT_LENGTH)])

    solver.add(And(and_constraints))

    ############## Encode circuit states #######################
    solver.add(Or([And([x1 == x2 for x1, x2 in zip(in_vars, in_vars_list[-1])]) for in_vars in in_vars_list[: -1]]))

    solver.add(encoder.get_in_out_constrs())
    solver.add(encoder.get_nn_constrs())

    start_t = time.time()
    res = "sat" if solver.check() == sat else "unsat"
    total_t = time.time() - start_t

    return res, total_t

def verify_safety_property(k=2):
    pass


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

    # solver.add_soft(encoder.assign_constr(in_vars,x))
    encoder.assign_track(in_vars, x)
    encoder.approximate_neq_constr(out_vars, y)
    # solver.add(encoder.argmax_neq_constr(out_vars, np.argmax(y)))
    
    tracks = encoder.get_tracks()
    for name, expr in tracks.items():
        solver.assert_and_track(expr, name)

    solver.add(encoder.get_in_out_constrs())
    solver.add(encoder.get_nn_constrs())
    i=0
    with open('expr_qualitatively_anchor_decima.txt','w',encoding='utf-8') as f:#使用with open()新建对象f
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
    i=0
    with open('expr_qualitatively_anchor.txt','w',encoding='utf-8') as f:#使用with open()新建对象f
        f.write(solver.sexpr())
    clauses = t(g)
    i=0
    with open('clause_qualitatively_anchor.txt','w',encoding='utf-8') as f:#使用with open()新建对象f
        for clause in clauses[0]:
            i=i+1
            f.write(str(i)+":"+str(clause)+"\n")
    
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
    i=0
    with open('assert_decima_minimal_attack_power.txt','w',encoding='utf-8') as f:#使用with open()新建对象f
        for c in solver.assertions():
            i=i+1
            f.write(str(i)+":"+str(c)+"\n ##############################################\n")
    clauses = t(g)
    i=0
    with open('clause_decima_minimal_attack_power.txt','w',encoding='utf-8') as f:#使用with open()新建对象f
        for clause in clauses[0]:
            i=i+1
            f.write(str(i)+":"+str(clause)+"\n")
    
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
    print(f'The minimal attack power is {epsilon: .8f}, time: {total_t: .2f}s')

    return epsilon, total_t

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
    i=0
    with open('assert_decima_decision_boundaries.txt','w',encoding='utf-8') as f:#使用with open()新建对象f
        for c in solver.assertions():
            i=i+1
            f.write(str(i)+":"+str(c)+"\n ##############################################\n")
    clauses = t(g)
    i=0
    with open('clause_decima_decision_boundaries.txt','w',encoding='utf-8') as f:#使用with open()新建对象f
        for clause in clauses[0]:
            i=i+1
            f.write(str(i)+":"+str(clause)+"\n")
    
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
        # print(v,bound,r)
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
    i=0
    with open('assert_decima_counterfactual.txt','w',encoding='utf-8') as f:#使用with open()新建对象f
        for c in solver.assertions():
            i=i+1
            f.write(str(i)+":"+str(c)+"\n ##############################################\n")
    clauses = t(g)
    i=0
    with open('clause_decima_counterfactual.txt','w',encoding='utf-8') as f:#使用with open()新建对象f
        for clause in clauses[0]:
            i=i+1
            f.write(str(i)+":"+str(clause)+"\n")
    
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
    # encoder.approximate_neq_constr(out_vars, y_)
    solver.add(encoder.get_in_out_constrs())
    solver.add(encoder.get_nn_constrs())
    i=0
    start_t = time.time()
    with open('assert_aurora_sensitivity_analysis.txt','w',encoding='utf-8') as f:#使用with open()新建对象f
        f.write(solver.sexpr())
    print(solver.check())
    my_list=solver.model()
    total_t = time.time()-start_t
    return my_list,total_t

def partial_dependence_plot():
    from src import NeurouNetwork
    import random
    log_path = f'{BASE_DIR}/logs/pdp_decima.csv'
    file = open(log_path, 'w', newline='')
    writer = csv.writer(file)
    net = NeurouNetwork(MODEL_PATH, SYSTEM_NAME)
    x_, y_ = data_list[0]['x'], data_list[0]['y']
    x_ = [data_multi(v) for v in x_]
    y_ = [v * MULTI for v in y_]
    origin=x_[13]
    control_input=[0.0, -2.0, 2.5, 0.45635 ,0.06, 
            1.0752194, 0.50348467, 0.3458266, 0.30929404, 0.69685227,
            0.39543328, -0.0,  0.2488276 ,  0.0, 2.265516,
            0.5268107,   0.0, 0.0, 0.0, 0.0,
            1.0689152,  0.0, 0.0,  0.89891183 , 0.0,
            0.0, 0.66647846,0.0,0.3362084]
    writer.writerow(['K',"X", 'Result', 'Time'])
    for j in range(100):
        sum=0
        predictions = list()
        start_time = time.time()
        for control_value in range(100):
            one_predictions = list()
            # control_input = np.random.uniform(low=0., high=1., size=(100, 1 * 32)).astype(np.float32)

            # random_values = np.random.uniform(low=0., high=1., size=(100, 1 * 32 - 5)).astype(np.float32)
            # control_input = np.concatenate([x_[:5], random_values], axis=1)
            # control_input=x_
            for i in range(29):
                if 13<=i<=20:
                    control_input[i]=x_[i]+j*20
                else:
                    control_input[i]=random.uniform(FEATURE_BOUNDS[i][0],FEATURE_BOUNDS[i][1])
            print(control_input)
            one_predictions.append(net.forward(control_input))
            
            
            p=float(one_predictions[0])
            # writer.writerow([100, p*0.001, t])
            sum=sum+p
            # print(predictions[-1].shape)
        t=time.time()-start_time
        sum=sum/100
        print(sum)
        writer.writerow([j, j*0.01,sum*0.001, t])
        print('Elapsed time: {:.2f} seconds.'.format(time.time() - start_time))


    file.flush()
 
    # file.flush()
    # print(res, total_t)
    file.close()
    # 11.03 seconds. 10.75 seconds
    # with open(f'{BASE_DIR}/logs/deeprm_pdp.pkl', 'rb') as fr:
    #     print(pickle.load(fr))

if __name__ == '__main__':

    ####################  Check Pensieve encoding #############################
    solver = SolverFor('LRA')
    
    x_, y_ = data_list[0]['x'], data_list[0]['y']
    x_ = [data_multi(v) for v in x_]
    y_ = [v * MULTI for v in y_]
    print(x_,y_)
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
#     log_path = f'{BASE_DIR}/logs/adversarial_decima.csv'
#     file = open(log_path, 'w', newline='')
#     writer = csv.writer(file)
#     writer.writerow(['Features', 'Historys', 'Epsilon', 'Result', 'Time'])
    
#     # perturbation feature's index
#     p_features = [
#         0,
#         1,
#         2,
#         3,
#         [0, 1, 2, 3]
#     ]
    
#     historys_list = [
#    1,
#    1,
#    1,
#    1,
#    [1,1,1,1]
# ]
    
#     epsilons = [0.1 * (i + 1) for i in range(20)]
    
#     for features, historys in zip(p_features, historys_list):
#         for epsilon in epsilons:
#             res, total_t = adversarial_perturbation(features, historys, epsilon)
#             writer.writerow([features, historys, epsilon, res, total_t])
#             file.flush()
#     file.close()

    ###################### Missing features ######################################
    # log_path = f'{BASE_DIR}/logs/missing_features_decima.csv'
    # file = open(log_path, 'w', newline='')
    # writer = csv.writer(file)
    # writer.writerow(['Features', 'Historys', 'Result', 'Time'])
    # file.flush()
    
    # # missing feature's index
    # m_features = [i for i in range(FEATURE_TYPES)]
    
    # historys_list = [
    #     2,2,2,2
    # ]
    
    # for feature, historys in zip(m_features, historys_list):
    #     res, total_t = verify_missing_features(feature, historys)
    #     writer.writerow([feature, historys, res, total_t])
    #     file.flush()
    # file.close()

    ############################## Liveness property #############################
    # log_path = f'{BASE_DIR}/logs/liveness_pensieve.csv'
    # file = open(log_path, 'w', newline='')
    # writer = csv.writer(file)
    # writer.writerow(['K', 'Result', 'Time'])
    # file.flush()
    # for i in range(2, 8):
    #     res, total_t = verify_liveness_property(i)
    #     writer.writerow([i, res, total_t])
    #     file.flush()
    #     print(res, total_t)
    # file.close()

    ############################ Verify extreme_values ##############################
    # log_path = f'{BASE_DIR}/logs/extreme_values_pensieve.csv'
    # file = open(log_path, 'w', newline='')
    # writer = csv.writer(file)
    # writer.writerow(['Historys', 'Result', 'Time'])
    # file.flush()
    # historys = [8-i for i in range(8)]
    # for i in range(1, 8):
    #     res, t = verify_extreme_values()
    #     writer.writerow([i, res, t])

    ########################### Verify decision boundary ############################
    # verify_decision_boundary(f'{BASE_DIR}/logs/decision_boundary_pensieve.csv')

    ############################ Find minimal attack power ##########################
    # log_path = f'{BASE_DIR}/logs/minimal_epsilon_decima.csv'
    # file = open(log_path, 'w', newline='')
    # writer = csv.writer(file)
    # writer.writerow(['Features', 'Historys', 'Epsilon', 'Time'])
    # file.flush()
    # pfs = [
    #     0,
    #     1,
    #     2,
    #     3,
    #     [0, 1, 2, 3]
    # ]
    # historys_list = [
    #     1,
    #     1,
    #     1,
    #     1,
    #     [1, 1, 1, 1]
    # ]
    # for i in range(len(pfs)):
    #     epsilon,total_t = find_minimal_attack_power(pfs[i], historys_list[i])
    #     writer.writerow([pfs[i], historys_list[i], epsilon, total_t])
    #     file.flush()
    
    # file.close()

    ################## Decision boundary interpretability ############################
    # x_, y_ = data_list[0]['x'], data_list[0]['y']
    # # print(x_)
    # decision_boundaries(x_, y_, 6, 27, f'{BASE_DIR}/logs/decision_boundaries_decima.csv')

    ################# Qualitatively Anchor interpretability ############################
    # file = open(f'{BASE_DIR}/logs/qualitatively_anchor_{SYSTEM_NAME}_t.csv', 'a', newline='')
    # writer = csv.writer(file)
    # x_, expect_label = data_list[0]['x'], 1
    # y=data_list[0]["y"]
    # features, total_t = qualitatively_anchor_t(x_, y)
    # features.append(total_t)
    # writer.writerow(features)
    # file.flush()
    
    # file.close()

    # ################# Counterfactual Example interpretability ##########################
    # file = open(f'{BASE_DIR}/logs/counterfactual_decima.csv', 'w', newline='')

    # x_, expect_label = data_list[0]['x'], 2
    # writer = csv.writer(file)
    # names, values, total_t = counterfactual_example(x_, [2])
    # names.append('Time')
    # values.append(total_t)
    # writer.writerow([names])
    # writer.writerow([values])
    # file.close()
    # #################################Sensitivity Analysis##############################
    # file = open(f'{BASE_DIR}/logs/sensitivity_{SYSTEM_NAME}_origin.csv', 'w', newline='')
    # writer = csv.writer(file)
    # # x_, y_ = data_list[0]['x'], data_list[0]['y']
    # # sensitivity_analysis(x_,y_)
    # pfs = [
    #     0,
    #     1,
    #     2,
    #     3
    # ]
    # historys_list = [
    #     1,
    #     1,
    #     1,
    #     1
    # ]
    # for i in range(len(pfs)):
    #     jie,total_t = sensitivity_analysis(pfs[i], historys_list[i])
    #     writer.writerow([pfs[i], historys_list[i], jie, total_t])
    #     file.flush()
    
    # file.close()

#########################PDP####################################################
    partial_dependence_plot()

