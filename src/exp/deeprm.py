import argparse
import concurrent.futures
import csv
import json
import logging
import pickle
import time

import numpy as np
from z3 import *

logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.INFO)
print = logging.info

sys.path.append("../..")
from src import ConstraintEncoder, split_list_by_idxs, find_solution

from datetime import datetime



print(datetime.now().strftime('%H:%M:%S'))

parser = argparse.ArgumentParser()
parser.add_argument('--system-precision', type=float, default=0.01)  # 0.01 0.00001 sa #0.000001
# parser.add_argument('--is-multi', type=bool, default=True)
args = parser.parse_args()

BASE_DIR = '../..'
SYSTEM_NAME = 'deeprm'
# dataset
with open(f'{BASE_DIR}/model_file/deeprm/{SYSTEM_NAME}_dataset.json', 'r') as fr:
    dataset = json.load(fr)
# model
MODEL_PATH = f'{BASE_DIR}/model_file/deeprm/{SYSTEM_NAME}.pb'

# IS_MULTI = args.is_multi
PRECISION = args.system_precision
MULTI = int(1 / PRECISION)


# if IS_MULTI:
#     MULTI = int(1 / PRECISION)
# else:
#     MULTI = 1.0


# def data_round(value):
#     if IS_MULTI:
#         return round(value)
#     else:
#         return value

def data_round(value):
    return value
    # return round(value)


def scale_to_(value):
    # return value * MULTI
    return round(value * MULTI)


def scale_back(value):
    return value / MULTI


'''
def data_multi(value, reverse=False):
    if IS_MULTI:
        if reverse:
            return value / MULTI
        else:
            return round(value * MULTI)  
    else:
        return value
'''

OUT_LENGTH = 11
HISTORY_LEN = 20
FEATURE_NAMES = dataset['names']
FEATURE_TYPES = int(dataset['types'])
# FEATURE_BOUNDS = [(data_multi(lb), data_multi(ub)) for lb, ub in dataset['bounds']]
FEATURE_BOUNDS = [(scale_to_(0.), scale_to_(1.)) for _ in range(224 * 20)]
FEATURE_RANGES = [bound[1] - bound[0] for bound in FEATURE_BOUNDS]

data_list = dataset['data']


def generate_idxs(features, history=HISTORY_LEN - 1, pos='L'):
    # e_idxs = generate_idxs([1, 12], history)

    if pos == 'R':
        raise NotImplementedError
        # return [i * FEATURE_TYPES + f
        #         for i in range(HISTORY_LEN - history, HISTORY_LEN)
        #         for f in features]
    else:
        feature_to_num = {i: 10 for i in range(22)}
        feature_to_num.update({22: 3, 23: 1})

        base_x = list()
        for f in features:
            start_idx = sum([feature_to_num[b] for b in range(f)] + [0])
            end_idx = start_idx + feature_to_num[f]

            base_x.extend(list(range(start_idx, end_idx)))

            # for t in range(20):
            #     selected_x.extend(list(range(t * 20 + start_idx, t * 20 + end_idx)))

        # init_len = len(selected_x)
        # for time_step in range(history - 1):
        #     selected_x.extend([x + 224 * (time_step + 1) for x in selected_x[:init_len]])

        selected_x = [x + 224 * history for x in base_x]

        print('selected input x:')
        print(len(selected_x))
        print(selected_x)

        return selected_x

        # return [i * FEATURE_TYPES + f
        #         for i in range(history)
        #         for f in features]


############################### Verification Problem ######################################
def adversarial_perturbation(pfs, history, epsilon):
    print('*********** Verify Adversarial Perturbation *********')
    print(f'perturbation features: {pfs}, history: {history}')

    solver = SolverFor('LRA')

    x_, y_ = data_list[0]['x'], data_list[0]['y']
    x_ = [scale_to_(v) for v in x_]
    y_ = [scale_to_(v) for v in y_]

    encoder = ConstraintEncoder(MODEL_PATH, SYSTEM_NAME, FEATURE_NAMES, OUT_LENGTH, multi=MULTI)

    input_vars, out_vars = encoder.get_vars(0)

    # Generate perturbation features, values, bounds and ranges
    # p_idxs = generate_idxs(pfs, history)
    # p_idxs = list(range(224))
    p_idxs=list(range(10))

    perturb_vars, normal_vars = split_list_by_idxs(input_vars, p_idxs)
    perturb_values, normal_values = split_list_by_idxs(x_, p_idxs)

    perturb_ranges = [FEATURE_RANGES[idx] for idx in p_idxs]

    perturb_bounds = [(data_round(v - k * epsilon), data_round(v + k * epsilon))
                      for v, k in zip(perturb_values, perturb_ranges)]
    print(perturb_bounds[0])

    encoder.boundary_constr(perturb_vars, perturb_bounds)
    encoder.assign_constr(normal_vars, normal_values)
    encoder.approximate_neq_constr(out_vars, y_)

    # Add constraints to Z3
    solver.add(encoder.get_in_out_constrs())
    solver.add(encoder.get_nn_constrs())

    start_t = time.time()
    res = "unsat" if solver.check() == unsat else "sat"
    total_t = time.time() - start_t

    print(f'Epsilon: {epsilon} | Result: {res} | Time: {total_t} \n')

    return res, total_t


def verify_missing_features(mfs, history):
    print('******* Verify Missing Features *******')

    solver = SolverFor('LRA')

    x_, y_ = data_list[0]['x'], data_list[0]['y']
    # x_ = [data_multi(v) for v in x_]
    # y_ = [v * MULTI for v in y_]
    x_ = [scale_to_(v) for v in x_]
    y_ = [scale_to_(v) for v in y_]

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
    res = "unsat" if solver.check() == unsat else "sat"
    total_t = time.time() - start_t
    print(f'Missing feature: {mfs} | history: {history} | Result: {res} | Time: {total_t}')

    return res, total_t


def verify_extreme_values(mfs, history):
    print('******* Verify Extreme Values *******')

    solver = SolverFor('LRA')

    x_, y_ = data_list[0]['x'], data_list[0]['y']
    # x_ = [data_multi(v) for v in x_]
    # y_ = [v * MULTI for v in y_]
    x_ = [scale_to_(v) for v in x_]
    y_ = [scale_to_(v) for v in y_]

    encoder = ConstraintEncoder(MODEL_PATH, SYSTEM_NAME, FEATURE_NAMES, OUT_LENGTH, multi=MULTI)
    in_vars, out_vars = encoder.get_vars(0)

    # m_idxs = generate_idxs(mfs, history)
    # m_idxs=list(range(224))
    m_idxs = [0]
    if history > 0:
        for i in range(history):
            m_idxs.append(224 * (i + 1))

    missing_vars, normal_vars = split_list_by_idxs(in_vars, m_idxs)
    _, normal_values = split_list_by_idxs(x_, m_idxs)
    missing_bounds = [(scale_to_(0.0), scale_to_(0.05)) for _ in m_idxs]

    encoder.boundary_constr(missing_vars, missing_bounds)
    encoder.assign_constr(normal_vars, normal_values)
    encoder.approximate_neq_constr(out_vars, y_)

    solver.add(encoder.get_in_out_constrs())
    solver.add(encoder.get_nn_constrs())

    start_t = time.time()
    res = "unsat" if solver.check() == unsat else "sat"
    total_t = time.time() - start_t
    print(f'Extreme values: {mfs} | history: {history} | Result: {res} | Time: {total_t}')

    return res, total_t


def verify_extreme_values_old(history=10):
    print('*********** Verify Extreme Values *********')
    solver = SolverFor('LRA')

    x_, y_ = data_list[0]['x'], data_list[0]['y']
    x_ = [scale_to_(v) for v in x_]
    y_ = [scale_to_(v) for v in y_]

    encoder = ConstraintEncoder(MODEL_PATH, SYSTEM_NAME, FEATURE_NAMES, OUT_LENGTH, multi=MULTI)

    in_vars, out_vars = encoder.get_vars(0)

    m_idxs = generate_idxs([1], history)
    extreme_vars, normal_vars = split_list_by_idxs(in_vars, m_idxs)
    encoder.boundary_constr(extreme_vars, [(scale_to_(0.99), scale_to_(1.0))] * len(
        extreme_vars))

    normal_idxs = [i for i in range(len(in_vars)) if i not in m_idxs]
    _, normal_values = split_list_by_idxs(x_, normal_idxs)
    encoder.assign_constr(normal_vars, normal_values)

    encoder.approximate_eq_constr(out_vars, y_)

    '''
    # Generate extreme features and bounds
    # e_idxs = generate_idxs([1, 12], history)  
    e_idxs = generate_idxs([1], history)  
    e_bounds = []
    for idx in range(len(in_vars)):
        if idx in e_idxs:
            # e_bounds.append((data_multi(0.99), data_multi(1.0)))
            e_bounds.append()
        else:
            e_bounds.append(FEATURE_BOUNDS[idx])  

    encoder.boundary_constr(in_vars, e_bounds)
    encoder.approximate_eq_constr(out_vars, [0]) 
    '''

    solver.add(encoder.get_in_out_constrs())
    solver.add(encoder.get_nn_constrs())

    # with open('./extreme_values.smt2','wb') as fw:
    #     fw.write(bytes(solver.to_smt2(),encoding='utf-8'))
    # assert 1==2

    start_t = time.time()
    res = 'sat' if solver.check() == sat else 'unsat'
    total_t = time.time() - start_t

    print(f'History: {history} | Result: {res} | Time: {total_t} \n')

    return res, total_t


'''
def verify_decision_boundary(log_path):
    print('****** Verify Decision Boundary *******')

    file = open(log_path, 'w', newline='')
    writer = csv.writer(file)
    writer.writerow(['Features', 'Result', 'Time'])
    file.flush()

    # Initialize
    solver = SolverFor('LRA')

    x_, y_ = data_list[0]['x'], data_list[0]['y']
    # x_ = [data_multi(v) for v in x_]
    # y_ = [v * MULTI for v in y_]
    x_ = [scale_to_(v) for v in x_]
    y_ = [scale_to_(v) for v in y_]

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
'''


################################# Interpretability Problem ####################################
def qualitatively_anchor(x, y):
    print(f'******** Qualitatively Anchor *******')

    # x = [data_multi(v) for v in x]
    # y = [v * MULTI for v in y]
    x = [scale_to_(v) for v in x]
    # y = [scale_to_(v) for v in y]
    # print(y)
    solver = SolverFor('LRA')
    # solver.set(unsat_core=True)
    # solver.set("smt.core.minimize", True)
    # set_param('parallel.enable', True)

    encoder = ConstraintEncoder(MODEL_PATH, SYSTEM_NAME, FEATURE_NAMES, OUT_LENGTH, multi=MULTI)
    in_vars, out_vars = encoder.get_vars(0)

    encoder.assign_track(in_vars, x)
    # encoder.approximate_neq_constr(out_vars, y)

    tracks = encoder.get_tracks()
    # implies_vars = [Bool('time-step-{}'.format(i)) for i in range(20)]
    # implies_vars = [Bool('1'), Bool('2')]
    for name, expr in tracks.items():
        # solver.add(Implies(implies_vars[int(name.split('-')[0].split('_')[-1])], expr))

        solver.add(expr)

        # if int(name.split('-')[0].split('_')[-1]) < 10:
        #     solver.add(Implies(implies_vars[0], expr))
        # else:
        #     solver.add(Implies(implies_vars[1], expr))
        # solver.add(Implies(implies_vars[0], expr))

        # solver.add(Implies(Bool(name.split('-')[0]), expr))
        # solver.assert_and_track(expr, name)

    # encoder.argmax_neq_constr(out_vars, 0)  # in_out
    encoder.argmax_eq_constr(out_vars, 0)

    solver.add(encoder.get_in_out_constrs())
    solver.add(encoder.get_nn_constrs())

    # i=0
    # with open(f'{BASE_DIR}/logs/expr_{SYSTEM_NAME}_qualitatively_anchor.txt', 'w', encoding='utf8') as f:
    #     f.write(solver.sexpr())

    # IJCAI'15
    def tt(solver_, f):
        return is_true(solver_.model().eval(f))

    def get_mss(s, mss, ps):
        ps = ps - mss

        backbones = set([])
        while len(ps) > 0:
            print(len(ps))
            p = ps.pop()
            if sat == s.check(mss | backbones | {p}):
                mss = mss | {p} | {q for q in ps if tt(s, q)}
                ps = ps - mss
            else:
                backbones = backbones | {Not(p)}
        return mss

    def get_mss_base(solver_, ps):
        print('begin to compute')
        result = solver_.check()
        if sat != result:
            print('unsat')
            return []
        else:
            print('sat')

        mss = {q for q in ps if tt(solver_, q)}
        return get_mss(solver_, mss, ps)

    start_time = time.time()
    features = get_mss_base(solver, set([v for k, v in tracks.items()]))
    features = list(features)

    # features=get_mss_base(solver,)
    end_time = time.time()
    # print(features)
    print(len(features))

    return features, end_time - start_time

    '''
    print('-----------------------------------')
    start_t = time.time()
    # names = set([n.split('-')[0] for n, e in tracks.items()])
    # input_var_names = [Bool(n) for n in names]
    # print(input_var_names)
    # assert 1==2
    if solver.check(implies_vars[0], implies_vars[1]) == unsat:
        print('unsat')
        CoMSS = solver.unsat_core()
        # features = [str(x).split('$')[0] for x in CoMSS]
        features = [x for x in implies_vars if x in CoMSS]
        print(CoMSS)
    else:
        print('sat')
        features = []

    return features, time.time() - start_t
    '''


def counterfactual_example(x_, ye, precision=0.1):
    print(f'****** Search counterfactual example, ye = {ye[0]} ******')

    solver = SolverFor('LRA')
    # g = Goal()

    # t = Tactic('tseitin-cnf')
    # x_ = [data_multi(v) for v in x_]
    # ye = [v * MULTI for v in ye]
    x_ = [scale_to_(v) for v in x_]
    ye = [scale_to_(v) for v in ye]

    encoder = ConstraintEncoder(MODEL_PATH, SYSTEM_NAME, FEATURE_NAMES, OUT_LENGTH, multi=MULTI)
    in_vars, out_vars = encoder.get_vars(0)

    lb, ub = 0., 0.
    for feature, (low, high), ranger in zip(x_, FEATURE_BOUNDS, FEATURE_RANGES):
        if feature - low > high - feature:
            ub += (feature - low) / ranger
        else:
            ub += (high - feature) / ranger

    constr = RealVal(0)
    for var_desp, var_val, ranger in zip(in_vars, x_, FEATURE_RANGES):
        constr = constr + If(var_desp > var_val,
                             (var_desp - var_val) / ranger, (var_val - var_desp) / ranger)

    encoder.boundary_constr(in_vars, FEATURE_BOUNDS)
    encoder.approximate_eq_constr(out_vars, ye)
    # encoder.argmax_eq_constr(out_vars, 1)

    solver.add(encoder.get_in_out_constrs())
    solver.add(encoder.get_nn_constrs())

    # g.add(encoder.get_in_out_constrs())
    # g.add(encoder.get_nn_constrs())

    solver.push()
    solver.add(constr < ub)

    # g.add(constr < ub)

    # i=0
    # with open('assert_aurora_counterfactual.txt','w',encoding='utf-8') as f:
    #     for c in solver.assertions():
    #         i=i+1
    #         f.write(str(i)+":"+str(c)+"\n ##############################################\n")
    # clauses = t(g)
    # i=0
    # with open('clause_aurora_counterfactual.txt','w',encoding='utf-8') as f:
    #     for clause in clauses[0]:
    #         i=i+1
    #         f.write(str(i)+":"+str(clause)+"\n")

    start_t = time.time()
    # precision *= MULTI
    precision = scale_to_(precision)
    while ub - lb > precision:
        print(f'bound: ({lb: .6f}, {ub: .6f})')
        mid = (lb + ub) / 2

        solver.pop()
        solver.push()
        solver.add(constr < mid)

        if solver.check() == sat:
            ub = mid
            print('sat')
        else:
            lb = mid
            print('unsat')

        print('{}={}'.format(ub - lb, precision))

    names = [str(x).split('$')[0] for x in in_vars]
    solver.pop()
    solver.add(constr < ub)

    solver.check()
    values = find_solution(solver, in_vars, all=False)
    # values = [data_multi(v, reverse=True) for v in values]
    values = [scale_back(v) for v in values]
    total_t = time.time() - start_t

    return names, values, total_t



def decision_boundaries(x_, y_, feature1, feature2, log_path, total=10000):
    print('******* Searching Decision Boundaries *******')

    solver = SolverFor('LRA')
    # g = Goal()
    # t = Tactic('tseitin-cnf')
    # x_ = [data_multi(v) for v in x_]
    # y_ = [v * MULTI for v in y_]
    x_ = [scale_to_(v) for v in x_]
    y_ = [scale_to_(v) for v in y_]

    encoder = ConstraintEncoder(MODEL_PATH, SYSTEM_NAME, FEATURE_NAMES, OUT_LENGTH, multi=MULTI)
    in_vars, out_vars = encoder.get_vars(0)

    name1, name2 = str(in_vars[feature1]).split('$')[0], str(in_vars[feature2]).split('$')[0]
    file = open(log_path, 'w', newline='')
    writer = csv.writer(file)
    writer.writerow([name1, name2, 'Time'])

    bound_vars, assign_vars = split_list_by_idxs(in_vars, [feature1, feature2])
    _, assign_values = split_list_by_idxs(x_, [feature1, feature2])
    bounds = [FEATURE_BOUNDS[feature1], FEATURE_BOUNDS[feature2]]

    encoder.boundary_constr(bound_vars, bounds)
    encoder.assign_constr(assign_vars, assign_values)
    encoder.approximate_eq_constr(out_vars, y_)

    solver.add(encoder.get_in_out_constrs())
    solver.add(encoder.get_nn_constrs())
    # g.add(encoder.get_in_out_constrs())
    # g.add(encoder.get_nn_constrs())
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
        # values = [data_multi(v, reverse=True) for v in values]
        values = [scale_back(v) for v in values]
        total_t = time.time() - start_t
        writer.writerow([values[0], values[1], total_t])
        print(f'Solution: [{name1} = {values[0]}, {name2} = {values[1]}] | Time: {total_t: .2}s')
        file.flush()

        start_t = time.time()
        count += 1
    file.close()



def shh_decision_boundaries(x_, y_, feature1, feature2, log_path, total=10000):
    print('******* Searching Decision Boundaries *******')

    x_ = [scale_to_(v) for v in x_]
    y_ = [scale_to_(v) for v in y_]

    encoder = ConstraintEncoder(MODEL_PATH, SYSTEM_NAME, FEATURE_NAMES, OUT_LENGTH, multi=MULTI)
    in_vars, out_vars = encoder.get_vars(0)

    name1, name2 = str(in_vars[feature1]).split('$')[0], str(in_vars[feature2]).split('$')[0]
    file = open(log_path, 'w', newline='')
    writer = csv.writer(file)
    writer.writerow(['result', name1, name2, 'Time'])

    bound_vars, assign_vars = split_list_by_idxs(in_vars, [feature1, feature2])
    _, assign_values = split_list_by_idxs(x_, [feature1, feature2])
    # bounds = [FEATURE_BOUNDS[feature1], FEATURE_BOUNDS[feature2]]

    solver = SolverFor('LRA')
    from itertools import product
    for x1, x2 in product(range(100), range(100)):
        start_t = time.time()
        solver.reset()
        # encoder.boundary_constr(bound_vars, bounds)
        encoder.assign_constr(bound_vars, [scale_to_(x1 / 100), scale_to_(x2 / 100)])
        encoder.assign_constr(assign_vars, assign_values)
        encoder.approximate_eq_constr(out_vars, y_)

        solver.add(encoder.get_in_out_constrs())
        solver.add(encoder.get_nn_constrs())

        res='unsat'
        if solver.check()==sat:
            res='sat'
            # values = find_solution(solver, bound_vars)
        else:
            res = 'unsat'

        total_t = time.time() - start_t
        values = [x1 / 100, x2 / 100]
        writer.writerow([res, values[0], values[1], total_t])

        print(f'Solution: [{res} {name1} = {values[0]}, {name2} = {values[1]}] | Time: {total_t: .2}s')
        file.flush()

    file.close()


def find_minimal_attack_power(pf, history, precision=1e-6):
    print('***** Minimal Attack Power ******')
    print(f'Perturbation Features: {pf} | History: {history}')

    solver = SolverFor('LRA')
    # g = Goal()
    # t = Tactic('tseitin-cnf')
    x_, y_ = data_list[0]['x'], data_list[0]['y']
    # x_ = [data_multi(v) for v in x_]
    # y_ = [v * MULTI for v in y_]
    x_ = [scale_to_(v) for v in x_]
    y_ = [scale_to_(v) for v in y_]

    encoder = ConstraintEncoder(MODEL_PATH, SYSTEM_NAME, FEATURE_NAMES, OUT_LENGTH, multi=MULTI)
    in_vars, out_vars = encoder.get_vars(0)

    # Generate perturbation features' set
    # p_idxs = generate_idxs(pf, history)
    p_idxs = [pf + history * 224]
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
        # g.add(encoder.get_in_out_constrs())
        # g.add(encoder.get_nn_constrs())

    lb, epsilon = 0, 0.05
    start_t = time.time()
    reset_constraints(epsilon)

    # i=0
    # with open('assert_aurora_minimal_attack_power.txt','w',encoding='utf-8') as f:
    #     for c in solver.assertions():
    #         i=i+1
    #         f.write(str(i)+":"+str(c)+"\n ##############################################\n")
    # clauses = t(g)
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

def sensitivity_analysis(pf, history):
    solver = SolverFor('LRA')
    x_, y_ = data_list[0]['x'], data_list[0]['y']
    # x_ = [data_multi(v) for v in x_]
    # y_ = [v * MULTI for v in y_]
    x_ = [scale_to_(v) for v in x_]
    y_ = [scale_to_(v) for v in y_]

    encoder = ConstraintEncoder(MODEL_PATH, SYSTEM_NAME, FEATURE_NAMES, OUT_LENGTH, multi=MULTI)
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

    # with open('assert_aurora_sensitivity_analysis.txt','w',encoding='utf-8') as f:
    #     f.write(solver.sexpr())

    print(solver.check())
    my_list = solver.model()
    total_t = time.time() - start_t
    return my_list, total_t


def sensitivity_analysis_ground_truth():
    from src import NeurouNetwork

    net = NeurouNetwork(MODEL_PATH, SYSTEM_NAME)

    x_ = data_list[0]['x']
    print(net.forward(x_))

def sensitivity_analysis_new(pf):
    solver = SolverFor('LRA')
    x_, y_ = data_list[0]['x'], data_list[0]['y']
    x_ = [scale_to_(v) for v in x_]
    # y_ = [scale_to_(v) for v in y_]

    encoder = ConstraintEncoder(MODEL_PATH, SYSTEM_NAME, FEATURE_NAMES, OUT_LENGTH, multi=MULTI)
    in_vars, out_vars = encoder.get_vars(0)

    base_idxs = generate_idxs(pf, 0)

    p_idxs = list()
    p_idxs.extend(base_idxs)
    for i in range(19):
        p_idxs.extend([b + 224 * (i + 1) for b in base_idxs])
    print('perturbation index len is: {}'.format(len(p_idxs)))

    perturb_vars, normal_vars = split_list_by_idxs(in_vars, p_idxs)
    perturb_values, normal_values = split_list_by_idxs(x_, p_idxs)
    encoder.assign_constr(normal_vars, normal_values)

    perturb_ranges = [FEATURE_RANGES[idx] for idx in p_idxs]
    # epsilon = 5e-5
    # epsilon = 1e-3
    # perturb_bounds = [(data_round(v - k * epsilon), data_round(v + k * epsilon))
    #                   for v, k in zip(perturb_values, perturb_ranges)]
    noise = 2
    perturb_bounds = [(data_round(v - noise), data_round(v + noise))
                      for v, k in zip(perturb_values, perturb_ranges)]
    encoder.boundary_constr(perturb_vars, perturb_bounds)

    solver.add(encoder.get_in_out_constrs())
    solver.add(encoder.get_nn_constrs())

    start_t = time.time()
    count = 0
    output = []
    sum = 1200
    # select = []
    while count < sum and solver.check() == sat:
        solution = solver.model()
        output.append([eval(str(solution[out_vars[i]])) for i in range(11)])

        new_constraints = []
        for x in perturb_vars:
            # select.append(eval(str(solution[x])))
            new_constraints.append(x != solution[x])
        solver.add(And(new_constraints))

        count += 1

    total_t = time.time() - start_t

    return output, total_t


def sensitivity_analysis_wrapper():
    features = [divmod(i, 224)[1] for i in range(224 * 20)]
    history = [divmod(i, 224)[0] for i in range(224 * 20)]

    results = list()
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for i, (epsilon, total_t) in zip(range(224 * 20), executor.map(find_minimal_attack_power, features, history)):
            results.append([i, epsilon, total_t])

    return results


def partial_dependence_plot():
    '''
    while 1st_dimention < 1:
        1st_dimention=x
        for other_dimention = bound constraint
            solve one solution
            except this solution
        average target class value

    plot a figure: x-axis 是 1st-dimension，y-axis 是 target class value
    :return:
    '''
    # elapsed_time = list()
    # x_, y_ = data_list[0]['x'], data_list[0]['y']

    from src import NeurouNetwork

    net = NeurouNetwork(MODEL_PATH, SYSTEM_NAME)

    start_time = time.time()

    predictions = list()
    for control_value in range(100):

        one_predictions = list()
        for _ in range(10):
            control_input = np.random.uniform(low=0., high=1., size=(100, 224 * 20)).astype(np.float32)

            def get_all_index(start, end):
                base = list(range(start, end))

                all_index = list()
                for factor in range(20):
                    all_index.extend([b + 224 * factor for b in base])
                return all_index

            # control_input[:, get_all_index(0, 10)] = control_value / 100.
            # control_input[:, get_all_index(110, 120)] = control_value / 100.

            control_input[:, get_all_index(10, 20)] = control_value / 100.
            # control_input[:, get_all_index(20, 110)] = 0.
            control_input[:, get_all_index(120, 130)] = control_value / 100.
            # control_input[:, get_all_index(130, 220)] = 0.

            '''
            control_input[:,10] = control_value / 100.  
            control_input[:,110:120] = control_value / 100.

            control_input[:,10:20] = 1.
            control_input[:,20:110] = 0.
            control_input[:,120:130] = 1.
            control_input[:,130:220] = 0.
            '''

            one_predictions.append(net.forward(control_input))

        predictions.append(np.concatenate(one_predictions, axis=0))
        # print(predictions[-1].shape)

    # 11.03 seconds. 10.75 seconds
    print('Elapsed time: {:.2f} seconds.'.format(time.time() - start_time))

    with open(f'{BASE_DIR}/logs/deeprm_pdp.pkl', 'wb') as fw:
        pickle.dump(predictions, fw)

    # with open(f'{BASE_DIR}/logs/deeprm_pdp.pkl', 'rb') as fr:
    #     print(pickle.load(fr))

    '''
    predictions = list()
    for v in range(2):
        start_time = time.time()

        control_value = [scale_to_(v / 100.)]
        encoder.assign_constr(control_features, control_value)

        other_range = FEATURE_BOUNDS[1:]
        encoder.boundary_constr(other_features, other_range)

        solver.add(encoder.get_in_out_constrs())
        solver.add(encoder.get_nn_constrs())

        for idx in range(2):
            print('{}-{}-1200'.format(v, idx))

            if solver.check() == unsat:
                break

            print('1')

            solution = solver.model()
            print('1')

            predictions.append([v, eval(str(solution[out_vars[0]]))])
            print('1')

            exclude_constraints = [feature != solution[feature] for feature in other_features]
            solver.add(And(exclude_constraints))
        encoder.reset()
        solver.reset()

        elapsed_time.append(time.time() - start_time)
        
        
    with open(f'{BASE_DIR}/logs/deeprm_pdp.pkl', 'wb') as fw:
        pickle.dump([elapsed_time, predictions], fw)

    with open(f'{BASE_DIR}/logs/deeprm_pdp.pkl', 'rb') as fr:
        print(pickle.load(fr))
    '''


if __name__ == '__main__':
    '''
    #################################  Adversarial Perturbation ####################################
    file = open(f'{BASE_DIR}/logs/adversarial_{SYSTEM_NAME}.csv', 'w', newline='')
    writer = csv.writer(file)
    writer.writerow(['Features', 'History', 'Epsilon', 'Result', 'Time'])
    file.flush()

    # historys = [10, 5, 3]

    epsilons = [(i+1) / 20  for i in range(20)]

    for epsilon in epsilons:
        print(epsilon)
        res, total_t = adversarial_perturbation([11], 0, epsilon)

        writer.writerow([[11], 10, epsilon, res, total_t])
        file.flush()

    file.close()

    # 
    ###################################  Missing features ####################################
    m_features = [[i] for i in range(24)]

    file = open(f'{BASE_DIR}/logs/missing_features_{SYSTEM_NAME}.csv', 'w', newline='')
    writer = csv.writer(file)
    writer.writerow(['Features', 'History', 'Result', 'Time'])
    file.flush()

    for h in range(20):
        for mf in m_features:
            res, total_t = verify_missing_features(mf, h)

            writer.writerow([mf, 0, res, total_t])
            file.flush()
    file.close()

    ###############################  Extreme values #############################################
    file = open(f'{BASE_DIR}/logs/extreme_values_{SYSTEM_NAME}.csv', 'w', newline='')
    writer = csv.writer(file)
    writer.writerow(['History', 'Result', 'Time'])
    file.flush()

    for history in range(20):
        res, total_t = verify_extreme_values([0], history)

        writer.writerow([history, res, total_t])
        file.flush()
    file.close()

    ####################### Verify decision boundary ######################################
    # verify_decision_boundary(f'{BASE_DIR}/logs/decision_boundary_{SYSTEM_NAME}.csv')

    # ############################## Qualitatively Anchor Interpretability #################
    file = open(f'{BASE_DIR}/logs/qualitatively_anchor_{SYSTEM_NAME}.csv', 'a', newline='')
    writer = csv.writer(file)
    x_, y_ = data_list[0]['x'], data_list[0]['y']

    features, total_t = qualitatively_anchor(x_, y_)

    features.append(total_t)
    writer.writerow(features)
    file.flush()

    # ################# Conterfactual Example ###########################
    file = open(f'{BASE_DIR}/logs/counterfactual_{SYSTEM_NAME}_origin.csv', 'w', newline='')
    writer = csv.writer(file)

    # print(data_list[0]['x'])

    x_ = data_list[0]['x']
    ye = [0] * 11
    ye[1] = 1
    names, values, total_t = counterfactual_example(x_, ye)

    names.append('Time')
    values.append(total_t)
    writer.writerow([names])
    writer.writerow([values])
    file.close()


    # # # # ################## Decision boundary interpretability ############################
    x_, y_ = data_list[0]['x'], data_list[0]['y']
    # decision_boundaries(x_, y_, 0, 1, f'{BASE_DIR}/logs/decision_boundaries_{SYSTEM_NAME}.csv')
    shh_decision_boundaries(x_, y_, 0, 110, f'{BASE_DIR}/logs/decision_boundaries_{SYSTEM_NAME}.csv')
    # decision_boundaries(x_, [2.0], 0, 1, f'{BASE_DIR}/logs/decision_boundaries_{SYSTEM_NAME}_2.csv')

    # # ################### Feature Importance —— Find Minimal Attack Power ###########################
    file = open(f'{BASE_DIR}/logs/minimal_attack_{SYSTEM_NAME}.csv', 'w', newline='')
    writer = csv.writer(file)
    writer.writerow(['Features', 'Epsilon', 'Time'])
    file.flush()

    # (10+10*10)*2+3+1
    # 224

    # arguements = [(divmod(i, 224)[1], divmod(i, 224)[0]) for i in range(224 * 20)]
    results = sensitivity_analysis_wrapper()
    for row in results:
        writer.writerow(row)
    file.flush()
    file.close()
    
    # for i in range(224 * 20):
    #     history, feature = divmod(i, 224)
    #     epsilon, total_t = find_minimal_attack_power(feature,history)
    #
    #     writer.writerow([i, epsilon, total_t])
    #     file.flush()

    # #################################Sensitivity Analysis##############################
    feature_to_num = {i: 10 for i in range(22)}
    feature_to_num.update({22: 3, 23: 1})  # 0-23, 共224。此外还有20个time steps

    with open('./deeprm_sensitivity.pkl', 'wb') as fw:
        predictions = list()
        for f in range(24):
            mylist, total_t = sensitivity_analysis_new([f])
            predictions.append([f, mylist, total_t])

            print(f)
            print(mylist)

        pickle.dump(predictions, fw)

    sensitivity_analysis_ground_truth()
    '''

    # ################################# Partial Dependence Plot ##############################
    partial_dependence_plot()