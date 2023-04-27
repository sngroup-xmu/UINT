from  z3 import And

def split_list_by_idxs(X, idxs):
    special_list, remain_list = [], []

    for i in range(len(X)):
        if i in idxs:
            special_list.append(X[i])
        else:
            remain_list.append(X[i])

    return special_list, remain_list

def find_solution(solver, variables: list, all=True):
    values = []
    new_constraints = []

    solution = solver.model()
    for x in variables:
        values.append(eval(str(solution[x])))
        if all:
            new_constraints.append(x != solution[x])
    if all:
        solver.add(And(new_constraints))

    return values

