import sys, os
import time
from pathos.multiprocessing import ProcessingPool as Pool
# from multiprocessing import Pool
from multiprocessing.pool import ThreadPool as Pool
from random import shuffle
from typing import List

path_current = '/home/ubuntu-h/PycharmProjects/scientificProject'
path_ = os.getcwd()
if path_current not in sys.path:
    sys.path.insert(1, '/home/ubuntu-h/PycharmProjects/scientificProject')
elif path_ not in sys.path:
    sys.path.insert(1, path_)

from PPOPT_main.PPOPT_main.src.ppopt.mpQCQP_program import MPQCQP_Program
from PPOPT_main.PPOPT_main.src.ppopt.mp_solvers.solver_utils import generate_children_sets, CombinationTester
from PPOPT_main.PPOPT_main.src.ppopt.solution import Solution
from PPOPT_main.PPOPT_main.src.ppopt.utils.mpqp_utils import gen_cr_from_active_set
from PPOPT_main.PPOPT_main.src.ppopt.utils.general_utils import num_cpu_cores

import pandapower as pp
net = pp.create_empty_network()

# create buses
bus1 = pp.create_bus(net, vn_kv=110.)
bus2 = pp.create_bus(net, vn_kv=110.)
bus3 = pp.create_bus(net, vn_kv=110.)
# bus4 = pp.create_bus(net, vn_kv=110.)
# bus5 = pp.create_bus(net, vn_kv=110.)

# create 110 kV lines
pp.create_line(net, bus2, bus3, length_km=90., std_type='149-AL1/24-ST1A 110.0')
# pp.create_line(net, bus3, bus4, length_km=50., std_type="149-AL1/24-ST1A 110.0")
# pp.create_line(net, bus4, bus2, length_km=40., std_type="149-AL1/24-ST1A 110.0")
pp.create_line(net, bus1, bus2, length_km=70., std_type='149-AL1/24-ST1A 110.0')

# create loads
pp.create_load(net, bus2, p_mw=50., controllable=False)
# pp.create_load(net, bus3, p_mw=70., controllable=False)
# pp.create_load(net, bus4, p_mw=25., controllable=False)

# create generators
eg = pp.create_ext_grid(net, bus1, min_p_mw=0, max_p_mw=1000, vm_pu=1.05)
g0 = pp.create_gen(net, bus3, p_mw=50, min_p_mw=0, max_p_mw=50, vm_pu=1.00, controllable=True)
# g1 = pp.create_gen(net, bus4, p_mw=50, min_p_mw=0, max_p_mw=50, vm_pu=1.00, controllable=True)


costeg = pp.create_poly_cost(net, 0, 'ext_grid', cp1_eur_per_mw=20)
costgen1 = pp.create_poly_cost(net, 0, 'gen', cp1_eur_per_mw=10)
costgen2 = pp.create_poly_cost(net, 1, 'gen', cp1_eur_per_mw=10)

net.bus["min_vm_pu"] = 0.96
net.bus["max_vm_pu"] = 1.04
net.line["max_loading_percent"] = 100
# net.sn_mva = 100
# net = pp.networks.case14()
# net.gen['vm_pu'] = 1.0
# net.bus["min_vm_pu"] = 0.95
# net.bus["max_vm_pu"] = 1.05
# net.poly_cost[:,3] = 0
# net.poly_cost['cp2_eur_per_mw2'] = 0
my_QCQP = MPQCQP_Program(net=net)

n_constrs = my_QCQP.num_constraints()
n_eq = my_QCQP.num_equality_constraints()
n_theta = my_QCQP.num_t() # correct
n_variables = my_QCQP.num_x()
print(f'There are {n_constrs} constraints, where {n_eq} are equalities;'
      f'\nThere are {n_variables} variables and {n_theta} parameters.')

program = my_QCQP


def check_child_feasibility(program: MPQCQP_Program, set_list: List[List[int]], combination_checker: CombinationTester) -> List[List[int]]:
    """
    Checks the feasibility of a list of active set combinations, if infeasible add to the combination checker and returns all feasible active set combinations

    :param program: An MPQP Program
    :param set_list: The list of active sets
    :param combination_checker: The combination checker that prunes
    :return: The list of all feasible active sets
    """
    output = list()
    for child in set_list:
        if program.check_feasibility(child):
            output.append(child)
        else:
            combination_checker.add_combo(child)

    return output



def full_process(program: MPQCQP_Program, active_set: List[int], murder_list, gen_children):
    """

    This is the fundamental building block of the parallel combinatorial algorithm, here we branch off of a known feasible active set combination\\
    and then


    :param program: A multiparametric program
    :param active_set: the active set combination that we are expanding on
    :param murder_list: the list containing all previously found
    :param gen_children: A boolean flag, that determines if we should generate the children subsets
    :return: a list of the following form [Optional[CriticalRegion], pruned active set combination,Possibly Feasible Active set combinations]
    """
    t_set = (*active_set,)

    return_list = [None, set(), list()]

    is_feasible_ = program.check_feasibility(active_set)

    if not is_feasible_:
        return_list[1].add(t_set)
        return return_list

    is_optimal_ = program.check_optimality(active_set)  # is_optimal(program, equality_indices)

    if not is_optimal_:
        if gen_children:
            return_list[2] = generate_children_sets(active_set, program.num_constraints(), murder_list)
        return return_list

    sol_local = is_optimal_
    return_list[0] = sol_local

    if return_list[0] is None:    # TODO 即使满足最优性，也会出现cr是None的情况，后期要处理。return_list[0]本来是CR，被改了
        return_list[1].add(t_set)
        return return_list

    if gen_children:
        return_list[2] = generate_children_sets(active_set, program.num_constraints(), murder_list)

    return return_list


save = []
start = time.time()

num_cores = num_cpu_cores()

print(f'Spawned threads across {num_cores}')

pool = Pool(32)  # to 64, compare the time consumption TODO

murder_list = CombinationTester()
to_check = list()

solution = Solution(program, [])

max_depth = max(program.num_x(), program.num_t()) - len(program.equality_indices)
# breath first to optimize the elimination

root_node = generate_children_sets(program.equality_indices, program.num_constraints(), murder_list)

to_check.extend(root_node)

for i in range(max_depth):
# if there are no other active sets to check break out of loop
# print(len(to_check))
    print(f'Time at depth test {i + 1}, {time.time() - start}')
    print(f'Number of active sets to be considered is {len(to_check)}')

    depth_time = time.time()

    gen_children = i + 1 != max_depth

    f = lambda x: full_process(program, x, murder_list=murder_list, gen_children=gen_children)

    future_list = list()

    shuffle(to_check)

    outputs = pool.map(f, to_check)
    # outputs = map(f, to_check)

    print(f'Time to run all tasks in parallel {time.time() - depth_time}')
    depth_time = time.time()

    if i + 1 == max_depth:  # 如果达到最大深度，将所有的CR加入到solution中
        for output in outputs:
            if output[0] is not None:
                save.append(output[0]) # 已经将CR改成sol_local了，所以放到save中
                # solution.add_region(output[0])
        break

    for output in outputs:
        murder_list.add_combos(output[1])
        future_list.extend(output[2])
        if output[0] is not None:
            save.append(output[0]) # 已经将CR改成sol_local了，所以放到save中
            # solution.add_region(output[0])

    print(f'Time to process all depth outputs {time.time() - depth_time}')

    to_check = future_list

    # If there are not more active sets to check we are done
    if len(to_check) == 0:
        break

# we never actually tested the program base active set
# if program.check_feasibility(program.equality_indices):
#     sol_ = program.check_optimality(program.equality_indices)
#     if sol_:
#         save.append(sol_) # 已经将CR改成sol_了，所以放到save中
#         # region = gen_cr_from_active_set(program, program.equality_indices)
#         # if region is not None:
#         #     solution.add_region(region)

pool.close()

end_time = time.time()
print("耗时: {:.2f}秒".format(end_time - start))

print('--'*60)

for element in save:
    print('**'*60)
    for key, values in element.items():
    # if isinstance(values, List):

        print(f"The {key}: {values}")
print(f"Number of Save is {len(save)}")