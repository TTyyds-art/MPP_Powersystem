import os
import sys
import time
import warnings
from multiprocessing.pool import ThreadPool as Pool
from typing import List

from ppopt_main.PPOPT_main.src.ppopt.mp_solvers.solver_utils import \
    generate_children_sets, CombinationTester
from ppopt_main.PPOPT_main.src.ppopt.solution import Solution

import pandapower as pp

path_current = '/home/huzuntao/PycharmProjects/MPP_Powersystem/'
path_ = os.getcwd()
if path_current not in sys.path:
    sys.path.insert(1, '/home/huzuntao/PycharmProjects/MPP_Powersystem/')
elif path_ not in sys.path:
    sys.path.insert(1, path_)

from ppopt_main.PPOPT_main.src.ppopt.mpQCQP_program import MPQCQP_Program
from ppopt_main.PPOPT_main.src.ppopt.utils.general_utils import num_cpu_cores

# 设定输出警告信息的方式
warnings.filterwarnings("ignore")  # 忽略掉所有警告

# 执行某些可能产生警告的代码

net = pp.create_empty_network()

# create buses
bus1 = pp.create_bus(net, vn_kv=110.)
bus2 = pp.create_bus(net, vn_kv=110.)
bus3 = pp.create_bus(net, vn_kv=110.)
bus4 = pp.create_bus(net, vn_kv=110.)
bus5 = pp.create_bus(net, vn_kv=110.)
bus6 = pp.create_bus(net, vn_kv=110.)

# create 110 kV lines
pp.create_line(net, bus4, bus5, length_km=90., std_type='149-AL1/24-ST1A 110.0')
pp.create_line(net, bus3, bus4, length_km=90., std_type='149-AL1/24-ST1A 110.0')
pp.create_line(net, bus2, bus3, length_km=90., std_type='149-AL1/24-ST1A 110.0')
pp.create_line(net, bus1, bus2, length_km=70., std_type='149-AL1/24-ST1A 110.0')
pp.create_line(net, bus6, bus3, length_km=70., std_type='149-AL1/24-ST1A 110.0')

# create loads
p_load_1 = 10
p_load_2 = 30
pp.create_load(net, bus2, p_mw=p_load_1, controllable=False)
pp.create_load(net, bus4, p_mw=p_load_2/2, controllable=False)
pp.create_load(net, bus5, p_mw=p_load_2/2, controllable=False)
pp.create_load(net, bus6, p_mw=p_load_2/2, controllable=False)
# create generators
eg = pp.create_ext_grid(net, bus1, min_p_mw=0, max_p_mw=1000, vm_pu=1.05)
g0 = pp.create_gen(net, bus3, p_mw=80, min_p_mw=0, max_p_mw=80, vm_pu=1.00, controllable=True)

costeg = pp.create_poly_cost(net, 0, 'ext_grid', cp1_eur_per_mw=20)
costgen1 = pp.create_poly_cost(net, 0, 'gen', cp1_eur_per_mw=10)
costgen2 = pp.create_poly_cost(net, 1, 'gen', cp1_eur_per_mw=10)

net.bus["min_vm_pu"] = 0.96
net.bus["max_vm_pu"] = 1.04
net.line["max_loading_percent"] = 100
my_QCQP = MPQCQP_Program(net=net)

# 以下是测试代码，先暂时注释掉
# sol = my_QCQP.check_feasibility_SOC([10, 24])  # 检查可行性
# print(f"sol:{sol}")
# sol = my_QCQP.check_optimality_SOC([10, 24])  # 检查最优性
# print(f"sol opt:{sol}")


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
        if program.check_feasibility_SOC(child):
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

    is_feasible_ = program.check_feasibility_SOC(active_set)

    if not is_feasible_:
        return_list[1].add(t_set)
        return return_list

    is_optimal_ = program.check_optimality_SOC(active_set)  # is_optimal(program, equality_indices)

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
start_time = time.time()
start = time.time()

num_cores = num_cpu_cores()

print(f'Spawned threads across {num_cores}')

pool = Pool(num_cores/2)  # to 64, compare the time consumption TODO

murder_list = CombinationTester()
to_check = list()

solution = Solution(my_QCQP, [])

# TODO 这个最大深度的公式，有待分析
max_depth = max(my_QCQP.num_x(), my_QCQP.num_t())
# breath first to optimize the elimination

root_node = generate_children_sets([], my_QCQP.num_inequality_constraints(), murder_list)

to_check.extend(root_node)


for i in range(max_depth):
# if there are no other active sets to check break out of loop
# print(len(to_check))

    future_sets = list()
    # creates the list of feasible active sets
    feasible_sets = check_child_feasibility(my_QCQP, to_check, murder_list)
    for child_set in feasible_sets:

        # soln = check_optimality(program, equality_indices=child_set)
        # The active set is optimal try to build a critical region

        # print(f"Child set: {child_set}")
        # if soln is not None:
        print('--'*50)
        sol = my_QCQP.check_optimality_SOC(child_set) # 检查最优性
        if sol is not None:
            critical_region = 1
            save.append(sol)
            print(child_set)
            # critical_region = gen_cr_from_active_set(program, child_set)  #TODO CR 是否是由 inactive set 组成?
            # Check the dimensions of the critical region
            if critical_region is not None:
                solution.add_region(critical_region)

        # propagate sets

        if i + 1 != max_depth:
            future_sets.extend(generate_children_sets(child_set, my_QCQP.num_inequality_constraints(), murder_list))

    to_check = future_sets
end_time = time.time()
print("耗时: {:.2f}秒".format(end_time - start_time))

print('--'*60)

for element in save:
    print('**'*60)
    for key, values in element.items():
    # if isinstance(values, List):

        print(f"The {key}: {values}")
print(f"Number of Save is {len(save)}")


# 暂时注释掉，因为这里的代码是为了测试QCQP的求解结果是否正确。23.11.21
# 处理输出的排列顺序；
# temp_1 = np.squeeze(net.res_bus)
# print(f"bus: \n{temp_1}")
# temp_2 = np.squeeze(net.res_ext_grid.values)
# print(f'o = {r_[temp_2[0], temp_1[0], temp_2[1], temp_1[1]]}')
#
# A_x, b_x, A_l, b_l = my_QCQP.optimal_control_law()
#
# p_load_ = ones([3, 1])*15
# p_load_[0] = 12.0
# pq_load = r_[p_load_, zeros([3, 1])]
# x_mpp = A_x @ pq_load + b_x.reshape([len(b_x), 1])
# print(f'x = {x_mpp.reshape(len(x_mpp))[-4:]}.')
# # print(f'dh.T*x = {my_QCQP.dh.T @ x_mpp}')
# # print(f'dh.T*x_0 = {my_QCQP.dh.T @ my_QCQP.x_0}')
# print(f'x_0 = {my_QCQP.x_0}')
# # 对比不同负载下，三种求解的结果（pipopt, QCQP, MMP）
# mu_ = my_QCQP.raw['lmbda']['mu']
# idx_act = find(mu_> 1e-3).tolist() # 修改成了 1e-3, 因为对互补性的阈值是1e-6，阈值=z*mu/(1+max(x)) => z*mu<5e-6;mu>1e-3能够保证z<1e-3
# # print(f'idx_act = {idx_act}; mu = {mu_[idx_act]}')
# cri_region = gen_cr_from_active_set(my_QCQP, active_set=idx_act)
# print(f'CR = {cri_region}')