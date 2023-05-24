import pandapower as pp
import warnings
import time
from numpy import flatnonzero as find, ones, zeros, Inf, pi, exp, conj, r_
from PPOPT_main.PPOPT_main.src.ppopt.mp_solvers.solver_utils import \
    generate_children_sets, CombinationTester
from PPOPT_main.PPOPT_main.src.ppopt.solution import Solution
import sys, os
path_current = '/home/huzuntao/PycharmProjects/MPP_Powersystem/'
path_ = os.getcwd()
if path_current not in sys.path:
    sys.path.insert(1, '/home/huzuntao/PycharmProjects/MPP_Powersystem/')
elif path_ not in sys.path:
    sys.path.insert(1, path_)

from PPOPT_main.PPOPT_main.src.ppopt.mpQCQP_program import MPQCQP_Program
from PPOPT_main.PPOPT_main.src.ppopt.utils.mpqp_utils import gen_cr_from_active_set
import numpy as np

# 设定输出警告信息的方式
warnings.filterwarnings("ignore")  # 忽略掉所有警告

# 执行某些可能产生警告的代码

net = pp.create_empty_network()

# create buses
bus1 = pp.create_bus(net, vn_kv=110.)
bus2 = pp.create_bus(net, vn_kv=110.)
bus3 = pp.create_bus(net, vn_kv=110.)

# create 110 kV lines
pp.create_line(net, bus2, bus3, length_km=90., std_type='149-AL1/24-ST1A 110.0')
pp.create_line(net, bus1, bus2, length_km=70., std_type='149-AL1/24-ST1A 110.0')

# create loads
p_load = 60
pp.create_load(net, bus2, p_mw=p_load, controllable=False)

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
# sol = my_QCQP.check_optimality([10]) # 检查最优性
#
# murder_list = CombinationTester()
# to_check = list()
#
# solution = Solution(my_QCQP, [])
#
# # TODO 这个最大深度的公式，有待分析
# max_depth = max(my_QCQP.num_x(), my_QCQP.num_t())
# # breath first to optimize the elimination
#
# root_node = generate_children_sets([], my_QCQP.num_inequality_constraints(), murder_list)
#
# to_check.extend(root_node)
#
# from typing import List
# def check_child_feasibility(program: MPQCQP_Program, set_list: List[List[int]], combination_checker: CombinationTester) -> List[List[int]]:
#     """
#     Checks the feasibility of a list of active set combinations, if infeasible add to the combination checker and returns all feasible active set combinations
#
#     :param program: An MPQP Program
#     :param set_list: The list of active sets
#     :param combination_checker: The combination checker that prunes
#     :return: The list of all feasible active sets
#     """
#     output = list()
#     for child in set_list:
#         if program.check_feasibility(child):
#             output.append(child)
#         else:
#             combination_checker.add_combo(child)
#
#     return output
#
# save = []
# start_time = time.time()
# for i in range(max_depth):
# # if there are no other active sets to check break out of loop
# # print(len(to_check))
#
#     future_sets = list()
#     # creates the list of feasible active sets
#     feasible_sets = check_child_feasibility(my_QCQP, to_check, murder_list)
#     for child_set in feasible_sets:
#
#         # soln = check_optimality(program, equality_indices=child_set)
#         # The active set is optimal try to build a critical region
#
#         # print(f"Child set: {child_set}")
#         # if soln is not None:
#         print('--'*50)
#         sol = my_QCQP.check_optimality(child_set) # 检查最优性
#         if sol is not None:
#             critical_region = 1
#             save.append(sol)
#             print(child_set)
#             # critical_region = gen_cr_from_active_set(program, child_set)  #TODO CR 是否是由 inactive set 组成?
#             # Check the dimensions of the critical region
#             if critical_region is not None:
#                 solution.add_region(critical_region)
#
#         # propagate sets
#
#         if i + 1 != max_depth:
#             future_sets.extend(generate_children_sets(child_set, my_QCQP.num_inequality_constraints(), murder_list))
#
#     to_check = future_sets
# end_time = time.time()
# print("耗时: {:.2f}秒".format(end_time - start_time))
#
# print('--'*60)
#
# for element in save:
#     print('**'*60)
#     for key, values in element.items():
#     # if isinstance(values, List):
#
#         print(f"The {key}: {values}")
# print(f"Number of Save is {len(save)}")


# 暂时注释掉，因为这里的代码是为了测试QCQP的求解结果是否正确。23.11.21
# 处理输出的排列顺序；
temp_1 = np.squeeze(net.res_gen.values)[:2]
temp_2 = np.squeeze(net.res_ext_grid.values)
print(f'o = {r_[temp_2[0], temp_1[0], temp_2[1], temp_1[1]]}')

A_x, b_x, A_l, b_l = my_QCQP.optimal_control_law()

p_load_ = zeros([2, 1])
p_load_[0] = 62.0
x_mpp = A_x @ p_load_ + b_x.reshape([len(b_x), 1])
print(f'x = {x_mpp.reshape(len(x_mpp))[-4:]}.')
# print(f'dh.T*x = {my_QCQP.dh.T @ x_mpp}')
# print(f'dh.T*x_0 = {my_QCQP.dh.T @ my_QCQP.x_0}')
# print(f'x_0 = {my_QCQP.x_0}')
# 对比不同负载下，三种求解的结果（pipopt, QCQP, MMP）
mu_ = my_QCQP.raw['lmbda']['mu']
idx_act = find(mu_ > 0).tolist()
cri_region = gen_cr_from_active_set(my_QCQP, active_set=idx_act)
print(f'CR = {cri_region}')