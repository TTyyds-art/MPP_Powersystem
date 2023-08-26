# %%
import pandapower as pp
import warnings
import time
import numpy as np
from numpy import flatnonzero as find, ones, zeros, Inf, pi, exp, conj, r_
from PPOPT_main.PPOPT_main.src.ppopt.mp_solvers.solver_utils import \
    generate_children_sets, CombinationTester
from PPOPT_main.PPOPT_main.src.ppopt.solution import Solution
from PPOPT_main.PPOPT_main.src.ppopt.plot import plotly_plot
from PPOPT_main.PPOPT_main.src.ppopt.plot import parametric_plot
from multiprocessing.pool import ThreadPool as Pool

from typing import List
import sys, os
path_current = '/home/huzuntao/PycharmProjects/MPP_Powersystem/'
path_ = os.getcwd()
if path_current not in sys.path:
    sys.path.insert(1, '/home/huzuntao/PycharmProjects/MPP_Powersystem/')
elif path_ not in sys.path:
    sys.path.insert(1, path_)

from PPOPT_main.PPOPT_main.src.ppopt.mpQCQP_program_0731 import MPQCQP_Program
from PPOPT_main.PPOPT_main.src.ppopt.critical_region import CriticalRegion
from PPOPT_main.PPOPT_main.src.ppopt.utils.mpqp_utils import gen_cr_from_active_set
# from PPOPT_main.PPOPT_main.src.ppopt.utils.general_utils import num_cpu_cores


# 设定输出警告信息的方式
warnings.filterwarnings("ignore")  # 忽略掉所有警告

# 执行某些可能产生警告的代码
#
net = pp.create_empty_network()

# create buses
bus1 = pp.create_bus(net, vn_kv=110.)
bus2 = pp.create_bus(net, vn_kv=110.)
bus3 = pp.create_bus(net, vn_kv=110.)
# bus4 = pp.create_bus(net, vn_kv=110.)
# bus5 = pp.create_bus(net, vn_kv=110.)
# bus6 = pp.create_bus(net, vn_kv=110.)

# create 110 kV lines
# pp.create_line(net, bus4, bus5, length_km=90., std_type='149-AL1/24-ST1A 110.0')
# pp.create_line(net, bus3, bus4, length_km=90., std_type='149-AL1/24-ST1A 110.0')
pp.create_line(net, bus2, bus3, length_km=90., std_type='149-AL1/24-ST1A 110.0')
pp.create_line(net, bus1, bus2, length_km=70., std_type='149-AL1/24-ST1A 110.0')
# pp.create_line(net, bus6, bus3, length_km=70., std_type='149-AL1/24-ST1A 110.0')

# create loads
p_load_o = [70, 15, 15, 15]
q_load_o = [20, 10, 10, 10]
pp.create_load(net, bus2, p_mw=p_load_o[0], q_mvar=q_load_o[0] , controllable=False)
# pp.create_load(net, bus4, p_mw=p_load_o[1], controllable=False)
# pp.create_load(net, bus5, p_mw=p_load_o[2], controllable=False)
# pp.create_load(net, bus6, p_mw=p_load_o[3], controllable=False)
# create generators
eg = pp.create_ext_grid(net, bus1, min_p_mw=0, max_p_mw=1000, vm_pu=1.05)
g0 = pp.create_gen(net, bus3, p_mw=80, min_p_mw=0, max_p_mw=80, vm_pu=1.00, controllable=True)

costeg = pp.create_poly_cost(net, 0, 'ext_grid', cp1_eur_per_mw=20)
costgen1 = pp.create_poly_cost(net, 0, 'gen', cp1_eur_per_mw=10)
costgen2 = pp.create_poly_cost(net, 1, 'gen', cp1_eur_per_mw=10)

net.bus["min_vm_pu"] = 0.96
net.bus["max_vm_pu"] = 1.04
net.line["max_loading_percent"] = 100


# from scipy.io import loadmat

# ppc = loadmat('cases/pglib_opf_case3_lmbd.m')
# import pandapower.converter as tb
# .from_mpc
# Load MATPOWER case
# net = tb.from_mpc('cases/pglib_opf_case3_lmbd.mat')
num_load = len(net.load)
new_QCQP = MPQCQP_Program(net=net, lb_loads=r_[p_load_o[:1], q_load_o[:1]]*0.8, ub_loads=r_[p_load_o[:1], q_load_o[:1]]*1.2)
print(f"load: {net.load.p_mw}")
# 处理输出的排列顺序；
temp_1 = np.squeeze(net.res_bus)  # 获得net计算后的结果 23-08-01
print(f"bus: \n{temp_1}")
#
# A_x, b_x, A_l, b_l = my_QCQP.optimal_control_law()  # 获得当前的系数
#
# %%
# 生成当前的critical region
mu_ = new_QCQP.raw['lmbda']['mu']
idx_act = find(mu_> 1e-3).tolist()  # 修改成了 1e-3, 因为对互补性的阈值是1e-6，阈值=z*mu/(1+max(x)) => z*mu<5e-6;mu>1e-3能够保证z<1e-3
cri_region = gen_cr_from_active_set(new_QCQP, active_set=idx_act)    # 生成当前的critical region
#
p_load_new = p_load_init = net.load.p_mw.copy()
q_load_new = q_load_init = net.load.q_mvar.copy()
flag_load = True

while flag_load:
    p_load_init = p_load_o[:1]
    q_load_init = q_load_o[:1]
    for i in range(len(p_load_init)):  # 对p_load_o中的每一个值，随机乘以0.8到1.2之间的数
        p_load_new[i] = 76 * (0.9 + 0.2 * np.random.rand())
    p_load_new = np.array(p_load_new).reshape([num_load, 1])  # 生成新的负载 [4, 1]
    for i in range(len(q_load_init)):  # 对p_load_o中的每一个值，随机乘以0.8到1.2之间的数
        q_load_new[i] = 16.5 * (0.9 + 0.2 * np.random.rand())
    q_load_new = np.array(q_load_new).reshape([num_load, 1])  # 生成新的负载 [4, 1]
    pq_load = r_[p_load_new, q_load_new]  # 生成有功和无功负载 [8, 1]

    print(f'pq_load  = {pq_load}')

    # 如果pq_load 在当前的critical region中，那么就可以直接计算出来
    if cri_region.is_inside(pq_load):
        flag_load = False
        print('pq_load is inside the current critical region.')
        x_mpp = cri_region.evaluate(pq_load)    # 获得由MPP计算而来的结果， 也就是x_mpp
        # x_mpp 是由[V,W, P, Q]组成的，而且是按照net的顺序排列的。 输出P和Q
        print(f'x = {x_mpp.reshape(len(x_mpp))[:]}.')

# 对于新的pq_load, 获得基于pandpower的结果
net.load.p_mw = p_load_new
pp.runopp(net)
# 输出power generator的结果
print(f"bus: \n {net.res_bus}\ngen from Pandapower: \n{net.res_gen};\nExt_grid from Pandapower: \n{net.res_ext_grid}")
# %% solution
opf_anly_sol = Solution(program=new_QCQP, critical_regions=[cri_region])

Add_flag = 0
Add_tot = 500
iden_points = [r_[p_load_o[:1], q_load_o[:1]]]
while Add_flag <= Add_tot:
    Add_flag += 1
    print(f"Add_flag = {Add_flag}")

    # 生成新的load
    for i in range(len(p_load_init)):  # 对 p_load_o 中的每一个值，随机乘以0.8到1.2之间的数
        # 如果p_load_init[i]的值不是10， 报错
        if p_load_init[i] != 70:
            raise ValueError("p_load_init[i] != 70")
        p_load_new[i] = p_load_init[i] * (0.8 + 0.4 * np.random.rand())
    p_load_new = np.array(p_load_new).reshape([num_load, 1])  # 生成新的负载 [3, 1]
    for i in range(len(q_load_init)):  # 对p_load_o中的每一个值，随机乘以0.8到1.2之间的数
        q_load_new[i] = q_load_init[i] * (0.9 + 0.2 * np.random.rand())
    q_load_new = np.array(q_load_new).reshape([num_load, 1])  # 生成新的负载 [4, 1]
    pq_load = r_[p_load_new, q_load_new]  # 生成有功和无功负载 [8, 1]  # 生成有功和无功负载 [6, 1]
    print(f'pq_load  = {pq_load}')

    print(f"The region: \n {bool(opf_anly_sol.get_region(pq_load))}")
    # 如果pq_load 不在当前的所有critical region中，那么就计算新的critical region
    if not opf_anly_sol.get_region(pq_load):
        print('pq_load is not inside the current critical region.')
        # 将这个点加入到iden_points中
        iden_points.append(pq_load)

        # 对于新的pq_load, 获得基于pandpower的结果
        net.load.p_mw = p_load_new
        net.load.q_mvar = q_load_new
        # 如果报错，就重新运行while循环
        try:
            pp.runopp(net, delta=1e-16, RETURN_RAW_DER=1)
        except:
            print("Error in running the optimal power flow.")
            continue

        # 生成当前的critical region
        new_QCQP = MPQCQP_Program(net=net, lb_loads=r_[p_load_init, q_load_init]*0.8, ub_loads=r_[p_load_init, q_load_init]*1.2)
        mu_ = new_QCQP.raw['lmbda']['mu']
        idx_act = find(mu_ > 1e-3).tolist()  # 修改成了 1e-3, 因为对互补性的阈值是1e-6，阈值=z*mu/(1+max(x)) => z*mu<5e-6;mu>1e-3能够保证z<1e-3
        cri_region = gen_cr_from_active_set(new_QCQP, active_set=idx_act)  # 生成当前的critical region
        opf_anly_sol.add_region(cri_region)

print(f"The number of critical regions is {len(opf_anly_sol.critical_regions)}")
# %% 画图
# for region in opf_anly_sol.critical_regions:
parametric_plot(solution=opf_anly_sol, save_path=None, points=iden_points)   # 能处理高维的吗, 只能处理2维的
# '/home/huzuntao/PycharmProjects/MPP_Powersystem/PPOPT_main/PPOPT_main/src/ppopt'
