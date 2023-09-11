import pandapower as pp
import warnings
import numpy as np
from numpy import flatnonzero as find, zeros, r_

import sys, os
path_current = '/home/huzuntao/PycharmProjects/MPP_Powersystem/'
path_ = os.getcwd()
if path_current not in sys.path:
    sys.path.insert(1, '/home/huzuntao/PycharmProjects/MPP_Powersystem/')
elif path_ not in sys.path:
    sys.path.insert(1, path_)

from ppopt_main.PPOPT_main.src.ppopt.mpQCQP_program_0731 import MPQCQP_Program
from ppopt_main.PPOPT_main.src.ppopt.utils.mpqp_utils import gen_cr_from_active_set
# from ppopt_main.ppopt_main.src.ppopt.utils.general_utils import num_cpu_cores


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
p_load_o = [10, 15, 15, 15]
pp.create_load(net, bus2, p_mw=p_load_o[0], controllable=False)
pp.create_load(net, bus4, p_mw=p_load_o[1], controllable=False)
pp.create_load(net, bus5, p_mw=p_load_o[2], controllable=False)
pp.create_load(net, bus6, p_mw=p_load_o[3], controllable=False)
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

# 处理输出的排列顺序；
temp_1 = np.squeeze(net.res_bus)  # 获得net计算后的结果 23-08-01
print(f"bus: \n{temp_1}")
# temp_2 = np.squeeze(net.res_ext_grid.values)
# print(f'o = {r_[temp_2[0], temp_1[0], temp_2[1], temp_1[1]]}')

A_x, b_x, A_l, b_l = my_QCQP.optimal_control_law()  # 获得当前的系数

# 生成当前的critical region
mu_ = my_QCQP.raw['lmbda']['mu']
idx_act = find(mu_> 1e-3).tolist() # 修改成了 1e-3, 因为对互补性的阈值是1e-6，阈值=z*mu/(1+max(x)) => z*mu<5e-6;mu>1e-3能够保证z<1e-3
cri_region = gen_cr_from_active_set(my_QCQP, active_set=idx_act)    # 生成当前的critical region
# print(f'CR = {cri_region}')
p_load_new = p_load_o.copy()
flag_load = True
while flag_load:

    # 对p_load_o中的每一个值，随机乘以0.8到1.2之间的数
    for i in range(len(p_load_o)):
        p_load_new[i] = p_load_o[i] * (0.9 + 0.2 * np.random.rand())
    p_load_new = np.array(p_load_new).reshape([4, 1])  # 生成新的负载 [4, 1]
    pq_load = r_[p_load_new, zeros([4, 1])]  # 生成有功和无功负载 [8, 1]
    print(f'p_load_new = {p_load_new}')
    # 如果pq_load 在当前的critical region中，那么就可以直接计算出来
    if cri_region.is_inside(pq_load):
        flag_load = False
        print('pq_load is inside the current critical region.')
        x_mpp = cri_region.evaluate(pq_load)    # 获得由MPP计算而来的结果， 也就是x_mpp
    # x_mpp 是由[V,W, P, Q]组成的，而且是按照net的顺序排列的。 输出P和Q
        print(f'x = {x_mpp.reshape(len(x_mpp))[-4:]}.')

# 对于新的pq_load, 获得基于pandpower的结果
net.load.p_mw = p_load_new
pp.runopp(net)
# 输出power generator的结果
# temp_1 = np.squeeze(net.res_gen)  # 获得net计算后的结果 23-08-01
print(f"gen: \n{net.res_gen};\next_grid: \n{net.res_ext_grid}")


# # 对比不同负载下，三种求解的结果（pipopt, QCQP, MMP）


# cr_current = CriticalRegion(A_x, b_x, A_l, b_l, x_mpp, pq_load)  # 生成当前的critical region
# print(f'x = {x_mpp.reshape(len(x_mpp))[-4:]}.')
# print(f'dh.T*x = {my_QCQP.dh.T @ x_mpp}')
# print(f'dh.T*x_0 = {my_QCQP.dh.T @ my_QCQP.x_0}')
# print(f'x_0 = {my_QCQP.x_0}')