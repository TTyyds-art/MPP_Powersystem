# %% 0. 导入所需的包
import pandapower as pp
import warnings
import time
import numpy as np
from numpy import flatnonzero as find, r_

from PPOPT_main.PPOPT_main.src.ppopt.solution import Solution

import sys, os
path_current = '/home/huzuntao/PycharmProjects/MPP_Powersystem/'
path_ = os.getcwd()
if path_current not in sys.path:
    sys.path.insert(1, '/home/huzuntao/PycharmProjects/MPP_Powersystem/')
elif path_ not in sys.path:
    sys.path.insert(1, path_)

from PPOPT_main.PPOPT_main.src.ppopt.mpQCQP_program_0731 import MPQCQP_Program
from PPOPT_main.PPOPT_main.src.ppopt.utils.mpqp_utils import gen_cr_from_active_set

# 设定输出警告信息的方式
warnings.filterwarnings("ignore")  # 忽略掉所有警告

# %% 1. 118-bus system 跑通
from pandapower.networks import case30
net = case30()

# %% 1.1 使用Pandapower求解该问题
pp.runopp(net, verbose=False)
print(net.res_bus[:3])
# 还可以打印其他你感兴趣的结果，如线路负载、变压器等等

p_load_o = net.res_load.p_mw.values
q_load_o = net.res_load.q_mvar.values
num_load = len(p_load_o)
# %% 1.2 使用MPQCQP_Program求解该问题，其中包含GUROBI求解器
new_QCQP = MPQCQP_Program(net=net, lb_loads=r_[p_load_o, q_load_o]*0.8, ub_loads=r_[p_load_o, q_load_o]*1.2)
print(f"load: {net.load.p_mw[:5]}")
# 处理输出的排列顺序；
temp_1 = np.squeeze(net.res_bus)  # 获得net通过Pandapower计算后的结果
print(f"bus: \n{temp_1[:5]}")

# %% 2. 生成当前的critical region
# 设置开始的时间
start_time = time.time()

mu_ = new_QCQP.raw['lmbda']['mu']
idx_act = find(mu_> 1e-3).tolist() # 修改成了 1e-3, 因为对互补性的阈值是1e-6，阈值=z*mu/(1+max(x)) => z*mu<5e-6;mu>1e-3能够保证z<1e-3
cri_region = gen_cr_from_active_set(new_QCQP, active_set=idx_act)    # 生成当前的critical region
#输出cri_region E and f
print(f"cri_region.E = \n{cri_region.E}")
print(f"cri_region.f = \n{cri_region.f}")

p_load_new = p_load_init = net.load.p_mw.copy()
q_load_new = q_load_init = net.load.q_mvar.copy()
flag_load = True

 # %%  3. 生成最终的solution
opf_anly_sol = Solution(program=new_QCQP, critical_regions=[cri_region])

Add_flag = 0
Add_tot = 40000
error_count = 0
while Add_flag <= Add_tot:
    Add_flag += 1
    print(f"Add_flag = {Add_flag}")

    # 生成新的load
    for i in range(num_load):  # 对 p_load_o 中的每一个值，随机乘以0.8到1.2之间的数
        p_load_new[i] = p_load_init[i] * (0.9 + 0.2 * np.random.rand())
    p_load_new = np.array(p_load_new).reshape([num_load, 1])  # 生成新的负载 [3, 1]
    for i in range(num_load):  # 对p_load_o中的每一个值，随机乘以0.8到1.2之间的数
        q_load_new[i] = q_load_init[i] * (0.9 + 0.2 * np.random.rand())
    q_load_new = np.array(q_load_new).reshape([num_load, 1])  # 生成新的负载 [4, 1]
    pq_load = r_[p_load_new, q_load_new]  # 生成有功和无功负载 [8, 1]  # 生成有功和无功负载 [6, 1]
    # print(f'pq_load  = {pq_load.squeeze(axis=1)[:5]}')

    # print(f"The region: {bool(opf_anly_sol.get_region(pq_load))}")
    # 如果pq_load 不在当前的所有critical region中，那么就计算新的critical region
    if not opf_anly_sol.get_region(pq_load):
        print('pq_load is not inside the current critical region.')
        # 将这个点加入到iden_points中

        # 对于新的pq_load, 获得基于pandpower的结果
        net.load.p_mw = p_load_new
        # 如果报错，就重新运行while循环
        try:
            pp.runopp(net, delta=1e-16, RETURN_RAW_DER=1)
        except:
            error_count += 1
            print(f"Error No.{error_count} happens in running the OPF.")
            continue

        # 生成当前的critical region
        new_QCQP = MPQCQP_Program(net=net, lb_loads=r_[p_load_init, q_load_init]*0.9, ub_loads=r_[p_load_init, q_load_init]*1.1)
        mu_ = new_QCQP.raw['lmbda']['mu']
        idx_act = find(mu_ > 1e-3).tolist()  # 修改成了 1e-3, 因为对互补性的阈值是1e-6，阈值=z*mu/(1+max(x)) => z*mu<5e-6;mu>1e-3能够保证z<1e-3
        cri_region = gen_cr_from_active_set(new_QCQP, active_set=idx_act)  # 生成当前的critical region
        opf_anly_sol.add_region(cri_region)

# 设置结束的时间，并输出总的消耗时间
end_time = time.time()
print(f"The total time is {end_time - start_time}")
print(f"The number of critical regions is {len(opf_anly_sol.critical_regions)}\nError_count = {error_count}")


