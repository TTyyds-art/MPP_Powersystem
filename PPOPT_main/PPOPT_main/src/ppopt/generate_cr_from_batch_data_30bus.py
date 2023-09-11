import time
from joblib import load
import time

import numpy as np
import torch
from joblib import load
from numpy import flatnonzero as find
from numpy import r_

import pandapower as pp
from pandapower.networks import case30
from ppopt_main.src.ppopt.mpQCQP_program_0731 import MPQCQP_Program
from ppopt_main.src.ppopt.solution_torch import SolutionTorch
from ppopt_main.src.ppopt.utils.mpqp_utils import gen_cr_from_active_set_torch_add

# 检查是否存在GPU，如果存在则使用，否则使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 设定随机数种子 numpy
# np.random.seed(1)
# torch.manual_seed(0)

loaded_dataset = torch.load('dataset_test_30_bus_torch/dataset_20230826_174101_N4000.pth', map_location=device)  # 加载数据集
ineq_keep_del = [34, 69, 70, 9, 105, 75, 109, 50, 114, 93, 122, 28, 29, 158]
net = case30()
pp.runopp(net, verbose=False)
p_load_init = net.load.p_mw.copy()
q_load_init = net.load.q_mvar.copy()
new_QCQP = MPQCQP_Program(net=net, lb_loads=r_[p_load_init, q_load_init] * 0.8,
                                  ub_loads=r_[p_load_init, q_load_init] * 1.2)
opf_anly_sol = SolutionTorch(program=new_QCQP, critical_regions=[])
# 开始时间
# %%  1. 输出结果
start_time = time.time()
batch_size = 16
# for iteration, item in enumerate(loaded_dataset):
#     train_dataloader = DataLoader(item, batch_size=batch_size, shuffle=True)
#
#     for batch_idx, batch_data in enumerate(train_dataloader):
#         print(f"Iter {iteration}; Batch {batch_idx + 1}:")
#         # A_x, b_x, A_l, b_l = optimal_control_law_torch(batch_data)
#         # if iteration == 0:
#             # ineq_keep_del = None
#         cri_region_list = gen_cr_from_active_set_torch(batch_data, ineq_keep_del)  # 生成critical region; 从GPU转移到CPU
#         opf_anly_sol.add_regions(cri_region_list)
#         num_regions = len(opf_anly_sol.critical_regions)
#         print(f"num_regions = {num_regions}")
            # print(f"cri_region[0].E = \n{cri_region_list[0].E[0]}\nlen of cri_region_list = {len(cri_region_list)}")
#         # 输出 A_x
#         # print(f"A_x = \n{A_x[:, :1]}")
#     #     # cri_region = gen_cr_from_active_set_torch(batch_data)  # 生成当前的critical region
#     #
#         # for key, value in batch_data.items():
#         #     # 选取每一批数据前5个示例
#         #     example_values = value[:3]
#         #
#         #     # 打印键值对
#         #     for example_idx, example_value in enumerate(example_values):
#         #         if key == 'idx_act':
#         #             print(f"{key} (Example {example_idx + 1}): {example_value[:5]}")
#         # print("-" * 50)  # 分隔线
#
# # 结束时间
# end_time = time.time()
# print(f"The total time is {end_time - start_time}")

# Save the instance using joblib
# file_path = 'solution_instance.joblib'
file_path = 'solution_instance_30bus_4000.joblib'
# dump(opf_anly_sol, file_path)

# Load the instance using joblib
loaded_opf_anly_sol = load(file_path)

# print(device)
# %%  2. 评估结果
# 创建lambda函数 将tensor转换为numpy
num_regions = len(loaded_opf_anly_sol.critical_regions)
print(f"Final num_regions = {num_regions}")
to_np = lambda x: x.cpu().detach().numpy()
batch_size = 8
n_t = 0
n_t_c = 0
# for iteration, item in enumerate(loaded_dataset): # 为什么分成了34个？有34种active set的长度
#     train_dataloader = DataLoader(item, batch_size=batch_size, shuffle=True)
#     # print(f"iteration: {iteration}")
#     for batch_idx, batch_data in enumerate(train_dataloader):
#
#         x_hat_batch = batch_data['x_0']
#         PQ_load_batch = batch_data['pq_load']
#
#         for idx, each_pq in enumerate(to_np(PQ_load_batch)): # 从GPU转移到CPU
#             n_t += 1
#             # print(f"each pq:\n{each_pq}")
#             x_star = loaded_opf_anly_sol.evaluate(each_pq)
#             if x_star is not None:
#                 n_t_c += 1
#                 x_star = x_star.squeeze()
#                 x_hat = to_np(x_hat_batch[idx])
#                 # x_hat[: int(len(x_hat) / 4)] = x_hat[: int(len(x_star) / 4)] * pi / 180
#                 # 计算样本的每个维度的误差
#                 error = max(np.abs(x_hat - x_star))
#                 print(f"{iteration}-{batch_idx}-{idx}")
#                 print(f"No. {n_t}-{n_t_c} Error: \n{error}")
error_count = 0
in_idx = 0
tot_idx = 0
for i in range(50000):
    def generate_combined_load(load_init):
        return np.array([val * (0.8 + 0.4 * np.random.rand()) for val in load_init]).reshape([-1, 1])

    combined_init = np.vstack((p_load_init, q_load_init))
    pq_load = generate_combined_load(combined_init)

    # 如果pq_load 不在当前的所有critical region中，那么就计算新的critical region
    if (not loaded_opf_anly_sol.get_region(pq_load)) or isinstance(loaded_opf_anly_sol.get_region(pq_load), dict):

        # 对于新的pq_load, 获得基于pandpower的结果
        net.load.p_mw = pq_load[:net.load.shape[0]]
        net.load.q_mvar = pq_load[net.load.shape[0]:]
        # 如果报错，就重新运行while循环
        try:
            pp.runopp(net, delta=1e-16, RETURN_RAW_DER=1)
            tot_idx += 1
        except:
            print("Error in running the optimal power flow.")
            continue
        print('No.{} pq_load is not inside the current critical region.'.format(tot_idx-in_idx))
        # 生成当前的critical region
        new_QCQP = MPQCQP_Program(net=net, lb_loads=r_[p_load_init, q_load_init] * 0.8,
                                  ub_loads=r_[p_load_init, q_load_init] * 1.2)
        mu_ = new_QCQP.raw['lmbda']['mu']
        idx_act = find(mu_ > 1e-3).tolist()  # 修改成了 1e-3, 因为对互补性的阈值是1e-6，阈值=z*mu/(1+max(x)) => z*mu<5e-6;mu>1e-3能够保证z<1e-3
        cri_region = gen_cr_from_active_set_torch_add(new_QCQP, active_set=idx_act, keep_ineq_indices=ineq_keep_del)  # 生成当前的critical region
        loaded_opf_anly_sol.add_regions(cri_region)
    else:
        print('No.{} pq_load is inside the current critical region.'.format(in_idx))
        tot_idx += 1
        in_idx += 1
        # x_star = loaded_opf_anly_sol.evaluate(pq_load)
        # if isinstance(x_star, dict):
        #     print(f"Distance {x_star.keys()}: {x_star.values()}")
        #     continue
        # net.load.p_mw = pq_load[:net.load.shape[0]]
        # net.load.q_mvar = pq_load[net.load.shape[0]:]
        # try:
        #     om, ppopt, raw = pp.runopp(net, delta=1e-16, RETURN_RAW_DER=1)
        # except:
        #     print("Error in running the optimal power flow.")
        #     continue
        # parameters_grid, update_dgdh, ieq_idxs = clear_om(om=om, net=net)
        #
        # x_0 = parameters_grid[-1]
        #
        # x_hat = np.expand_dims(x_0, axis=1)
        # if x_star is not None:
        #     error = max(np.abs(x_hat[60:] - x_star[60:]))
        #     print(f"Error: {error} No. {in_idx} ")
        # else:
        #     print(f"No. {in_idx} Error: \n{None}")

# dump(loaded_opf_anly_sol, file_path)