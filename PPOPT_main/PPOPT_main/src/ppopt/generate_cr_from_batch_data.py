import time

import torch
from torch.utils.data import DataLoader
from test_30_bus_torch import QCQPDataset

from PPOPT_main.PPOPT_main.src.ppopt.utils.mpqp_utils_torch import optimal_control_law_torch, \
    gen_cr_from_active_set_torch

loaded_dataset = torch.load('dataset_test_30_bus_torch/dataset_20230826_174101_N4000.pth')  # 加载数据集

# 开始时间
start_time = time.time()
batch_size = 16
for iteration, item in enumerate(loaded_dataset):
    train_dataloader = DataLoader(item, batch_size=batch_size, shuffle=True)

    # %%  4. 输出结果
    for batch_idx, batch_data in enumerate(train_dataloader):
        print(f"Iter {iteration}; Batch {batch_idx + 1}:")
        # A_x, b_x, A_l, b_l = optimal_control_law_torch(batch_data)
        # if iteration == 10:
        cri_region_list = gen_cr_from_active_set_torch(batch_data)  # 生成当前的critical region
        print(f"cri_region[0].E = \n{cri_region_list[0].E[0]}\nlen of cri_region_list = {len(cri_region_list)}")
        # 输出A_x
        # print(f"A_x = \n{A_x[:, :1]}")
    #     # cri_region = gen_cr_from_active_set_torch(batch_data)  # 生成当前的critical region
    #
        for key, value in batch_data.items():
            # 选取每一批数据前5个示例
            example_values = value[:3]

            # 打印键值对
            for example_idx, example_value in enumerate(example_values):
                if key == 'idx_act':
                    print(f"{key} (Example {example_idx + 1}): {example_value}")

        print("-" * 50)  # 分隔线

# 结束时间
end_time = time.time()
print(f"The total time is {end_time - start_time}")