### required packages:
### Pytorch-1.0+ (if below 1.1, replace torch.solve with torch.gesv)
### scipy
### numpy

import torch


def scale_rows(L, b):
    norm_val = 1.0 / (torch.norm(L, dim=2, keepdim=True))  # 添加一个小的正值以避免除以零
    return [L * norm_val, b * norm_val]


def physarum_solve(A, b, c, x, step_size=1, max_iter = 10):
    '''
    返回结果的索引。result_idx 的每个元素表示 batch 中对应的 m 维度上的值是否都小于阈值：
    如果所有值都小于阈值，对应的元素为 True；否则为 False
    :param A: batch_size x m x n
    :param b: batch_size x m x 1
    :param c: batch_size x n
    :param x: initial solution, batch_size x n
    :param step_size: learning rate
    :param max_iter: \
    :return: result_idx: [1, 0, 0, ..., 1] (batch_size x 1)
    '''


    batch_size = A.shape[0]
    n = A.shape[2]
    if x is None:
        xs = torch.rand([batch_size, n]).cuda()
    else:
        xs = x.clone().detach().squeeze(2)
    for i in range(max_iter):
        W_diag = xs / c.squeeze(2)
        W = torch.diag_embed(W_diag.float())
        L = torch.matmul(torch.matmul(A, W), torch.transpose(A, 1, 2))
        # L, b = scale_rows(L, b)
        # 重新整理print（min,max）
        print(f"No.{i}\nA_max = {torch.max(A[0])}; A_min = {torch.min(A[0])};xs_max = {torch.max(xs[0])}; xs_min = {torch.min(xs[0])}"
              f";\n c_max={torch.max(c[0])};c_min={torch.min(c[0])}; W_max = {torch.max(W[0])}; W_min = {torch.min(W[0])};\n"
              f"L_max = {torch.max(L[0])}; L_min = {torch.min(L[0])}")

        print(f"L_det = {torch.linalg.det(L[0])}\nL_Contition = {torch.linalg.cond(L[0])}")
        print(f"A_Contition = {torch.linalg.cond(A[0])}")
        print(f"W_det = {torch.linalg.det(W[0])}\nW_Contition = {torch.linalg.cond(W[0])}")
        # 打印出完整的L
        # print(f"L = {L[0]}")

        # print(f"L = {L[0]}")

        p = torch.linalg.solve(L, b)  # use torch.gesv to replace torch.solve if below Pytorch1.1
        q = torch.matmul(torch.matmul(W, torch.transpose(A, 1, 2)), p)
        xs = (1 - step_size) * xs + step_size * q.squeeze(2)
        xs = torch.clamp(xs, min=1e-6, max=1e+4)

    # check if the solution is feasible
    # if not, return None
    # if yes, return the solution
    solution_temp = torch.matmul(A, xs.unsqueeze(2)) - b
    # solution_temp 的第一维度是batch_size，第二维度是m，第三维度是1, 我们比较m维度上的值>1e-3，所以输出的是一个batch_size x 1的矩阵
    # 假设 solution_temp 是一个形状为 (batch_size, m, 1) 的 PyTorch 张量

    # 定义阈值
    threshold = 1e-3

    # 检查 solution_temp 中的值是否大于阈值
    comparison = solution_temp < threshold

    # 检查 m 维度上的所有值是否都大于阈值
    # 输出是一个形状为 (batch_size, 1) 的布尔张量
    result_idx = torch.all(comparison, dim=1, keepdim=True)

    # result_idx 中的每个元素表示 batch 中对应的 m 维度上的值是否都小于阈值
    # 如果所有值都小于阈值，对应的元素为 True；否则为 False

    return result_idx
