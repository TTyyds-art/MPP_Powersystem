import time

import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

def batch_linear_programming(A, b, c, learning_rate=0.1, max_iter=5000):


    A, b = A.to(device), b.to(device)

    batch_size, m, n = A.shape
    _, u, _ = b.shape

    # 初始化x
    x = torch.randn(batch_size, n, 1).to(device)

    for i in range(max_iter):
        # 计算约束违反
        constraint_violation1 = torch.relu(A.matmul(x))
        constraint_violation2 = (b[:, :, 0] * x[:, 0:1, 0] + b[:, :, 1] * x[:, 1:2, 0] + c).unsqueeze(2)

        # 创建一个掩码，以找出哪些批次的constraint_violation2非常接近零
        violation2 = torch.abs(constraint_violation2)
        # 计算梯度
        gradient1 = torch.matmul(A.transpose(1, 2), constraint_violation1)
        gradient2 = torch.cat(
            (b[:, :, 0].unsqueeze(2), b[:, :, 1].unsqueeze(2), torch.zeros(batch_size, m-b.shape[1], 1).to(device)), dim=1)

        # 更新x
        x = x - learning_rate * (gradient1 + gradient2*violation2)

        # 检查是否满足约束
        if torch.all(A.matmul(x) <= 1e-3) and torch.all(torch.abs(constraint_violation2.squeeze(2)) < 1e-4):
            print(f"Found a solution at iteration {i}")
            break
    else:
        print("No solution found within the max iteration")

    return x

def batch_linear_programming_mom(A, b, idx=None, x=None, learning_rate=0.1, max_iter=5000, tol1=1e-3, tol2=1e-4, beta=0.99):
    r"""
        This is the breakout for solving linear programs in batch. The linear program is of the following form:

        .. math::

            \min_{x} 0

        .. math::
            \begin{align}
            A[x] &\leq b\\
            A_{eq}[x] &= b_{eq}\\
            x &\in R^n\\
            \end{align}
    """

    A, b = A.to(device), b.to(device)

    batch_size, m, n = A.shape

    mask = idx.bool()
    A_equal = A[mask.squeeze(2)].view(batch_size, 1, n)  # batch_size x 1 x n # 选取等式约束的A,每次只有一个等式约束
    b_equal = b[mask.squeeze(2)].view(batch_size, 1, 1)  # batch_size x 1 x 1 # 选取等式约束的b,每次只有一个等式约束

    # 初始化 x 和动量
    if x is None:
        x = torch.randn(batch_size, n, 1).to(device).float()
    else:
        x = x.to(device).float()
    v = torch.zeros_like(x)

    satisfaction_record = torch.zeros((batch_size, 1), dtype=torch.bool).to(device)

    for i in range(max_iter):
        # 计算约束违反
        constraint_violation1 = torch.relu(A.matmul(x) - b_equal)
        constraint_violation2 = A_equal.matmul(x) - b_equal

        # 创建一个掩码，以找出哪些批次的constraint_violation2非常接近零
        violation2 = -constraint_violation2

        # 计算梯度
        gradient1 = torch.matmul(A.transpose(1, 2), constraint_violation1)
        #
        gradient2 = torch.matmul(A_equal.transpose(1, 2), constraint_violation2)

        # 计算动量更新
        v = beta * v + (1 - beta) * (gradient1 + gradient2)
        # m = beta * m + (1 - beta) * (gradient1 + gradient2) ** 2

        # 使用动量更新x
        x = x - learning_rate * v

        # 检查是否满足约束，记录每个batch的情况
        condition1 = torch.all(constraint_violation1.squeeze(2) <= tol1, dim=1)
        condition2 = torch.all(violation2.squeeze(2) < tol2, dim=1)
        satisfaction_record = (condition1 & condition2)

        # 检查是否所有批次都已满足约束条件，如果满足，则提前退出
        if torch.all(satisfaction_record):
            print(f"All batches found a solution at iteration {i}")
            break
    else:
        print("No solution found within the max iteration")

    return x, satisfaction_record


def test_batch_linear_programming_mom(batch_size=10, m=5, n=5):
    assert m <= n, "m should be less than or equal to n"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 生成随机测试数据
    A = torch.rand((batch_size, m, n), device=device)
    b = torch.rand((batch_size, m, 1), device=device)

    # 生成idx矩阵，每个batch中只有一行是1，其余是0
    idx = torch.zeros((batch_size, m, 1), device=device)
    for i in range(batch_size):
        idx[i, torch.randint(0, m, (1,)).item(), 0] = 1

    # 调用函数
    x, satisfaction_record = batch_linear_programming_mom(A, b, idx=idx)

    # 检查结果
    assert x.shape == (batch_size, n, 1)
    assert satisfaction_record.shape == (batch_size,)
    print(f"result = \n{torch.matmul(A,x)-b}")

    # 检查满足条件的记录
    print(f"Satisfaction record: {satisfaction_record}")

    # 可以添加更多的测试，包括测试满足特定约束的情况


if __name__ == '__main__':
    start_time = time.time()
    test_batch_linear_programming_mom(batch_size=256, m=600, n=600)
    end_time = time.time()
    print(f"The total time is {end_time - start_time}")
