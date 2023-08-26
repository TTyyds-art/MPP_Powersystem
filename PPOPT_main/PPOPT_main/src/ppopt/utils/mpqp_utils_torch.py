from dataclasses import dataclass
from typing import List, Optional, Tuple, Iterable

import numpy
from typing import Dict
import gurobipy as gp
from gurobipy import GRB

import torch
from torch import zeros, ones
from torch.linalg import pinv
from torch.utils.data import DataLoader

from .chebyshev_ball import chebyshev_ball
from ..critical_region import CriticalRegion
from ..mpQCQP_program import MPQCQP_Program
from ..solver import Solver
from ..utils.constraint_utilities import cheap_remove_redundant_constraints, remove_duplicate_rows

# 检查是否存在GPU，如果存在则使用，否则使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ppopt_block(mat_list):
    if not isinstance(mat_list[0], list):
        mat_list = [mat_list]

    x_size = sum(el.shape[1] for el in mat_list[0])
    y_size = sum(el[0].shape[0] for el in mat_list)

    output_data = numpy.zeros((y_size, x_size))

    x_cursor = 0
    y_cursor = 0

    for mat_row in mat_list:
        y_offset = 0

        for matrix_ in mat_row:
            shape_ = matrix_.shape
            output_data[y_cursor: y_cursor + shape_[0], x_cursor: x_cursor + shape_[1]] = matrix_
            x_cursor += shape_[1]
            y_offset = shape_[0]

        y_cursor += y_offset
        x_cursor = 0

    return output_data

def ppopt_block_torch(mat_list):
    if not isinstance(mat_list[0], list):
        mat_list = [mat_list]

    batch_size = mat_list[0][0].shape[0]
    x_size = sum(el.shape[2] for el in mat_list[0])
    y_size = sum(el[0].shape[1] for el in mat_list)

    output_data = zeros((batch_size, y_size, x_size))

    x_cursor = 0
    y_cursor = 0

    for mat_row in mat_list:
        y_offset = 0

        for matrix_ in mat_row:
            shape_ = matrix_.shape[1:]
            output_data[:, y_cursor: y_cursor + shape_[0], x_cursor: x_cursor + shape_[1]] = matrix_
            x_cursor += shape_[0]
            y_offset = shape_[1]

        y_cursor += y_offset
        x_cursor = 0

    return output_data

def scale_constraint(A: numpy.ndarray, b: numpy.ndarray) -> List[numpy.ndarray]:
    """
    Normalizes constraints based on the L2 norm.

    :param A: LHS Matrix constraint
    :param b: RHS column vector constraint
    :return: a list [A_scaled, b_scaled] of normalized constraints
    """
    norm_val = 1.0 / numpy.linalg.norm(A, axis=1, keepdims=True)
    return [A * norm_val, b * norm_val]

def scale_constraint_torch(A: torch.Tensor, b: torch.Tensor) -> list:
    """
    Normalizes constraints based on the L2 norm for tensors.

    :param A: LHS Matrix constraint
    :param b: RHS column vector constraint
    :return: a list [A_scaled, b_scaled] of normalized constraints
    """
    norm_val = 1.0 / torch.norm(A, dim=2, keepdim=True)
    return [A * norm_val, b * norm_val]

def get_boundary_types(region: numpy.ndarray, omega: numpy.ndarray, lagrange: numpy.ndarray, regular: numpy.ndarray) -> \
        List:
    """
    Classifies the boundaries of a polytope into Omega constraints, Lagrange multiplier = 0 constraints, and Activated program constraints

    :param region:
    :param omega:
    :param lagrange:
    :param regular:
    :return:
    """

    num_constraints = region.shape[0]

    is_labeled = numpy.zeros(num_constraints)

    def label(compare):
        output = list()
        output_2 = list()
        for i in range(num_constraints):
            for j in range(compare.shape[0]):
                if is_labeled[i] == 0:
                    if numpy.allclose(region[i], compare[j]):
                        is_labeled[i] = 1
                        output.append(i)
                        output_2.append(j)
        return output, output_2

    omega_list = label(omega)
    lagrange_list = label(lagrange)
    regular_list = label(regular)

    return [omega_list, lagrange_list, regular_list]


def build_suboptimal_critical_region(program: MPQCQP_Program, active_set: List[int]):
    """
    Builds the critical region without considering culling facets or any other operation.
    Primary uses for this is based on culling lower dimensional feasible sets.

    :param program: the MQMP_Program to be solved
    :param active_set: the active set combination to build this critical region from
    :return: Returns the associated critical region if fully dimensional else returns None
    """
    inactive = [i for i in range(program.num_constraints()) if i not in active_set]

    parameter_A, parameter_b, lagrange_A, lagrange_b = program.optimal_control_law(active_set)

    # reduced constraints
    omega_A, omega_b = program.A_t, program.b_t
    lambda_A, lambda_b = cheap_remove_redundant_constraints(-lagrange_A, lagrange_b)

    # x as a function of theta representation of the inactive constraints
    inactive_A = program.A[inactive] @ parameter_A - program.F[inactive]
    inactive_b = program.b[inactive] - program.A[inactive] @ parameter_b

    # reduce these constraints
    inactive_A, inactive_b = cheap_remove_redundant_constraints(inactive_A, inactive_b)

    constraints_A = ppopt_block([[omega_A], [lambda_A], [inactive_A]])
    constraints_b = ppopt_block([[omega_b], [lambda_b], [inactive_b]])

    # combine them together
    region_A, region_b = remove_duplicate_rows(constraints_A, constraints_b)

    return region_A, region_b


def gen_cr_from_active_set_torch(batch_data: Dict[str, torch.Tensor]):

    """
    Builds the critical region of the given mpqp from the active set.

    :param program: the MQMP_Program to be solved
    :param active_set: the active set combination to build this critical region from
    :param check_full_dim: Keyword Arg, if true will return null if the region has lower dimensionality
    :return: Returns the associated critical region if fully dimensional else returns None
    """
    cr_region_list = []
    batch_size = batch_data['x_0'].shape[0]
    num_equality = batch_data['num_eq_conts']
    mu_ = batch_data['mu']
    active_set = batch_data['idx_act']  # 将tensor转换为list
    num_ineq_conts = batch_data['num_ineq_conts']

    # Create a 2D tensor with all possible indices repeated for each batch
    all_indices = torch.arange(num_ineq_conts[0], device=active_set.device).unsqueeze(0).repeat(active_set.shape[0], 1)

    # Create a mask of active indices for each batch
    mask = torch.zeros_like(all_indices, dtype=torch.bool)
    for i in range(active_set.shape[0]):
        mask[i, active_set[i].long()] = 1

    # Use the inverse of the mask to get the inactive indices
    inactive_set = all_indices[~mask].view(batch_size, -1)

    # 这里的lagrange_A对应激活的不等式约束和等式约束
    parameter_A, parameter_b, lagrange_A, lagrange_b = optimal_control_law_torch(batch_data)

    #
    lambda_A_torch, lambda_b_torch = -lagrange_A[:, num_equality[0]:], lagrange_b[:, num_equality[0]:]  # num_equality等第一个维度是batch_size

    # Theta(loads) Constraints
    omega_A_torch, omega_b_torch = batch_data['A_t'], batch_data['b_t']
    # omega_b = numpy.expand_dims(omega_b, axis=1)  #该omega_b本来是不增加维度的，但报错。所以在上面加queeze(),在此加expand_dims()

    # Inactive Constraints remain inactive; 这里的 program.A/b 都是不等式约束，没有等式约束
    A_inact = torch.gather(batch_data['A'], 1, inactive_set.unsqueeze(2).expand(-1, -1, batch_data['A'].shape[-1]))
    F_inact = torch.gather(batch_data['F'], 1, inactive_set.unsqueeze(2).expand(-1, -1, batch_data['F'].shape[-1]))
    b_inact = torch.gather(batch_data['b'], 1, inactive_set)
    inactive_A_torch = A_inact @ parameter_A - F_inact # inactive的索引需要处理
    inactive_b_torch = b_inact.unsqueeze(2) - A_inact @ parameter_b

    # # we need to check for zero rows
    # lamba_nonzeros = lambda_A.abs().sum(dim=2) > 0
    # indices_lam = torch.nonzero(lamba_nonzeros, as_tuple=True)[1].view(batch_size, -1)
    # indices_A = indices_lam.unsqueeze(-1).expand(-1, -1, lambda_A.shape[-1])
    #
    # # Block of all critical region constraints
    # lambda_Anz = torch.gather(lambda_A, 1, indices_A)
    # indices_b = indices_lam.unsqueeze(-1).expand(-1, -1, lambda_b.shape[-1])
    # lambda_bnz = torch.gather(lambda_b, 1, indices_b)
    #
    # ineq_nonzeros = inactive_A.abs().sum(dim=2) > 0
    # indices_ineq = torch.nonzero(ineq_nonzeros, as_tuple=True)[1].view(batch_size, -1)
    # indices_A_ineq = indices_ineq.unsqueeze(-1).expand(-1, -1, inactive_A.shape[-1])
    # inactive_Anz = torch.gather(inactive_A, 1, indices_A_ineq)  # 同上
    # indices_b_ineq = indices_ineq.unsqueeze(-1).expand(-1, -1, inactive_b.shape[-1])
    # inactive_bnz = torch.gather(inactive_b, 1, indices_b_ineq)

    # CR_A = ppopt_block_torch([[lambda_Anz], [inactive_Anz], [omega_A]])
    # CR_b = ppopt_block_torch([[lambda_bnz], [inactive_bnz], [omega_b]])  # 使每个matrix都是2D的；

    # CR_As, CR_bs = scale_constraint_torch(CR_A, CR_b)

    # if check_full_dim is set check if region is lower dimensional if so return None
    # if check_full_dim:
    #     # if the resulting system is not fully dimensional return None
    #     A_temp = numpy.ones_like(CR_As)*10
    #     b_temp = numpy.zeros_like(CR_bs) + numpy.ones_like(CR_bs)*0.2
    #     if not is_full_dimensional(CR_As, CR_bs, program.solver):
    #         return None

    # if it is fully dimensional we get to classify the constraints and then reduce them (important)!
    to_numpy = lambda x: x.cpu().detach().numpy()


    # iterate over the non-zero lagrange constraints
    for batch_idx in range(batch_size):

        tensor_list = [lambda_A_torch, lambda_b_torch, inactive_A_torch, inactive_b_torch, omega_A_torch, omega_b_torch]
        lambda_A, lambda_b, inactive_A, inactive_b, omega_A_batch, omega_b_batch = [
            to_numpy(tensor[batch_idx]) for tensor in tensor_list]

        lamba_nonzeros = [i for i, t in enumerate(lambda_A) if numpy.nonzero(t)[0].shape[0] > 0]
        ineq_nonzeros = [i for i, t in enumerate(inactive_A) if numpy.nonzero(t)[0].shape[0] > 0]

        # Block of all critical region constraints

        lambda_Anz = lambda_A[lamba_nonzeros]
        lambda_bnz = lambda_b[lamba_nonzeros]

        inactive_Anz = inactive_A[ineq_nonzeros]
        inactive_bnz = inactive_b[ineq_nonzeros]

        kept_lambda_indices = []
        kept_inequality_indices = []
        kept_omega_indices = []


        CR_As_batch = ppopt_block([[lambda_Anz], [inactive_Anz], [omega_A_batch]])
        CR_bs_batch = ppopt_block([[lambda_bnz], [inactive_bnz], [omega_b_batch]])
        CR_As_batch, CR_bs_batch = scale_constraint(CR_As_batch, CR_bs_batch)
        for index in range(len(lamba_nonzeros)):

            sol = solve_lp_gurobi_torch(None, None, CR_As_batch, CR_bs_batch, [index])

            if sol is not None:
                kept_lambda_indices.append(index)

        # iterate over the non-zero inequaltity constraints
        for index in range(len(ineq_nonzeros)):

            sol = solve_lp_gurobi_torch(None,None, CR_As_batch, CR_bs_batch, [index + len(lamba_nonzeros)])

            if sol is not None:
                kept_inequality_indices.append(index)

        # iterate over the omega constraints
        for index in range(omega_A_batch.shape[1]):

            sol = solve_lp_gurobi_torch(None, None, CR_As_batch, CR_bs_batch, [index + len(lamba_nonzeros) + len(ineq_nonzeros)])

            if sol is not None:
                kept_omega_indices.append(index)

    # create out reduced Critical region constraint block
    # 这一部分不能使用GPU，因为kept_lambda_indices等的长度不一样
        CR_As = ppopt_block(
            [[lambda_Anz[kept_lambda_indices]], [inactive_Anz[kept_inequality_indices]], [omega_A_batch[kept_omega_indices]]])
        CR_bs = ppopt_block(
            [[lambda_bnz[kept_lambda_indices]], [inactive_bnz[kept_inequality_indices]], [omega_b_batch[kept_omega_indices]]])

        # recover the lambda boundaries that remain
        relevant_lambda = [active_set[batch_idx][index] for index in kept_lambda_indices]

        real_regular = [inactive_set[batch_idx][index] for index in kept_inequality_indices]
        regular = real_regular

        # remove any possible duplicate constraints
        # and rescale since we did not rescale this particular set of constraints!!!
        CR_As, CR_bs = remove_duplicate_rows(CR_As, CR_bs)
        CR_As, CR_bs = scale_constraint(CR_As, CR_bs)

        cr_region = CriticalRegion(parameter_A, parameter_b, lagrange_A, lagrange_b, CR_As, CR_bs, active_set,
                          kept_omega_indices, relevant_lambda, regular)
        cr_region_list.append(cr_region)

    return cr_region_list

def optimal_control_law_torch(batch_data, active_set: List[int] = None) -> Tuple:
    r"""
    This function calculates the optimal control law corresponding to an active set combination

    :param active_set: an active set combination
    :return: a tuple of the optimal x* and λ* functions in the following form(A_x, b_x, A_l, b_l)

    .. math::

        \begin{align*}
        x^*(\theta) &= A_x\theta + b_x\\
        \lambda^*(\theta) &= A_l\theta + b_l\\
        \end{align*}
    """

    if active_set is None:  # 如果没有给定active_set，就用 mu来判断当前的 active_set
        active_set = batch_data['idx_act'].long()

    mu = batch_data['mu']
    mu_act = torch.gather(mu, 1, active_set)  # 从mu中取出active_set对应的元素
    labda = batch_data['lam']
    n_c = batch_data['dg'].shape[2] + active_set.shape[1]  # 等式约束和激活的不等式约束之和. shape[0]是batch_size
    batch_size = batch_data['dg'].shape[0]
    #
    dh_iq, dg_eq = batch_data['dh'], batch_data['dg']  # dh_iq, dg_eq 分别指的是不等式约束和等式约束的导数


    # active_set的形状为[batch_size, num_active_constraints]，且每个batch中的active_set的长度一样
    batch_indices, nx_indices, active_set_indices = torch.meshgrid(
        torch.arange(dh_iq.shape[0], device=dh_iq.device),
        torch.arange(dh_iq.shape[1], device=dh_iq.device),
        torch.arange(active_set.shape[1], device=dh_iq.device)
    )

    # 使用这三个扩展的索引从dh_iq中获取激活约束
    dh_iq_ac = dh_iq[batch_indices, nx_indices, active_set[batch_indices, active_set_indices]]
    # 现在，dh_iq_ac的形状应该为[batch_size, n_x, num_active_constraints]
    transposed_shape = dh_iq_ac.shape[-2:][::-1]
    mu_b = mu_act.reshape(mu_act.shape[0], mu_act.shape[1], 1).expand(-1, *transposed_shape)

    M = torch.cat([
        torch.cat([batch_data['Lxx'], dg_eq, dh_iq_ac], dim=2),
        torch.cat([torch.cat([dg_eq.permute(0, 2, 1), -(mu_b * dh_iq_ac.permute(0, 2, 1))], dim=1), torch.zeros((batch_size, n_c, n_c)).to(device)], dim=2)
    ], dim=1)  # M 的维度是 (n_x + n_c, n_x + n_c); z = [x, mu, labda[active_set]]; n_z = n_x + n_c #TODO 0519 加M的公式(次要)

    inverse_M = pinv(M)

    # print(f"the  M is {M}; \n The inverse_M is {inverse_M}")

    # 计算 N; N 的维度是 (n_x + n_c, n_pl+n_ql)
    Pd = batch_data['Pd']
    Qd = batch_data['Qd']
    idx_load = batch_data['idx_load']  # idx_load 是一个tensor，里面是所有有功负载的索引, [batch_size, num_nonzeros]
    n_load = idx_load.shape[1]  # 有功负载的个数

    n_x = batch_data['Lxx'].shape[1]
    # L_sl_x 的形状是 (2*n_load, n_x), 因为有 2*n_load 个负载（有功加无功），n_x个决策变量
    L_sl_x = zeros((batch_size, 2 * n_load, n_x)).to(device)
    # gh_sl 的形状是 (2*n_load, n_c), 因为有 2*n_load 个负载（有功加无功），n_c个约束
    gh_sl = zeros((batch_size, 2 * n_load, n_c)).to(device)
    for i in range(n_load):
        gh_sl[:, i, idx_load[:, i]] = 1
        gh_sl[:, i + n_load, idx_load[:, i] + batch_data['nb'][0]] = 1  # 此处的batch_data['nb']是batch_size个元素的tensor，所以要加[0]

    # labda_b_T = np.broadcast_to(labda.reshape((1, len(labda))), (2 * n_load, len(labda)))
    # mu_b_T = np.broadcast_to(mu.reshape((batch_size, 1, mu.shape[1])), (batch_size, 2 * n_load, mu.shape[1]))
    mu_b_T = mu_act.unsqueeze(1).expand(batch_size, 2 * n_load, mu_act.shape[1])

    lab_mu = torch.cat([ones([batch_size, labda.shape[1], 2 * n_load]).to(device), mu_b_T.permute(0, 2, 1)], dim=1).permute(0, 2, 1) # labda部分使用ones填充，mu部分使用mu_b_T填充是因为它对应的乘子都是0，所以任何数字都没有意义
    GH_sl = - lab_mu * gh_sl  # GH_sl 的形状是 (batch_size, 2*n_load, n_c); '*'可以在torch中完成element-wise的乘法

    N = torch.cat([L_sl_x, GH_sl], dim=2).permute(0, 2, 1)  # N 的形状是 (batch_size, 2*n_load, n_x + n_c).T = (batch_size, n_x + n_c, 2*n_load)

    A_x = (inverse_M @ N)[:, :n_x]  # A_x 的形状是 (batch_size, n_x, 2*n_load)
    A_l = (inverse_M @ N)[:, n_x:]  # A_l 的形状是 (batch_size, n_c, 2*n_load)

    x_0 = batch_data['x_0']
    Pd_load, Qd_load = torch.gather(Pd, 1, idx_load), torch.gather(Qd, 1, idx_load)
    b_x = x_0.unsqueeze(2) - A_x @ torch.cat([Pd_load, Qd_load], dim=1).unsqueeze(2)
    b_l = torch.cat([labda, mu_act], dim=1).unsqueeze(2) - A_l @ torch.cat([Pd_load, Qd_load], dim=1).unsqueeze(2)  # TODO 注意 active_set 的长度是变化的

    return A_x, b_x, A_l, b_l

def solve_lp_gurobi_torch(Q: numpy.ndarray = None, c: numpy.ndarray = None, A: numpy.ndarray = None,
                      b: numpy.ndarray = None,
                      equality_constraints: Iterable[int] = None,
                      bin_vars: Iterable[int] = None, verbose: bool = False,
                      get_duals: bool = True):
    r"""
    This is the breakout for solving mixed integer quadratic programs with gruobi

    The Mixed Integer Quadratic program programming problem

    .. math::

        \min_{xy} \frac{1}{2} [xy]^TQ[xy] + c^T[xy]

    .. math::
        \begin{align}
        A[xy] &\leq b\\
        A_{eq}[xy] &= b_{eq}\\
        x &\in R^n\\
        y &\in \{0, 1\}^m
        \end{align}

    :param Q: Square matrix, can be None
    :param c: Column Vector, can be None
    :param A: Constraint LHS matrix, can be None
    :param b: Constraint RHS matrix, can be None
    :param equality_constraints: List of Equality constraints
    :param bin_vars: List of binary variable indices
    :param verbose: Flag for output of underlying Solver, default False
    :param get_duals: Flag for returning dual variable of problem, default True (false for all mixed integer models)

    :return: A dictionary of the Solver outputs, or none if infeasible or unbounded. \\n output['sol'] = primal
    variables, output['dual'] = dual variables, output['obj'] = objective value, output['const'] = slacks,
    output['active'] = active constraints.
    """
    model = gp.Model()

    if not verbose:
        model.setParam("OutputFlag", 0)

    if equality_constraints is None:
        equality_constraints = []

    if bin_vars is None:
        bin_vars = []

    if len(bin_vars) == 0:
        model.setParam("Method", 0)

    if len(bin_vars) == 0 and Q is None:
        model.setParam("Method", 0)
        model.setParam("Quad", 0)

    # in the case of non-convex QPs add the non-convex flag, set the MIP gap the 0 we want exact solutions
    if Q is not None:
        if numpy.min(numpy.linalg.eigvalsh(Q)) < 0:
            model.Params.NonConvex = 2
            # noinspection SpellCheckingInspection
            model.Params.MIPgap = 0

    # define num variables and num constraints variables
    num_vars, num_constraints = get_program_parameters(Q, c, A, b)

    if A is None and Q is None:
        return None

    var_types = [GRB.BINARY if i in bin_vars else GRB.CONTINUOUS for i in range(num_vars)]
    # noinspection PyTypeChecker
    x = model.addMVar(num_vars, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=var_types)

    if num_constraints != 0:
        # sense = numpy.chararray(num_constraints)
        sense = [GRB.LESS_EQUAL for _ in range(num_constraints)]
        for i in equality_constraints:
            sense[i] = GRB.EQUAL

        # sense.fill(GRB.LESS_EQUAL)
        # sense[equality_constraints] = GRB.EQUAL
        # inequality = [i for i in range(num_constraints) if i not in equality_constraints]
        # sense[inequality] = GRB.LESS_EQUAL

        model.addMConstr(A, x, sense, b)

    objective = 0

    if Q is not None and c is None:
        objective = .5 * (x @ Q @ x)

    if c is not None and Q is None:
        objective = c.flatten() @ x

    if Q is not None and c is not None:
        objective = .5 * (x @ Q @ x) + c.flatten() @ x

    model.setObjective(objective, sense=GRB.MINIMIZE)

    model.optimize()
    model.update()

    # get gurobi status
    status = model.status
    # if not solved return None
    if status != GRB.OPTIMAL and status != GRB.SUBOPTIMAL:
        return None

    # create the Solver return object
    sol = SolverOutput(obj=model.getAttr("ObjVal"), sol=numpy.array(x.X), slack=None,
                       active_set=None, dual=None)

    # if we have a constrained system we need to add in the slack variables and active set
    if num_constraints != 0:

        if get_duals:
            # dual variables only really make sense if the system doesn't have binaries
            if len(bin_vars) == 0:
                sol.dual = numpy.array(model.getAttr("Pi"))

        sol.slack = numpy.array(model.getAttr("Slack"))
        sol.active_set = numpy.where((A @ sol.sol.flatten() - b.flatten()) ** 2 < 10 ** -12)[0]

    return sol

@dataclass
class SolverOutput:
    """
    This is the generic Solver information object. This will be the general return object from all the back end
    solvers. This was done to remove the need for the user to specialize IO for any particular Solver. It contains
    all the information you would need for the optimization solution including, optimal value, optimal solution,
    the active set, the value of the slack variables and the largange multipliers associated with every constraint (
    these are listed) as the dual variables.

    Members:
    obj: objective value of the optimal solution \n
    sol: x*, numpy array \n

    Optional Parameters -> None or numpy.ndarray type

    slack: the slacks associated with every constraint \n
    equality_indices: the active set of the solution, including strongly and weakly active constraints \n
    dual: the lagrange multipliers associated with the problem\n

    """
    obj: float
    sol: numpy.ndarray

    slack: Optional[numpy.ndarray]
    active_set: Optional[numpy.ndarray]
    dual: Optional[numpy.ndarray]

    def __eq__(self, other):
        if not isinstance(other, SolverOutput):
            return NotImplemented

        return numpy.allclose(self.slack, other.slack) and numpy.allclose(self.active_set,
                                                                          other.active_set) and numpy.allclose(
            self.dual, other.dual) and numpy.allclose(self.sol, other.sol) and numpy.allclose(self.obj, other.obj)


def get_program_parameters(Q: Optional[numpy.ndarray], c: Optional[numpy.ndarray], A: Optional[numpy.ndarray],
                           b: Optional[numpy.ndarray]):
    """ Given a set of possibly None optimization parameters determine the number of variables and constraints """
    num_c = 0
    num_v = 0

    if Q is not None:
        num_v = Q.shape[0]

    if A is not None:
        num_v = A.shape[1]
        num_c = A.shape[0]

    if c is not None:
        num_v = numpy.size(c)

    return num_v, num_c

def is_full_dimensional(A, b, solver: Solver = None):
    """
    This checks the dimensionality of a polytope defined by P = {x: Ax≤b}. Current method is based on checking if the
    radii of the chebychev ball is nonzero. However, this is numerically not so stable, and will eventually be replaced
    by looking at the ratio of the 2 chebychev balls

    :param A: LHS of polytope constraints
    :param b: RHS of polytope constraints
    :param solver: the solver interface to direct the deterministic solver
    :return: True if polytope is fully dimensional else False
    """

    if solver is None:
        solver = Solver()

    # TODO: Add second chebychev ball to get a more accurate estimate of lower dimensionality

    soln = chebyshev_ball(A, b, deterministic_solver=solver.solvers['lp'])

    if soln is not None:
        return soln.sol[-1] > 10 ** -8
    return False
