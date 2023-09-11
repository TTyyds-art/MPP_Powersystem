from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Optional, Iterable

import gurobipy as gp
import numpy
from gurobipy import GRB

from .chebyshev_ball import chebyshev_ball
from ..critical_region import CriticalRegion
from ..mpQCQP_program import MPQCQP_Program
from ..solver import Solver
from ..utils.constraint_utilities import cheap_remove_redundant_constraints, remove_duplicate_rows, \
    scale_constraint
from ..utils.general_utils import ppopt_block


# set_start_method("spawn")


def parallel_solve_ineq(args):
    sol = solve_lp_gurobi_torch(None, None, args[0], args[1], [args[2] + args[3]])
    if sol is not None:
        return args[2]
    else:
        return None


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


def gen_cr_from_active_set_torch_add(program: MPQCQP_Program, active_set: List[int], keep_ineq_indices: Optional[List[int]] = None, check_full_dim=True) -> Optional[
    CriticalRegion]:
    """
    Builds the critical region of the given mpqp from the active set.

    :param program: the MQMP_Program to be solved
    :param active_set: the active set combination to build this critical region from
    :param check_full_dim: Keyword Arg, if true will return null if the region has lower dimensionality
    :return: Returns the associated critical region if fully dimensional else returns None
    """

    num_equality = program.num_equality_constraints()

    if keep_ineq_indices is None:
        keep_ineq_indices = list(range(program.num_inequality_constraints()))
    # 这里的lagrange_A对应激活的不等式约束和等式约束
    inactive = [i for i in keep_ineq_indices if i not in active_set]
    parameter_A, parameter_b, lagrange_A, lagrange_b = program.optimal_control_law(active_set)

    #
    lambda_A, lambda_b = -lagrange_A[num_equality:], lagrange_b[num_equality:]  #

    # Theta(loads) Constraints
    omega_A, omega_b = program.A_t, program.b_t.squeeze()
    omega_b = numpy.expand_dims(omega_b, axis=1)  #该omega_b本来是不增加维度的，但报错。所以在上面加queeze(),在此加expand_dims()

    # Inactive Constraints remain inactive; 这里的 program.A/b 都是不等式约束，没有等式约束
    inactive_A = program.A[inactive] @ parameter_A - program.F[inactive]
    inactive_b = program.b[inactive] - program.A[inactive] @ parameter_b

    # we need to check for zero rows
    lamba_nonzeros = [i for i, t in enumerate(lambda_A) if numpy.nonzero(t)[0].shape[0] > 0]
    ineq_nonzeros = [i for i, t in enumerate(inactive_A) if numpy.nonzero(t)[0].shape[0] > 0]

    # Block of all critical region constraints

    lambda_Anz = lambda_A[lamba_nonzeros]
    lambda_bnz = lambda_b[lamba_nonzeros]

    inactive_Anz = inactive_A[ineq_nonzeros]
    inactive_bnz = inactive_b[ineq_nonzeros]

    CR_A = ppopt_block([[lambda_Anz], [inactive_Anz], [omega_A]])
    CR_b = ppopt_block([[numpy.expand_dims(lambda_bnz, axis=1)], [numpy.expand_dims(inactive_bnz, axis=1)],
                        [omega_b]]) # 使每个matrix都是2D的

    CR_As, CR_bs = scale_constraint(CR_A, CR_b)

    # if check_full_dim is set check if region is lower dimensional if so return None
    # if check_full_dim:
    #     # if the resulting system is not fully dimensional return None
    #     A_temp = numpy.ones_like(CR_As)*10
    #     b_temp = numpy.zeros_like(CR_bs) + numpy.ones_like(CR_bs)*0.2
    #     if not is_full_dimensional(CR_As, CR_bs, program.solver):
    #         return None

    # if it is fully dimensional we get to classify the constraints and then reduce them (important)!


    # with get_context("spawn").Pool(processes=4) as pool:
    #     args_list = [(CR_As, CR_bs, i, 0) for i in range(len(lamba_nonzeros))]
    #     results_1 = pool.map(parallel_solve_ineq, args_list)
    #
    #     args_list = [(CR_As, CR_bs, i, len(lamba_nonzeros)) for i in range(len(ineq_nonzeros))]
    #     results_2 = pool.map(parallel_solve_ineq, args_list)
    #
    #     args_list = [(CR_As, CR_bs, i, len(lamba_nonzeros) + len(ineq_nonzeros)) for i in
    #                  range(omega_A.shape[1])]
    #     results_3 = pool.map(parallel_solve_ineq, args_list)
    with ThreadPoolExecutor(max_workers=6) as executor:
        args_list = [(CR_As, CR_bs, i, 0) for i in range(len(lamba_nonzeros))]
        results_1 = list(executor.map(parallel_solve_ineq, args_list))

        args_list = [(CR_As, CR_bs, i, len(lamba_nonzeros)) for i in range(len(ineq_nonzeros))]
        results_2 = list(executor.map(parallel_solve_ineq, args_list))

        args_list = [(CR_As, CR_bs, i, len(lamba_nonzeros) + len(ineq_nonzeros)) for i in
                     range(omega_A.shape[1])]
        results_3 = list(executor.map(parallel_solve_ineq, args_list))

    # Filter out None values from results to get the kept inequality indices
    kept_lambda_indices = [index for index in results_1 if index is not None]
    kept_inequality_indices = [index for index in results_2 if index is not None]
    kept_omega_indices = [index for index in results_3 if index is not None]
    # create out reduced Critical region constraint block
    CR_As = ppopt_block(
        [[lambda_Anz[kept_lambda_indices]], [inactive_Anz[kept_inequality_indices]], [omega_A[kept_omega_indices]]])
    CR_bs = ppopt_block(
        [[numpy.expand_dims(lambda_bnz[kept_lambda_indices], axis=1)], [numpy.expand_dims(inactive_bnz[kept_inequality_indices], axis=1)], [omega_b[kept_omega_indices]]])

    # recover the lambda boundaries that remain
    relevant_lambda = [active_set[index] for index in kept_lambda_indices]

    real_regular = [inactive[index] for index in kept_inequality_indices]
    regular = real_regular

    # remove any possible duplicate constraints
    # and rescale since we did not rescale this particular set of constraints!!!
    CR_As, CR_bs = remove_duplicate_rows(CR_As, CR_bs)
    CR_As, CR_bs = scale_constraint(CR_As, CR_bs)

    return CriticalRegion(parameter_A, parameter_b, lagrange_A, lagrange_b, CR_As, CR_bs, active_set,
                          kept_omega_indices, relevant_lambda, regular)



def gen_cr_from_active_set(program: MPQCQP_Program, active_set: List[int], check_full_dim=True) -> Optional[
    CriticalRegion]:
    """
    Builds the critical region of the given mpqp from the active set.

    :param program: the MQMP_Program to be solved
    :param active_set: the active set combination to build this critical region from
    :param check_full_dim: Keyword Arg, if true will return null if the region has lower dimensionality
    :return: Returns the associated critical region if fully dimensional else returns None
    """

    num_equality = program.num_equality_constraints()

    # 这里的lagrange_A对应激活的不等式约束和等式约束
    inactive = [i for i in range(program.num_inequality_constraints()) if i not in active_set]
    parameter_A, parameter_b, lagrange_A, lagrange_b = program.optimal_control_law(active_set)

    #
    lambda_A, lambda_b = -lagrange_A[num_equality:], lagrange_b[num_equality:]  #

    # Theta(loads) Constraints
    omega_A, omega_b = program.A_t, program.b_t.squeeze()
    omega_b = numpy.expand_dims(omega_b, axis=1)  #该omega_b本来是不增加维度的，但报错。所以在上面加queeze(),在此加expand_dims()

    # Inactive Constraints remain inactive; 这里的 program.A/b 都是不等式约束，没有等式约束
    inactive_A = program.A[inactive] @ parameter_A - program.F[inactive]
    inactive_b = program.b[inactive] - program.A[inactive] @ parameter_b

    # we need to check for zero rows
    lamba_nonzeros = [i for i, t in enumerate(lambda_A) if numpy.nonzero(t)[0].shape[0] > 0]
    ineq_nonzeros = [i for i, t in enumerate(inactive_A) if numpy.nonzero(t)[0].shape[0] > 0]

    # Block of all critical region constraints

    lambda_Anz = lambda_A[lamba_nonzeros]
    lambda_bnz = lambda_b[lamba_nonzeros]

    inactive_Anz = inactive_A[ineq_nonzeros]
    inactive_bnz = inactive_b[ineq_nonzeros]

    CR_A = ppopt_block([[lambda_Anz], [inactive_Anz], [omega_A]])
    CR_b = ppopt_block([[numpy.expand_dims(lambda_bnz, axis=1)], [numpy.expand_dims(inactive_bnz, axis=1)],
                        [omega_b]]) # 使每个matrix都是2D的

    CR_As, CR_bs = scale_constraint(CR_A, CR_b)

    # if check_full_dim is set check if region is lower dimensional if so return None
    # if check_full_dim:
    #     # if the resulting system is not fully dimensional return None
    #     A_temp = numpy.ones_like(CR_As)*10
    #     b_temp = numpy.zeros_like(CR_bs) + numpy.ones_like(CR_bs)*0.2
    #     if not is_full_dimensional(CR_As, CR_bs, program.solver):
    #         return None

    # if it is fully dimensional we get to classify the constraints and then reduce them (important)!

    kept_lambda_indices = []
    kept_inequality_indices = []
    kept_omega_indices = []

    # iterate over the non-zero lagrange constraints
    for index in range(len(lamba_nonzeros)):

        sol = program.solver.solve_lp(None, CR_As, CR_bs, [index])

        if sol is not None:
            kept_lambda_indices.append(index)

    # iterate over the non-zero inequaltity constraints
    for index in range(len(ineq_nonzeros)):

        sol = program.solver.solve_lp(None, CR_As, CR_bs, [index + len(lamba_nonzeros)])

        if sol is not None:
            kept_inequality_indices.append(index)

    # iterate over the omega constraints
    for index in range(omega_A.shape[0]):

        sol = program.solver.solve_lp(None, CR_As, CR_bs, [index + len(lamba_nonzeros) + len(ineq_nonzeros)])

        if sol is not None:
            kept_omega_indices.append(index)

    # create out reduced Critical region constraint block
    CR_As = ppopt_block(
        [[lambda_Anz[kept_lambda_indices]], [inactive_Anz[kept_inequality_indices]], [omega_A[kept_omega_indices]]])
    CR_bs = ppopt_block(
        [[numpy.expand_dims(lambda_bnz[kept_lambda_indices], axis=1)], [numpy.expand_dims(inactive_bnz[kept_inequality_indices], axis=1)], [omega_b[kept_omega_indices]]])

    # recover the lambda boundaries that remain
    relevant_lambda = [active_set[index] for index in kept_lambda_indices]

    real_regular = [inactive[index] for index in kept_inequality_indices]
    regular = real_regular

    # remove any possible duplicate constraints
    # and rescale since we did not rescale this particular set of constraints!!!
    CR_As, CR_bs = remove_duplicate_rows(CR_As, CR_bs)
    CR_As, CR_bs = scale_constraint(CR_As, CR_bs)

    return CriticalRegion(parameter_A, parameter_b, lagrange_A, lagrange_b, CR_As, CR_bs, active_set,
                          kept_omega_indices, relevant_lambda, regular)


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
