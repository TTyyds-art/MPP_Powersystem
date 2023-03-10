from typing import List, Optional, Tuple

import numpy

infi = float('inf')
import os.path
import sys
import pandapower.networks as pn
import pandapower as pp
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
from numpy import flatnonzero as find, ones, zeros, Inf, pi, exp, conj, r_, arange, array
from pypower_.makeYbus import makeYbus
from pypower_.idx_brch import F_BUS, T_BUS, RATE_A, PF, QF, PT, QT, MU_SF, MU_ST
from pypower_.idx_gen import GEN_BUS, PG, QG
from pypower_.idx_cost import MODEL, PW_LINEAR, NCOST, POLYNOMIAL
from pypower_.totcost import totcost
from pypower_.makeSbus import makeSbus
from pypower_.idx_bus import BUS_TYPE, REF, VM, VA, MU_VMAX, MU_VMIN, LAM_P, LAM_Q
import cvxpy as cp
import numpy as np
from scipy.sparse import csr_matrix as sparse
from pypower_.idx_bus import PD, QD
from pypower_.idx_gen import GEN_BUS, PG, QG, GEN_STATUS

# from .mplp_program import MPLP_Program
# from .solver_interface.solver_interface import SolverOutput
# from .utils.general_utils import latex_matrix, remove_size_zero_matrices, ppopt_block, make_column
net_class = pp.create_empty_network()

def create_act_index(n_eq, nb, nl, n_Pg, active_set):
    '''
    create index for \lambda and \nu
    :param n_eq: the number of equality
    :param nb: the number of buses
    :param nl: the number of lines
    :param n_Pg: the number of generators
    :return: indexes for activated constraints
    '''


    act_Vmax = [True if i in active_set else False for i in range(n_eq, n_eq+nb-1)]

    act_Vmin = [True if i in active_set else False for i in range(n_eq+nb-1, n_eq+2*nb-2)]

    act_f = [True if i in active_set else False for i in range(n_eq+2*nb-2, n_eq +2*nb-2+nl)]
    act_t = [True if i in active_set else False for i in range(n_eq +2*nb-2+nl, n_eq +2*nb-2+2*nl)]

    act_Pgmax = [True if i in active_set else False for i in range(n_eq +2*nb-2+2*nl, n_eq+2*nb-2+2*nl+n_Pg)]
    act_Pgmin = [True if i in active_set else False for i in range(n_eq +2*nb-2+2*nl+n_Pg, n_eq  +2*nb-2+2*nl+2*n_Pg)]


    return act_Vmax, act_Vmin, act_f, act_t, act_Pgmax, act_Pgmin


def create_inact_index(n_eq, nb, nl, n_Pg, active_set):
    '''
    create index for \lambda and \nu
    :param n_eq: the number of equality
    :param nb: the number of buses
    :param nl: the number of lines
    :param n_Pg: the number of generators
    :return: indexes for non-activated constraints
    '''

    inact_Umax = [i for i in range(n_eq, n_eq + nb-1) if i not in active_set]
    inact_Umin = [i for i in range(n_eq + nb-1, n_eq + 2 * nb-2) if i not in active_set]

    inact_f = [i for i in range(n_eq + 2 * nb - 2, n_eq + 2 * nb + nl - 2) if i not in active_set]
    inact_t = [i for i in range(n_eq + 2 * nb + nl - 2, n_eq + 2 * nb + 2 * nl - 2) if i not in active_set]

    inact_Pgmax = [i for i in range(n_eq + 2 * nb + 2 * nl - 2, n_eq + 2 * nb + 2*nl + n_Pg - 2) if i not in active_set]
    inact_Pgmin = [i for i in range(n_eq + 2 * nb + 2*nl + n_Pg - 2, n_eq + 2 * nb + 2*nl + n_Pg * 2 - 2) if i not in active_set]

    return  inact_Umax, inact_Umin, inact_f, inact_t, inact_Pgmax, inact_Pgmin

    ##todo


def clear_om(om, net):
    '''
    This function transform the information from 'om' to OPF parameters.
    :param om: om is from net object
    :return: the params. that are used for GUROBI modelling
    '''

    path_current = '/home/ubuntu-h/PycharmProjects/scientificProject'
    if path_current not in sys.path:
        sys.path.insert(1, '/home/ubuntu-h/PycharmProjects/scientificProject')

    ## unpack data
    ppc = om.get_ppc()
    baseMVA, bus, gen, branch, gencost = \
        ppc["baseMVA"], ppc["bus"], ppc["gen"], ppc["branch"], ppc["gencost"]
    vv, _, _, _ = om.get_idx()

    ## problem dimensions
    nb = bus.shape[0]  ## number of buses
    nl = branch.shape[0]  ## number of branches
    ny = om.getN('var', 'y')  ## number of piece-wise linear costs, 'y' is a part of p-w linear costs

    ## bounds on optimization vars
    x0, xmin, xmax = om.getv()

    # build admittance matrices
    Ybus, Yf, Yt = makeYbus(baseMVA, bus, branch)

    ## Set the lower and upper bound for all variables
    ll, uu = xmin.copy(), xmax.copy()
    # print(f'll : {ll}; uu : {uu}')
    ll[xmin == -Inf] = -1e10  ## replace Inf with numerical proxies
    uu[xmax == Inf] = 1e10
    Va_refs = bus[bus[:, BUS_TYPE] == REF, VA] * (pi / 180)
    # print(f"Va_refs: {Va_refs}")
    ll[vv["i1"]["Va"]:vv["iN"]["Va"]] = -np.ones_like(bus[:, VA]) * (pi / 2)  # Va lower bound ??????
    uu[vv["i1"]["Va"]:vv["iN"]["Va"]] = np.ones_like(bus[:, VA]) * (pi / 2)  # Va upper bound ??????
    ## deal with the Va_refs
    ll[vv["i1"]["Va"]:vv["iN"]["Va"]][bus[:, BUS_TYPE] == REF] = Va_refs  # Va_refs lower bound ??????
    uu[vv["i1"]["Va"]:vv["iN"]["Va"]][bus[:, BUS_TYPE] == REF] = Va_refs  # Va_refs upper bound ??????
    ## deal with reactive power, whose ll and uu is 1e9
    ll[ll < -1e4] = -100
    uu[uu > 1e4] = 100
    v_max = uu[vv["i1"]["Vm"]:vv["iN"]["Vm"]][-1]

    # ??????
    x0[vv["i1"]["Vm"]:vv["iN"]["Vm"]] = bus[:, VM]

    Va_refs = bus[bus[:, BUS_TYPE] == REF, VA] * (pi / 180)
    x0[vv["i1"]["Va"]:vv["iN"]["Va"]] = bus[:, VA] * (pi / 180)  # ??????

    ## ??????
    x0[vv["i1"]["Pg"]:vv["iN"]["Pg"]] = gen[:, PG]
    x0[vv["i1"]["Qg"]:vv["iN"]["Qg"]] = gen[:, QG]

    ## grab Pg & Qg
    Pg = x0[vv["i1"]["Pg"]:vv["iN"]["Pg"]]  ## active generation in p.u.
    Qg = x0[vv["i1"]["Qg"]:vv["iN"]["Qg"]]  ## reactive generation in p.u.

    ##----- evaluate objective function -----
    ## polynomial cost of P and Q
    # use totcost only on polynomial cost in the minimization problem formulation, pwl cost is the sum of the y variables.
    ipol = find(gencost[:, MODEL] == POLYNOMIAL)  ## poly MW and MVAr costs
    xx = r_[Pg, Qg] * baseMVA
    if len(ipol) > 0:
        f = sum(totcost(gencost[ipol, :], xx[ipol]))  ## cost of poly P or Q
    else:
        f = 0
    First_Or_Con = 4

    ## put Pg & Qg back in gen
    gen[:, PG] = Pg * baseMVA  ## active generation in MW
    gen[:, QG] = Qg * baseMVA  ## reactive generation in MVAr
    on = find(gen[:, GEN_STATUS] > 0)  ## which generators are on?
    gbus = gen[on, GEN_BUS]  ## what buses are they at?

    ## form net complex bus power injection vector
    nb = bus.shape[0]
    ngon = on.shape[0]
    ## connection matrix, element i, j is 1 if gen on(j) at bus i is ON
    Cg = sparse((ones(ngon), (gbus, range(ngon))), (nb, ngon))

    ## power injected by gens plus power injected by loads converted to p.u.
    Sbus = (Cg * (gen[on, PG] + 1j * gen[on, QG]) - (bus[:, PD] + 1j * bus[:, QD])) / baseMVA
    # Sbus = makeSbus(baseMVA, bus, gen) ## net injected power in p.u.

    ## reconstruct V
    Va = x0[vv["i1"]["Va"]:vv["iN"]["Va"]]
    Vm = x0[vv["i1"]["Vm"]:vv["iN"]["Vm"]]
    V = Vm * exp(1j * Va)

    ## find branches with flow limits
    il = find((branch[:, RATE_A] != 0) & (branch[:, RATE_A] < 1e10))
    nl2 = len(il)  ## number of constrained lines

    ## ?????????????????? ??????????????????
    ppopt = {}
    il = find(branch[:, RATE_A] < 1e10)
    nl2 = len(il)  ## number of constrained lines
    if any(net.line['max_i_ka']) > 0:
        ppopt['OPF_FLOW_LIM'] = 2

    if nl2 > 0:
        flow_max = (branch[il, RATE_A] / baseMVA) ** 2
        flow_max[flow_max == 0] = 1e5
        print(f"flow_max: {flow_max}")
        if ppopt['OPF_FLOW_LIM'] == 2:  ## current magnitude limit, |I|
            If = Yf * V
            It = Yt * V
            h = r_[If * conj(If) - flow_max,  ## branch I limits (from bus)
                   It * conj(It) - flow_max].real  ## branch I limits (to bus)
        else:
            ## compute branch power flows
            ## complex power injected at "from" bus (p.u.)
            Sf = V[branch[il, F_BUS].astype(int)] * conj(Yf * V)
            ## complex power injected at "to" bus (p.u.)
            St = V[branch[il, T_BUS].astype(int)] * conj(Yt * V)
            if ppopt['OPF_FLOW_LIM'] == 1:  ## active power limit, P (Pan Wei)
                h = r_[Sf.real ** 2 - flow_max,  ## branch P limits (from bus)
                       St.real ** 2 - flow_max]  ## branch P limits (to bus)
            else:  ## apparent power limit, |S|
                h = r_[Sf * conj(Sf) - flow_max,  ## branch S limits (from bus)
                       St * conj(St) - flow_max].real  ## branch S limits (to bus)
    else:
        h = zeros((0, 1))

    llv = ll[vv["i1"]["Vm"]:vv["iN"]["Vm"]] ** 2
    uuv = uu[vv["i1"]["Vm"]:vv["iN"]["Vm"]] ** 2

    # Parameters
    # obtain the complex load
    Pd = bus[:, PD]
    Qd = bus[:, QD]
    slack_v = net.ext_grid['vm_pu']

    lb_update = -infi

    # obtain the range of voltage; the range is from the setting
    V_range = v_max - 1
    V_ll, V_uu = 1.0 - V_range, 1.0 + V_range

    # obtain the power range
    Pg_lower, Pg_upper = ll[vv["i1"]["Pg"]:vv["iN"]["Pg"]], uu[vv["i1"]["Pg"]:vv["iN"]["Pg"]] * baseMVA
    Qg_lower, Qg_upper = ll[vv["i1"]["Qg"]:vv["iN"]["Qg"]], uu[vv["i1"]["Qg"]:vv["iN"]["Qg"]] * baseMVA

    # the dimension of the OPF
    nb = bus.shape[0]  ## number of buses
    nl = branch.shape[0]  ## number of branches

    para_results = [Pd, Qd, slack_v, lb_update, V_ll, V_uu, Pg_lower, Pg_upper, Qg_lower, Qg_upper, nb, nl, gencost, Ybus, Yf, Yt, ngon, Cg, baseMVA, ipol, First_Or_Con, flow_max]

    return para_results


def opt_modelling(params, active_set = None):
    Pd_original, Qd, slack_v, lb_update, V_ll, V_uu, Pg_lower, Pg_upper, Qg_lower, Qg_upper, nb, nl, gencost, Ybus, Yf, Yt, ngon,\
        Cg, baseMVA,ipol, First_Or_Con, flow_max = params

    orig_equa = 6 * nb + 4 * nl + 2  # 6*nb:temp1, temp2, P_net,Q_net, G_real, G_imag; 4*nl: line_temp_1~4; 2: slack U W.




    # create the model; set the model type: NonConvex.
    model = gp.Model("PowerGeneration")
    model.setParam('NonConvex', 2)

    if active_set is None:
        Pd = Pd_original
        active_set = []
    else:
        Pd = model.addMVar(nb, vtype=GRB.CONTINUOUS, name='Changeable variable: theta.')
        model.addConstr(0.8*Pd_original <= Pd , name='Range of theta.')
        model.addConstr(Pd <= 1.2 * Pd_original, name='Range of theta.')


    # creat variables for the model; the numbers in the following codes should be replaced by parameters
    V_re = model.addMVar(nb, lb=lb_update, vtype=GRB.CONTINUOUS, name='V real')
    V_im = model.addMVar(nb, lb=lb_update, vtype=GRB.CONTINUOUS, name='V imag')
    Pg = model.addMVar(ngon, lb=Pg_lower, ub=Pg_upper, vtype=GRB.CONTINUOUS, name='Pg')
    Qg = model.addMVar(ngon, lb=Qg_lower, ub=Qg_upper, vtype=GRB.CONTINUOUS, name='Qg')
    temp1 = model.addMVar(nb, lb=lb_update, vtype=GRB.CONTINUOUS, name='Temp 1')
    temp2 = model.addMVar(nb, lb=lb_update, vtype=GRB.CONTINUOUS, name='Temp 2')
    Pg_net = model.addMVar(nb, lb=lb_update, vtype=GRB.CONTINUOUS, name='Pg net')  # the number of buses, too
    Qg_net = model.addMVar(nb, lb=lb_update, vtype=GRB.CONTINUOUS, name='Qg net')
    Line_temp_1 = model.addMVar(nl, lb=lb_update, vtype=GRB.CONTINUOUS,
                                name='Line temp 1')  # the shape should be line number
    Line_temp_2 = model.addMVar(nl, lb=lb_update, vtype=GRB.CONTINUOUS,
                                name='Line temp 2')  # the shape should be line number
    Line_temp_3 = model.addMVar(nl, lb=lb_update, vtype=GRB.CONTINUOUS,
                                name='Line temp 3')  # the shape should be line number
    Line_temp_4 = model.addMVar(nl, lb=lb_update, vtype=GRB.CONTINUOUS,
                                name='Line temp 4')  # the shape should be line number

    # the prameters of branches
    G = np.real(Ybus)
    B = np.imag(Ybus)
    # the parameters for currents in branches
    G_yf, B_yf = np.real(Yf), np.imag(Yf)
    G_yt, B_yt = np.real(Yt), np.imag(Yt)

    # Constraints; insert the formulas
    Temp_c1 = model.addConstr(G @ V_re - B @ V_im == temp1, 'Temp Constr 1')
    Temp_c2 = model.addConstr(G @ V_im + B @ V_re == temp2, 'Temp Constr 2')
    Pg_net_c = model.addConstr(Cg @ Pg - Pd == Pg_net, 'Pg net Constr')
    Qg_net_c = model.addConstr(Cg @ Qg - Qd == Qg_net, 'Qg net Constr')
    # print(f'Pd:{Pd}\n Qd:{Qd}')
    CVL_CPL = model.addConstrs(
        (V_re[i] * temp1[i] * baseMVA + V_im[i] * temp2[i] * baseMVA == Pg_net[i] for i in range(nb)),
        'Cvl P')  # break into different parts
    CVL_CPL_Q = model.addConstrs(
        (V_im[i] * temp1[i] * baseMVA - V_re[i] * temp2[i] * baseMVA == Qg_net[i] for i in range(nb)), 'Cvl Q')

    # the constraints of voltages on all buses; set the constraint for slack bus, then give the upper and lower constraints for other buses
    V_slack = model.addConstr(V_re[0] == slack_v, 'V slack real')
    V_slack_a = model.addConstr(V_im[0] == 0, 'V slack imag')
    # the upper&lower constraints for buses except for the slack bus


    # Yf * W * Yf; the number is equal to that of the lines/branches
    Line_temp_c_1 = model.addConstr(G_yf @ V_re - B_yf @ V_im == Line_temp_1, 'Branch Con 1')
    Line_temp_c_2 = model.addConstr(G_yf @ V_im + B_yf @ V_re == Line_temp_2, 'Branch Con 2')
    Line_temp_c_3 = model.addConstr(G_yt @ V_re - B_yt @ V_im == Line_temp_3, 'Branch Con 3')
    Line_temp_c_4 = model.addConstr(G_yt @ V_im + B_yt @ V_re == Line_temp_4, 'Branch Con 4')

    ## ?????????????????????????????????02.21
    ## ?????????????????????????????? (nb - 1) * 2 ??? ??????: "??????".
    NU_upper_ineq = model.addConstrs((V_re[i] ** 2 + V_im[i] ** 2 <= V_uu ** 2 for i in range(1, nb) if i+orig_equa-1 not in active_set),
                               'V upper bound')  # ??? range ?????????????????????????????????????????????????????????????????????"???"??????????????????"???"???
    NU_upper_eq = model.addConstrs(
        (V_re[i] ** 2 + V_im[i] ** 2 == V_uu ** 2 for i in range(1, nb) if i + orig_equa-1 in active_set),
        'V upper bound')  # ??? range ?????????????????????????????????????????????????????????????????????"???"??????????????????"???"???
    #
    NU_lower_ineq = model.addConstrs((V_re[i] ** 2 + V_im[i] ** 2 >= V_ll ** 2 for i in range(1, nb) if i+orig_equa+nb-2 not in active_set),
                                    'V lower bound')
    NU_lower_eq = model.addConstrs(
        (V_re[i] ** 2 + V_im[i] ** 2 == V_ll ** 2 for i in range(1, nb) if i + orig_equa+nb-2 in active_set),
        'V lower bound')


    ##  ??????line????????????????????? ????????? 2* nl ?????????
    Line_c_1_ineq = model.addConstrs((Line_temp_1[i] ** 2 + Line_temp_2[i] ** 2 <= np.real(flow_max[i]) for i in range(nl) if i+orig_equa+nb*2-2 not in active_set),
                                'Branch Con 1')  # p.u.
    Line_c_1_eq = model.addConstrs((Line_temp_1[i] ** 2 + Line_temp_2[i] ** 2 == np.real(flow_max[i]) for i in range(nl) if i+orig_equa+nb*2-2 in active_set),
                                'Branch Con 1')  # p.u.

    Line_c_2_ineq = model.addConstrs((Line_temp_3[i] ** 2 + Line_temp_4[i] ** 2 <= np.real(flow_max[i]) for i in range(nl) if i+orig_equa+nb*2+nl-2 not in active_set),
                                'Branch Con 2')  # p.u.
    Line_c_2_eq = model.addConstrs((Line_temp_3[i] ** 2 + Line_temp_4[i] ** 2 == np.real(flow_max[i]) for i in range(nl) if i+orig_equa+nb*2+nl-2 in active_set),
                                'Branch Con 2')  # p.u.

    # ??????Pg??????????????????????????? 2* len(Pg)
    # ?????????????????????????????????????????????????????????????????????
    Pg_c_ineq = model.addConstrs(
        Pg[i] <= Pg_upper[i] for i in range(len(Pg_upper)) if i + orig_equa + nb * 2 + nl * 2 - 2 not in active_set)
    Pg_c_eq = model.addConstrs(Pg[i] == Pg_upper[i] for i in range(len(Pg_upper)) if i+orig_equa+nb*2+nl*2-2 in active_set)
    Pg_lower_eq = model.addConstrs(
        Pg[i] == Pg_lower[i] for i in range(len(Pg_upper)) if i + orig_equa + nb * 2 + nl * 2-2+len(Pg_upper) in active_set)
    Pg_lower_ineq = model.addConstrs(
        Pg[i] >= Pg_lower[i] for i in range(len(Pg_upper)) if i + orig_equa + nb * 2 + nl * 2 - 2 + len(Pg_upper) not in active_set)


    # set the objective of the model
    # The "gencost" changes with different cases. Keep in Mind!
    # model.setObjective(gencost[ipol, First_Or_Con]@Pg**2+gencost[ipol, First_Or_Con+1]@Pg, GRB.MINIMIZE)
    if len(active_set) == orig_equa or len(active_set) == 0:
        model.setObjective(gencost[ipol, First_Or_Con] @ Pg, GRB.MINIMIZE)

        # calculate the optimization
        model.optimize()

        # deal with the results
        Pg_star, Qg_star = Pg.X, Qg.X
        optimized_obj = gencost[ipol, First_Or_Con] @ Pg_star
        return model, Pg_star, Qg_star, optimized_obj
    else:
        model.setObjective(0, GRB.MINIMIZE)

        model.optimize()
        # get gurobi status
        status = model.status
        # if not solved return None
        if status != GRB.OPTIMAL and status != GRB.SUBOPTIMAL:
            return None
        # if solved return
        else:
            return status == GRB.OPTIMAL or status == GRB.SUBOPTIMAL




class MPQCQP_Program():
    r"""
    ???????????????OPF?????????

    .. math::
        \min \frac{1}{2}x^TQx + \theta^TH^Tx + c^Tx
    .. math::
        \begin{align}
        x^THx +d^Tx-z\theta=0\\
        Ax &\leq b + F\theta\\
        A_{eq}x &= b_{eq}\\
        A_\theta \theta &\leq b_\theta\\
        x &\in R^n\\
        \end{align}
    """

    def __init__(self, net=None):
        """Initialized the Non-convex OPF problem from a net object."""

        if net is not None:

                self.net = net
                self.om, self.ppopt = pp.runopp(net, delta=1e-16)  # solve the OPF problem
        else:
            raise Exception("The input is not a net object built on the PandaPower.")
            # print()


        res_cost = self.net.res_cost

        # get parameters from net obj
        self.parameters_grid = clear_om(om=self.om, net=self.net)

        # parameters_grid: Pd, Qd, slack_v, lb_update, V_ll, V_uu, Pg_lower, Pg_upper, Qg_lower, Qg_upper, nb, nl, gencost, Ybus, Yf, Yt, ngon,\
        #         Cg, baseMVA,ipol, First_Or_Con
        nb, nl = self.parameters_grid[10], self.parameters_grid[11]
        self.nb, self.nl, self.n_Pg = nb, nl, len(self.parameters_grid[6])
        self.model_grid, self.Pg_star, self.Qg_star, optimized_obj = opt_modelling(params=self.parameters_grid)
        self.equality_indices = [*range(6*nb + 4*nl + 2)]   # ??????????????????????????????????????????????????????

        if abs(res_cost / optimized_obj) - 1 <= 0.001:
            print('*'*75)
            print("The result of GUROBI-model is exact! Very GOOD!")
        else:
            print(f'The solveing error is {abs(res_cost / optimized_obj)}')

    def num_x(self) -> int:
        """Returns number of parameters."""
        return len(self.model_grid.getVars())

    def num_t(self) -> int:
        """Returns number of uncertain variables."""
        return self.nb   # the number of Pd

    def num_constraints(self) -> int:
        """Returns number of constraints: linear constraints + quadratic constraints"""
        return len(self.model_grid.getConstrs())+len(self.model_grid.getQConstrs())

    def num_inequality_constraints(self) -> int:
        return 2*(self.nb-1) + 2*self.nl + 2*self.n_Pg

    def num_equality_constraints(self) -> int:
        return len(self.equality_indices)

    def evaluate_objective(self, x: numpy.ndarray, theta_point: numpy.ndarray):
        pass

    # def warnings(self) -> List[str]:
    #     """Checks the dimensions of the matrices to ensure consistency."""
    #     warning_list = list()
    #
    #     # check if b is a column vector
    #     if len(self.b.shape) != 2:
    #         warning_list.append(f'The b matrix is not a column vector b{self.b.shape}')
    #         self.b = make_column(self.b)
    #         warning_list.append('This has been corrected')
    #
    #     # check if c is a column matrix
    #     if len(self.c.shape) != 2:
    #         warning_list.append(f'The c vector is not a column vector c{self.c.shape}')
    #         self.c = make_column(self.c)
    #         warning_list.append('This has been corrected')
    #
    #     # check if c and A have consistent dimensions
    #     if self.A.shape[1] != self.c.shape[0]:
    #         warning_list.append(
    #             f'The A and b matrices disagree in number of parameters A{self.A.shape}, c{self.c.shape}')
    #
    #     # check is A and b agree with each other
    #     if self.A.shape[0] != self.b.shape[0]:
    #         warning_list.append(f'The A and b matrices disagree in vertical dimension A{self.A.shape}, b{self.b.shape}')
    #
    #     # check is A and b agree with each other
    #     if self.A_t.shape[0] != self.b_t.shape[0]:
    #         warning_list.append(
    #             f'The A and b matrices disagree in vertical dimension A{self.A_t.shape}, b{self.b_t.shape}')
    #
    #     # check dimensions of A and F matrix
    #     if self.A.shape[0] != self.F.shape[0]:
    #         warning_list.append(
    #             f"The A and F matrices disagree in vertical dimension A{self.A.shape}, F {self.F.shape}")
    #
    #     # return warnings
    #     return warning_list
    #
    # # Checks warnings again and prints warnings
    # def display_warnings(self) -> None:
    #     """Displaces warnings."""
    #     print(self.warnings())
    #
    # def display_latex(self) -> None:
    #     """Displaces Latex text of the multiparametric problem."""
    #     output = self.latex()
    #     for i in output:
    #         print(i)

    # def latex(self) -> List[str]:
    #     """
    #     Generates latex of the multiparametric problem
    #
    #     :return: returns latex of the
    #     """
    #     output = list()
    #
    #     # create string variables for x and theta
    #     x = ['x_{' + f'{i}' + '}' for i in range(self.num_x())]
    #     theta = ['\\theta_{' + f'{i}' + '}' for i in range(self.num_t())]
    #
    #     # create the latex matrices that represent x and theta
    #     # using the latex_matrix function from utils.general_utils
    #     x_latex = latex_matrix(x)
    #     theta_latex = latex_matrix(theta)
    #
    #     # builds the objective latex
    #     added_term = ''
    #     if not numpy.allclose(self.H, numpy.zeros_like(self.H)):
    #         added_term = " + " + theta_latex + '^{T}' + latex_matrix(self.H) + x_latex
    #
    #     obj = "$$" + "\\min_{x}" + latex_matrix(self.c) + "^T" + x_latex + added_term + "$$"
    #
    #     output.append(obj)
    #
    #     # adds the inequality constraint latex if applicable
    #     if self.num_constraints() - len(self.equality_indices) > 0:
    #         A_ineq = latex_matrix(select_not_in_list(self.A, self.equality_indices))
    #         b_ineq = latex_matrix(select_not_in_list(self.b, self.equality_indices))
    #         F_ineq = latex_matrix(select_not_in_list(self.F, self.equality_indices))
    #         output.append("$$" + ''.join([A_ineq, x_latex, '\\leq', b_ineq, '+', F_ineq, theta_latex]) + "$$")
    #
    #     # adds the equality constraint latex if applicable
    #     if len(self.equality_indices) > 0:
    #         A_eq = latex_matrix(self.A[self.equality_indices])
    #         b_eq = latex_matrix(self.b[self.equality_indices])
    #         F_eq = latex_matrix(self.F[self.equality_indices])
    #         output.append("$$" + ''.join([A_eq, x_latex, '=', b_eq, '+', F_eq, theta_latex]) + "$$")
    #
    #     # adds the theta constraint latex
    #     output.append("$$" + latex_matrix(self.A_t) + theta_latex + '\\leq' + latex_matrix(self.b_t) + "$$")
    #
    #     return output

    # def scale_constraints(self) -> None:
    #     """Rescales the constraints of the multiparametric problem to ||[A|-F]||_i = 1, in the L2 sense."""
    #     # scale the [A| b, F] constraint by the H = [A|-F] rows
    #     H = numpy.block([self.A, -self.F])
    #     norm = constraint_norm(H)
    #     self.A = self.A / norm
    #     self.b = self.b / norm
    #     self.F = self.F / norm
    #
    #     # scale the A_t constraint by the norm of it's rows
    #     norm = constraint_norm(self.A_t)
    #     self.A_t = self.A_t / norm
    #     self.b_t = self.b_t / norm

    # def process_constraints(self, find_implicit_equalities=True) -> None:
    #     """Removes redundant constraints from the multiparametric programming problem."""
    #     self.constraint_datatype_conversion()
    #     self.scale_constraints()
    #
    #     if find_implicit_equalities:
    #         problem_A = ppopt_block([[self.A, -self.F]])
    #         problem_b = ppopt_block([[self.b]])
    #
    #         constraint_pairs = detect_implicit_equalities(problem_A, problem_b)
    #
    #         keep = [i[0] for i in constraint_pairs]
    #         remove = [i[1] for i in constraint_pairs]
    #
    #         keep = list(set(keep))
    #         keep.sort()
    #
    #         remove = list(set(remove))
    #         remove.sort()
    #
    #         # make sure to only remove the unneeded inequalities -> only for duplicate constraints
    #         remove = [i for i in remove if i not in keep]
    #
    #         # our temporary new active set for the problem
    #         temp_active_set = [*self.equality_indices, *keep]
    #
    #         # what we are keeping
    #         survive = lambda x: x not in temp_active_set and x not in remove
    #         kept_ineqs = [i for i in range(self.num_constraints()) if survive(i)]
    #
    #         # data marshaling
    #         A_eq = self.A[temp_active_set]
    #         b_eq = self.b[temp_active_set]
    #         F_eq = self.F[temp_active_set]
    #
    #         A_ineq = self.A[kept_ineqs]
    #         b_ineq = self.b[kept_ineqs]
    #         F_ineq = self.F[kept_ineqs]
    #
    #         self.A = ppopt_block([[A_eq], [A_ineq]])
    #         self.b = ppopt_block([[b_eq], [b_ineq]])
    #         self.F = ppopt_block([[F_eq], [F_ineq]])
    #
    #         # update problem active set
    #         self.equality_indices = [i for i in range(len(temp_active_set))]
    #
    #     # recalculate bc we have moved everything around
    #     problem_A = ppopt_block([[self.A, -self.F], [numpy.zeros((self.A_t.shape[0], self.A.shape[1])), self.A_t]])
    #     problem_b = ppopt_block([[self.b], [self.b_t]])
    #
    #     saved_indices = find_redundant_constraints(problem_A, problem_b, self.equality_indices,
    #                                                solver=self.solver.solvers['lp'])
    #     # saved_indices = calculate_redundant_constraints(problem_A, problem_b)
    #
    #     saved_upper = [i for i in saved_indices if i < self.A.shape[0]]
    #     # saved_lower = [i - self.A.shape[0] for i in saved_indices if i >= self.A.shape[0]]
    #
    #     self.A = self.A[saved_upper]
    #     self.F = self.F[saved_upper]
    #     self.b = self.b[saved_upper]
    #
    #     # recalculate bc we have moved everything around
    #     problem_A = ppopt_block([[self.A, -self.F], [numpy.zeros((self.A_t.shape[0], self.A.shape[1])), self.A_t]])
    #     problem_b = ppopt_block([[self.b], [self.b_t]])
    #
    #     # saved_indices = calc_weakly_redundant(problem_A, problem_b, self.equality_indices)
    #     # saved_indices = calculate_redundant_constraints(problem_A, problem_b)
    #
    #     saved_upper = [i for i in saved_indices if i < self.A.shape[0]]
    #     # saved_lower = [i - self.A.shape[0] for i in saved_indices if i >= self.A.shape[0]]
    #
    #     self.A = self.A[saved_upper]
    #     self.F = self.F[saved_upper]
    #     self.b = self.b[saved_upper]
    #
    #     # print(f'Removed {self.A.shape[0] - len(saved_upper)} Weakly Redundant Constraints')
    #
    #     self.scale_constraints()

    # def constraint_datatype_conversion(self) -> None:
    #     """
    #     Makes sure that all the data types of the problem are in fp64, this is important as some solvers do not accept integral data types
    #     """
    #     self.A = self.A.astype('float64')
    #     self.c = self.c.astype('float64')
    #     self.b = self.b.astype('float64')
    #     self.F = self.F.astype('float64')
    #     self.A_t = self.A_t.astype('float64')
    #     self.b_t = self.b_t.astype('float64')
    #     self.H = self.H.astype('float64')
    #     self.c_c = self.c_c.astype('float64')
    #     self.c_t = self.c_t.astype('float64')
    #     self.Q_t = self.Q_t.astype('float64')

    # def solve_theta(self, theta_point: numpy.ndarray, deterministic_solver='gurobi') -> Optional[SolverOutput]:
    #     r"""
    #     Substitutes theta into the multiparametric problem and solves the following optimization problem
    #
    #     .. math::
    #
    #         \min_{x} \tilde{c}^Tx
    #
    #     .. math::
    #         \begin{align}
    #         Ax &\leq \tilde{b}\\
    #         A_{eq}x &= \tilde{b}_{eq}\\
    #         x &\in R^n\\
    #         \end{align}
    #
    #     :param theta_point: An uncertainty realization
    #     :param deterministic_solver: Deterministic solver to use to solve the above quadratic program
    #     :return: The Solver output of the substituted problem, returns None if not solvable
    #     """
    #
    #     if not numpy.all(self.A_t @ theta_point <= self.b_t):
    #         return None
    #
    #     sol_obj = self.solver.solve_lp(c=self.H @ theta_point + self.c, A=self.A, b=self.b + self.F @ theta_point,
    #                                    equality_constraints=self.equality_indices)
    #
    #     if sol_obj is not None:
    #         sol_obj.obj += self.c_c + self.c_t.T @ theta_point + 0.5 * theta_point.T @ self.Q_t @ theta_point  # ?????????????????????????????????
    #         return sol_obj
    #
    #     return None

    # def solve_theta_variable(self) -> Optional[SolverOutput]:
    #     """
    #     Leaves Theta as an optimization variable, solves the following problem
    #
    #     define y' = [x^T theta^T]^T
    #
    #     min [c^T 0]^Ty'
    #     s.t. [A -F]y' <= b
    #
    #     :return: the Solver output of the substituted problem, returns None if not solvable
    #     """
    #
    #     A_prime = numpy.block([self.A, -self.F])
    #     c_prime = numpy.block([[self.c], [numpy.zeros((self.num_t(), 1))]])
    #
    #     return self.solver.solve_lp(c=c_prime, A=A_prime, b=self.b, equality_constraints=self.equality_indices)

    def optimal_control_law(self, active_set: List[int]) -> Tuple:
        r"""
        This function calculates the optimal control law corresponding to an active set combination

        :param active_set: an active set combination
        :return: a tuple of the optimal x* and ??* functions in the following form(A_x, b_x, A_l, b_l)

        .. math::

            \begin{align*}
            x^*(\theta) &= A_x\theta + b_x\\
            \lambda^*(\theta) &= A_l\theta + b_l\\
            \end{align*}
        """

        pass

    # noinspection SpellCheckingInspection


    def check_feasibility(self, active_set) -> bool:
        r"""
        Checks the feasibility of an active set combination w.r.t. a multiparametric program.

        .. math::

            \min_{x,\theta} 0

        .. math::
            \begin{align}
            Ax &\leq b + F\theta\\
            A_{i}x &= b_{i} + F_{i}\theta, \quad \forall i \in \mathcal{A}\\
            A_\theta \theta &\leq b_\theta\\
            x &\in R^n\\
            \theta &\in R^m
            \end{align}

        :param active_set: an active set
        :param check_rank: Checks the rank of the LHS matrix for a violation of LINQ if True (default)
        :return: True if active set feasible else False
        """

        # if check_rank:
        #     if not is_full_rank(self.A, active_set):
        #         return False
        # if len(active_set)

        check_result = opt_modelling(params=self.parameters_grid, active_set=active_set)
        if isinstance(check_result, Tuple):
            raise Exception("active set is not a requried !", active_set)

        return check_result is not None

    def check_optimality(self, active_set, theta = None):
        r"""
        Tests if the active set is optimal for the provided mpLP program

        .. math::

            \max_{x, \theta, \lambda, s, t} \quad t

        .. math::
            \begin{align*}
                H \theta + (A_{A_i})^T \lambda_{A_i} + c &= 0\\
                A_{A_i}x - b_ai-F_{a_i}\theta &= 0\\
                A_{A_j}x - b_{A_j}-F_{A_j}\theta + s{j_k} &= 0\\
               t*e_1 &\leq \lambda_{A_i}\\
               t*e_2 &\leq s_{J_i}\\
               t &\geq 0\\
               \lambda_{A_i} &\geq 0\\
               s_{J_i} &\geq 0\\
               A_t\theta &\leq b_t
            \end{align*}

        :param theta: check theta is in the CR or not
        :param active_set: active set being considered in the optimality test
        :return: dictionary of parameters, or None if active set is not optimal
        """
        if len(active_set) != self.num_x(): # ?????????????????????????????????????????????num_x??? ??????????????????:?????????A?????????????????????max_depth?????????
            # return False
            pass


        zeros = lambda x, y: numpy.zeros((x, y))
        num_x = self.num_x()

        num_constraints = self.num_constraints()
        num_active = len(active_set)
        # num_theta_c = self.A_t.shape[0]
        n_eq = len(self.equality_indices)
        num_activated = len(active_set) - len(self.equality_indices)

        inactive = [i for i in range(num_constraints) if i not in active_set]

        num_inactive = num_constraints - num_active
        num_inequa = num_constraints - n_eq

        num_theta = self.num_t()

        # The "gencost" changes with different cases. Keep in Mind!
        # model.setObjective(gencost[ipol, First_Or_Con]@Pg**2+gencost[ipol, First_Or_Con+1]@Pg, GRB.MINIMIZE)
        # model.setObjective(gencost[ipol, First_Or_Con] @ Pg, GRB.MINIMIZE); while 5 bus
        # [Pd, Qd, slack_v, lb_update, V_ll,            ; 5
        # V_uu, Pg_lower, Pg_upper, Qg_lower, Qg_upper, ; 5
        # nb, nl, gencost, Ybus, Yf,                    ; 5
        # Yt, ngon, Cg, baseMVA, ipol,                  ; 5
        # First_Or_Con, flow_max] = param.              ; 5
        Pd_original, Qd, slack_v, lb_update, V_ll,V_uu, Pg_lower, Pg_upper, Qg_lower, Qg_upper, nb, nl, gencost, Ybus, Yf, \
        Yt, ngon, Cg, baseMVA, ipol, First_Or_Con, flow_max = self.parameters_grid
        # gencost, ipol, First_Or_Con = self.parameters_grid[12], self.parameters_grid[19], self.parameters_grid[20]
        Fx =[]
        Fx.append(zeros(nb*2, 1))   # add zeros for U, W, where V = U+jW.
        F_Pg = gencost[ipol, First_Or_Con].reshape([-1,1])
        # F_Pg = np.ones((3,1))
        # Fx.append(F_Pg)  # add attributes for d gencost / d Pg
        # Fx.append(zeros(len(Qg_lower), 1))  # add attributes for d gencost / d Qg
        #
        # # create the model; set the model type: NonConvex.
        model = gp.Model("PowerGeneration")
        model.setParam('NonConvex', 2)
        #
        # creat variables for the model; the numbers in the following codes should be replaced by parameters
        if theta is None:
            Pd = model.addMVar(nb, vtype=GRB.CONTINUOUS, name='Changeable variable: theta.')   # lb =0 ??????
        elif len(theta) >= 0:
            Pd = model.addMVar(nb, lb=theta, ub=theta, vtype=GRB.CONTINUOUS, name='fixed theta for Pd')
            print('-'*75)
            print(f"Checking {theta}, the point is in the CR or not.")
        else:
            raise Exception("Theta is not a required !", theta)

        V_re = model.addMVar(nb, lb=lb_update, vtype=GRB.CONTINUOUS, name='V real')
        V_im = model.addMVar(nb, lb=lb_update, vtype=GRB.CONTINUOUS, name='V imag')

        Pg = model.addMVar(ngon, lb=Pg_lower, ub=Pg_upper, vtype=GRB.CONTINUOUS, name='Pg')
        Qg = model.addMVar(ngon, lb=Qg_lower, ub=Qg_upper, vtype=GRB.CONTINUOUS, name='Qg')
        temp1 = model.addMVar(nb, lb=lb_update, vtype=GRB.CONTINUOUS, name='Temp 1')
        temp2 = model.addMVar(nb, lb=lb_update, vtype=GRB.CONTINUOUS, name='Temp 2')
        Pg_net = model.addMVar(nb, lb=lb_update, vtype=GRB.CONTINUOUS, name='Pg net')  # the number of buses, too
        Qg_net = model.addMVar(nb, lb=lb_update, vtype=GRB.CONTINUOUS, name='Qg net')
        Line_temp_1 = model.addMVar(nl, lb=lb_update, vtype=GRB.CONTINUOUS,
                                    name='Line temp 1')  # the shape should be line number
        Line_temp_2 = model.addMVar(nl, lb=lb_update, vtype=GRB.CONTINUOUS,
                                    name='Line temp 2')  # the shape should be line number
        Line_temp_3 = model.addMVar(nl, lb=lb_update, vtype=GRB.CONTINUOUS,
                                    name='Line temp 3')  # the shape should be line number
        Line_temp_4 = model.addMVar(nl, lb=lb_update, vtype=GRB.CONTINUOUS,
                                    name='Line temp 4')  # the shape should be line number
        muu_ = model.addMVar(2*nb, lb=-infi, vtype=GRB.CONTINUOUS, name='mu')
        # lambda_ = model.addMVar(2*(nb-1)+2*nl+ngon*2, lb=0, vtype=GRB.CONTINUOUS, name='lambda')
        t = model.addMVar((1, 1), lb=1e-3, ub=10, vtype=GRB.CONTINUOUS,name='t')      # ub = 10 ?????????t????????????,lb=1e-3 t?????????0???????????????????????????
        s_j = model.addVars(num_inequa, lb=0, vtype=GRB.CONTINUOUS, name='S_j')
        R_G_Us = model.addMVar((nb, nb), lb=-infi, vtype=GRB.CONTINUOUS, name='R{G^s_U}')
        R_G_Ws = model.addMVar((nb, nb), lb=-infi, vtype=GRB.CONTINUOUS, name='R{G^s_W}')
        H_f_U = model.addMVar((nl, nb), lb=-infi, vtype=GRB.CONTINUOUS, name='H^f_U')
        H_f_W = model.addMVar((nl, nb), lb=-infi, vtype=GRB.CONTINUOUS, name='H^f_W')
        H_t_U = model.addMVar((nl, nb), lb=-infi, vtype=GRB.CONTINUOUS, name='H^t_U')
        H_t_W = model.addMVar((nl, nb), lb=-infi, vtype=GRB.CONTINUOUS, name='H^t_W')

        # the prameters of branches
        G = np.real(Ybus)
        B = np.imag(Ybus)
        # the parameters for currents in branches
        G_yf, B_yf = np.real(Yf), np.imag(Yf)
        G_yt, B_yt = np.real(Yt), np.imag(Yt)

        Real_G_s_U = model.addConstr(G@V_re*np.eye(nb) - B@V_im*np.eye(nb) + V_re*np.eye(nb)@G + V_im*np.eye(nb)@B == R_G_Us, 'Temp Constrs:R_G_Us ')
        Real_G_s_W = model.addConstr(B@V_re*np.eye(nb) + G @ V_im * np.eye(nb) + V_re*np.eye(nb)@B - V_im*np.eye(nb)@G == R_G_Ws, 'Temp Constrs: R_G_Ws')

        temp_A = model.addMVar((nl, nl), lb=-infi, vtype=GRB.CONTINUOUS, name='[Cal{ A }]')
        temp_B = model.addMVar((nl, nl), lb=-infi, vtype=GRB.CONTINUOUS, name='[Cal{ B }]')

        ## ????????????active set??????
        # ??????????????? A ??? B
        Temp_A_c = model.addConstr(G_yf@V_re*np.eye(nl) - B_yf@V_im*np.eye(nl) ==temp_A, 'Temp Constr: Cal A')
        Temp_B_c = model.addConstr(B_yf @ V_re*np.eye(nl) + G_yf @ V_im*np.eye(nl) == temp_B, 'Temp Constr: Cal B')

        # ????????.T*dG/dU,??.T*dG/dW, ??.T*dG/dPg, ??.T*dG/dQg
        mu_dG_dU = model.addMVar((1, nb), lb=-infi, vtype=GRB.CONTINUOUS, name='??.T x ???????????????U?????????')
        mu_dG_dW = model.addMVar((1, nb), lb=-infi, vtype=GRB.CONTINUOUS, name='??.T x ???????????????W?????????')
        mu_dG_dPg = model.addMVar((1, ngon), lb=-infi, vtype=GRB.CONTINUOUS, name='??.T x ???????????????Pg?????????')
        mu_dG_dQg = model.addMVar((1, len(Pg_upper)), lb=-infi, vtype=GRB.CONTINUOUS, name='??.T x ???????????????Qg?????????')

        # ??????mu_ ???vector?????????mu_[:nb]??????????????????????????????????????????????????????
        model.addConstr(muu_.T[:nb]@R_G_Us + muu_.T[nb:]@-R_G_Ws==mu_dG_dU, name='??.T x dG/dU')  # I{G^s_U} = -R{G^s_W}
        model.addConstr(muu_.T[:nb]@R_G_Ws + muu_.T[nb:]@R_G_Us==mu_dG_dW, name='??.T x dG/dW')   # I{G^s_W} = R{G^s_U}
        model.addConstr(muu_.T[:nb]@-Cg + muu_.T[nb:]@zeros(nb,len(Pg_upper))==mu_dG_dPg, name='??.T x dG/dPg')   # ????????????????????????
        model.addConstr(muu_.T[:nb]@zeros(nb,len(Pg_upper)) + muu_.T[nb:]@-Cg==mu_dG_dQg, name='??.T x dG/dQg')

        # ????????????????????????????????????????????????????????????????????????????????????????????????????????? H[i]????????????????????????????????????for????????????
        H_f_U_Constr = model.addConstr(2 * temp_A@G_yf + 2 * temp_B@B_yf==H_f_U, "???????????????H^2_U" )
        H_f_W_Constr = model.addConstr(-2 * temp_A @ B_yf + 2 * temp_B @G_yf == H_f_W, "????????????????????????????????????")
        H_t_U_Constr = model.addConstr(2 * temp_A @ G_yt + 2 * temp_B @ B_yt == H_t_U, "???????????????H^2_U")
        H_t_W_Constr = model.addConstr(-2 * temp_A @ B_yt + 2 * temp_B @ G_yt == H_t_W, "????????????????????????????????????")

        # H_Umax_U = np.eye(nb)
        # H_Umax_W = np.zeros(shape=[nb,nb])
        # H_Umin_U = -np.eye(nb)
        # H_Umin_W = -H_Umax_W
        # shape_tran = np.array([[0,1,0,0], [0,0,1,0], [0,0,0,1]])  ##TODO change with the topology
        shape_tran = np.array([[0, 1, 0 ], [0, 0, 1]])
        H_vmax_U = 2*V_re[1:]*np.eye(nb-1)@shape_tran
        H_vmax_W = 2*V_im[1:]*np.eye(nb-1)@shape_tran
        H_vmin_U = -H_vmax_U
        H_vmin_W = -H_vmax_W

        # ???????????????????????????????????????Active set ?????????????????????????????????????????????????????????

        # ????????.T*dH/dU,??.T*dH/dW, ??.T*dH/dPg, ??.T*dH/dQg ; ???????????????????????????active set

        lambda_dH_dU = model.addMVar((1, nb), lb=-infi, vtype=GRB.CONTINUOUS, name='??.T x ????????????????????????U?????????')
        lambda_dH_dW = model.addMVar((1, nb), lb=-infi, vtype=GRB.CONTINUOUS, name='??.T x ????????????????????????W?????????')
        lambda_dH_dPg = model.addMVar((1, len(Pg_upper)), lb=-infi, vtype=GRB.CONTINUOUS, name='??.T x ????????????????????????Pg?????????')
        lambda_dH_dQg = model.addMVar((1, len(Pg_upper)), lb=-infi, vtype=GRB.CONTINUOUS, name='??.T x ????????????????????????Qg?????????')

        #  dH/dPg ????????? 0 , ??????I ???-I ??? dH/dQg ?????? 0 ###todo
        n_Pg = len(Pg_upper)

        # ??????????????????2(nb-1)??? V**2 ?????? ??? 2 nl ????????????????????? n_Pg ????????????????????????n_Pg ?????????????????????
        act_Vmax, act_Vmin, act_f, act_t, act_Pgmax, act_Pgmin = create_act_index(n_eq, nb, nl, n_Pg, active_set)

        lambda_vmax = [infi if num is True else 0 for num in act_Vmax]
        lambda_Vmax = model.addMVar((nb - 1), lb=0, ub=lambda_vmax, vtype=GRB.CONTINUOUS, name='lambda Vmax')
        lambda_vmin = [infi if num is True else 0 for num in act_Vmin]
        lambda_Vmin = model.addMVar(nb - 1, lb=0, ub=lambda_vmin, vtype=GRB.CONTINUOUS,
                                    name='lambda Vmin')
        lambda_f = [infi if num is True  else 0 for num in act_f]
        lambda_F = model.addMVar(nl, lb=0, ub=lambda_f, vtype=GRB.CONTINUOUS,
                                    name='lambda F')
        lambda_t = [infi if num is True else 0 for num in act_t]
        lambda_T = model.addMVar(nl, lb=0, ub=lambda_t, vtype=GRB.CONTINUOUS,
                                    name='lambda T')
        lambda_Pgmax = [infi if num is True else 0 for num in act_Pgmax]
        lambda_PGmax = model.addMVar(n_Pg, lb=0, ub=lambda_Pgmax, vtype=GRB.CONTINUOUS,
                                    name='lambda PGmax')
        lambda_Pgmin = [infi if num is True else 0 for num in act_Pgmin]
        lambda_PGmin = model.addMVar(n_Pg, lb=0, ub=lambda_Pgmin, vtype=GRB.CONTINUOUS,
                                    name='lambda PGmin')



        model.addConstr(lambda_F @ H_f_U + lambda_T @ H_t_U
                        + lambda_Vmax @ H_vmax_U + lambda_Vmin @ H_vmin_U == lambda_dH_dU,
                        name='??.T x dH/dU')  #
        model.addConstr(lambda_F @ H_f_W + lambda_T @ H_t_W
                        + lambda_Vmax @ H_vmax_W + lambda_Vmin @ H_vmin_W == lambda_dH_dW,
                        name='??.T x dH/dW')  #
        model.addConstr(lambda_PGmax @ np.eye(n_Pg) - lambda_PGmin @ np.eye(n_Pg) == lambda_dH_dPg,
                        name='??.T x dH/dPg')
        # model.addConstr(mu_.T[:nb] @ zeros(nb, len(Pg_upper)) + mu_.T[nb:] @ -Cg == lamb_dG_dQg,
        #                 name='??.T x dH/dQg')

        # dF/dX + (??.T x dG/dX).T + (??.T x dH/dX).T = 0
        dL_dU = model.addConstr(zeros(nb,1) + mu_dG_dU.T + lambda_dH_dU.T == 0, 'd Largane/dU = 0')
        dL_dW = model.addConstr(zeros(nb,1) + mu_dG_dW.T + lambda_dH_dW.T == 0, 'd Largane/dW = 0' )
        dL_dPg = model.addConstr(F_Pg + mu_dG_dPg.T + lambda_dH_dPg.T == 0, 'd Largane/dPg = 0' ) # mu_dH_dPg = 0
        # dL_dQg = model.addConstr(zeros(len(Qg_upper),1)+ mu_dG_dQg.T + zeros(len(Qg_upper), 1) == 0, 'd Largane/dQg = 0' ) # mu_dH_dQg = 0

        ## ??????????????????: ????????????

        # Constraints; insert the formulas
        Temp_c1 = model.addConstr(G @ V_re - B @ V_im == temp1, 'Temp Constr 1')
        Temp_c2 = model.addConstr(G @ V_im + B @ V_re == temp2, 'Temp Constr 2')
        Pg_net_c = model.addConstr(Cg @ Pg - Pd == Pg_net, 'Pg net Constr')
        Qg_net_c = model.addConstr(Cg @ Qg - Qd == Qg_net, 'Qg net Constr')
        # print(f'Pd:{Pd}\n Qd:{Qd}')
        CVL_CPL = model.addConstrs(
            (V_re[i] * temp1[i] * baseMVA + V_im[i] * temp2[i] * baseMVA == Pg_net[i] for i in range(nb)),
            'Cvl P')  # break into different parts
        CVL_CPL_Q = model.addConstrs(
            (V_im[i] * temp1[i] * baseMVA - V_re[i] * temp2[i] * baseMVA == Qg_net[i] for i in range(nb)), 'Cvl Q')
        #
        # # the constraints of voltages on all buses; set the constraint for slack bus, then give the upper and lower constraints for other buses
        V_slack = model.addConstr(V_re[0] == slack_v, 'V slack real')
        V_slack_a = model.addConstr(V_im[0] == 0, 'V slack imag')
        # the upper&lower constraints for buses except for the slack bus

        # Yf * W * Yf; the number is equal to that of the lines/branches
        Line_temp_c_1 = model.addConstr(G_yf @ V_re - B_yf @ V_im == Line_temp_1, 'Branch Con 1')
        Line_temp_c_2 = model.addConstr(G_yf @ V_im + B_yf @ V_re == Line_temp_2, 'Branch Con 2')
        Line_temp_c_3 = model.addConstr(G_yt @ V_re - B_yt @ V_im == Line_temp_3, 'Branch Con 3')
        Line_temp_c_4 = model.addConstr(G_yt @ V_im + B_yt @ V_re == Line_temp_4, 'Branch Con 4')

        orig_equa = 6 * nb + 4 * nl + 2  #
        # ??????????????????: ???????????????;  sj ???active set ????????????, sj yicix;
        ##to
        inact_Vmax, inact_Vmin, inact_f, inact_t, inact_Pgmax, inact_Pgmin = \
            create_inact_index(n_eq, nb, nl, n_Pg, active_set)
        V_upper_ineq = model.addConstrs(
            (V_re[i] ** 2 + V_im[i] ** 2 + s_j[i-1] == V_uu ** 2 for i in range(1, nb) if i+n_eq-1 in inact_Vmax),
            'V upper bound')  # ??????????????????????????? ???????????? sj.; ??????????????? nb-1 ???????????????????????? 0 ??????????????? i-1 ?????????????????????

        #
        V_lower_ineq = model.addConstrs(
            (V_ll ** 2 - V_re[i] ** 2 - V_im[i] ** 2 + s_j[i+len(inact_Vmax)-1] == 0 for i in range(1, nb) if i+n_eq+nb-2 in inact_Vmin),
            'V lower bound')

        # ??????????????????????????????baseMVA
        Line_c_1_ineq = model.addConstrs(
            (Line_temp_1[i] ** 2 + Line_temp_2[i] ** 2 + s_j[i+len(inact_Vmax)+len(inact_Vmin)] == np.real(flow_max[i]) for i in range(nl) if i+n_eq+2*nb-2 in inact_f[0:4]),
            'Branch Con 1')  # p.u.


        Line_c_2_ineq = model.addConstrs(
            (Line_temp_3[i] ** 2 + Line_temp_4[i] ** 2 + s_j[i+len(inact_Vmax)+len(inact_Vmin)+len(inact_f)] == np.real(flow_max[i]) for i in range(nl) if i+n_eq+2*nb+nl-2 in inact_t),
            'Branch Con 2')  # p.u.

        # ?????????????????????????????????????????????????????????????????????
        Pg_c_ineq = model.addConstrs(
            Pg[i] + s_j[i+len(inact_Vmax)+len(inact_Vmin)+len(inact_f)+len(inact_t)] == Pg_upper[i] for i in range(n_Pg) if i+n_eq+2*nb+2*nl-2 in inact_Pgmax)
        Pg_lower_ineq = model.addConstrs(
            Pg_lower[i] - Pg[i] + s_j[i+len(inact_Vmax)+len(inact_Vmin)+len(inact_f)+len(inact_t)+len(inact_Pgmax)] == 0 for i in range(n_Pg) if i+n_eq+2*nb+2*nl-2+n_Pg in inact_Pgmin)

        # TODO ?????????????????????????????????
        V_upper_eq = model.addConstrs(
            (V_re[i] ** 2 + V_im[i] ** 2  == V_uu ** 2 for i in range(1, nb) if i + n_eq-1 in active_set),
            'V upper bound')  # ??????????????????????????? ???????????? sj.

        #
        V_lower_eq = model.addConstrs(
            (V_ll ** 2 - V_re[i] ** 2 - V_im[i] ** 2 == 0 for i in range(1, nb) if
             i + n_eq + nb - 2 in active_set),
            'V lower bound')

        # ??????????????????????????????baseMVA
        Line_c_1_eq = model.addConstrs(
            (Line_temp_1[i] ** 2 + Line_temp_2[i] ** 2 == np.real(
                flow_max[i]) for i in range(nl) if i + n_eq + 2 * nb - 2 in active_set),
            'Branch Con 1')  # p.u.

        Line_c_2_eq = model.addConstrs(
            (Line_temp_3[i] ** 2 + Line_temp_4[i] ** 2 == np.real(flow_max[i]) for i in range(nl) if
             i + n_eq + 2 * nb + nl - 2 in active_set),
            'Branch Con 2')  # p.u.

        # ?????????????????????????????????????????????????????????????????????
        Pg_c_eq = model.addConstrs(
            Pg[i] == Pg_upper[i] for i in
            range(n_Pg) if i + n_eq + 2 * nb + 2 * nl - 2 in active_set)
        Pg_lower_eq = model.addConstrs(
            Pg_lower[i] - Pg[i] == 0 for i in
            range(n_Pg) if i + n_eq + 2 * nb + 2 * nl - 2 + n_Pg in active_set)

        # ???????????????????????????????????

        model.addConstrs(t <= lambda_Vmax[act_Vmax] for i in lambda_vmax if i >= 1)   # ?????????????????????
        model.addConstrs(t <= lambda_Vmin[act_Vmin] for i in lambda_vmin if i >= 1)  # ?????????????????????
        model.addConstrs(t <= lambda_F[act_f] for i in lambda_f if i >= 1)  # ?????????????????????
        model.addConstrs(t <= lambda_T[act_t] for i in lambda_t if i >= 1)  # ?????????????????????
        model.addConstrs(t <= lambda_PGmax[act_Pgmax] for i in lambda_Pgmax if i >= 1)  # ?????????????????????
        model.addConstrs(t <= lambda_PGmin[act_Pgmin] for i in lambda_Pgmin if i >= 1)  # ?????????????????????

        model.addConstrs(t <= s_j[i] for i in range(num_inequa))

        # ??? ???????????????????????????????????
        model.addConstr(Pd <= 1.05*Pd_original, name='the upper range of Pd.')
        model.addConstr(0.95 * Pd_original <= Pd, name='the lower range of Pd.')


        # ???????????????????????? max t, ??? ?????? ???????????????
        model.setObjective(t, GRB.MAXIMIZE)
        model.setParam('TimeLimit', 300)
        # model.setParam('BestBdStop', 0.01)
        model.optimize()

        # get gurobi status
        status = model.status
        print(f"status of GUROBI: {status}")

        # if not solved return None
        if status != GRB.OPTIMAL and status != GRB.SUBOPTIMAL:
            return None

        else:# create the Solver return object??? ??????SolverOutput ?????????????????????????????????????????????????????????

            sol = {
                'U': V_re.X,
                'W': V_im.X,
                'Pg': Pg.X,
                'Qg': Qg.X,
                'theta': Pd.X,  # ??????????????????
                'mu': muu_.X,
                'lambda': [lambda_Vmax.X, lambda_Vmin.X, lambda_F.X, lambda_T.X, lambda_PGmax.X, lambda_PGmin.X],
                # 'slack': s_j, # ??????????????????*Awaiting Model update, ????????????????????????????????????????????????
                'F_Pg': F_Pg.tolist(),
                'mu_dG_dPg': mu_dG_dPg.X,
                'lambda_dH_dPg': lambda_dH_dPg.T.X,
                'slack_ele': [s_j[0].X, s_j[1].X, s_j[2].X, s_j[3].X, s_j[4].X, s_j[5].X, s_j[6].X, s_j[7].X, s_j[8].X, s_j[9].X, s_j[10].X, s_j[11].X ],
                't': t.X,
                'equality_indices': active_set
            } # F_Pg + mu_dG_dPg.T + lambda_dH_dPg.T == 0
            return sol

# my_QCQP = MPQCQP_Program(net=None)