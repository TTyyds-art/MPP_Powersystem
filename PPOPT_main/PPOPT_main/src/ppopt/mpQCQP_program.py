from typing import List, Optional, Tuple

import numpy
from numpy import array, Inf, any, isnan, ones, r_, finfo, \
    zeros, dot, absolute, log, flatnonzero as find
from numpy.linalg import norm
from pandapower.pypower.pipsver import pipsver
from scipy.sparse import vstack, hstack, eye, csr_matrix as sparse
from scipy.sparse.linalg import spsolve
from scipy.linalg import pinv

EPS = finfo(float).eps  # 为变量EPS赋值为float类型的最小正数精度值
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

import sys, os

path_current = '/home/huzuntao/PycharmProjects/MPP_Powersystem/'
path_ = os.getcwd()
if path_current not in sys.path:
    sys.path.insert(1, '/home/huzuntao/PycharmProjects/MPP_Powersystem/')
elif path_ not in sys.path:
    sys.path.insert(1, path_)

from PPOPT_main.PPOPT_main.src.ppopt.solver import Solver

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

    # ieq_idxs/active_set = [lines_f, lines_t, Vm, Pgmax, Pgmin]
    # 简化版如下：
    # 默认active_set中的元素是按照以下顺序排列的：
    # lines_f (nl), lines_t (nl), Vm<=max (nb-1), Pg<=max (n_Pg), Vm>= (nb-1), Pg>=min (n_Pg)
    act_f = [True if i in active_set else False for i in range(nl)]
    act_t = [True if i in active_set else False for i in range(nl, 2 * nl)]

    act_Vmax = [True if i in active_set else False for i in range(2 * nl, 2 * nl + nb - 1)]

    act_Pgmax = [True if i in active_set else False for i in
                 range(2 * nl + nb - 1, 2 * nl + nb - 1 + n_Pg)]

    act_Qgmax = [True if i in active_set else False for i in
                 range(2 * nl + nb - 1 + n_Pg, 2 * nl + nb - 1 + 2 * n_Pg)]

    act_Vmin = [True if i in active_set else False for i in
                range(2 * nl + nb - 1 + 2 * n_Pg, 2 * nl + 2 * nb - 2 + 2 * n_Pg)]

    act_Pgmin = [True if i in active_set else False for i in
                 range(2 * nb - 2 + 2 * nl + 2 * n_Pg, 2 * nb - 2 + 2 * nl + 3 * n_Pg)]

    act_Qgmin = [True if i in active_set else False for i in
                 range(2 * nb - 2 + 2 * nl + 3 * n_Pg, 2 * nb - 2 + 2 * nl + 4 * n_Pg)]

    return act_Vmax, act_Vmin, act_f, act_t, act_Pgmax, act_Pgmin, act_Qgmax, act_Qgmin


def create_inact_index(n_eq, nb, nl, n_Pg, active_set):
    '''
    create index for \lambda and \nu
    :param n_eq: the number of equality
    :param nb: the number of buses
    :param nl: the number of lines
    :param n_Pg: the number of generators
    :return: indexes for non-activated constraints
    '''

    inact_f = [i for i in range(nl) if i not in active_set]
    inact_t = [i for i in range(nl, 2 * nl) if i not in active_set]

    inact_Umax = [i for i in range(2 * nl, 2 * nl + nb - 1) if i not in active_set]

    inact_Pgmax = [i for i in range(2 * nl + nb - 1, 2 * nl + nb - 1 + n_Pg) if i not in active_set]
    # act_Qgmax = [True if i in active_set else False for i in
    # range(2 * nl + nb - 1 + n_Pg, 2 * nl + nb - 1 + 2 * n_Pg)]

    inact_Qgmax = [i for i in range(2 * nl + nb - 1 + n_Pg, 2 * nl + nb - 1 + 2 * n_Pg) if i not in active_set]

    inact_Umin = [i for i in range(2 * nl + nb - 1 + 2 * n_Pg, 2 * nl + 2 * nb - 2 + 2 * n_Pg) if i not in active_set]

    inact_Pgmin = [i for i in range(2 * nb + 2 * nl + 2 * n_Pg - 2, 2 * nb + 2 * nl + 3 * n_Pg - 2) if
                   i not in active_set]
    inact_Qgmin = [i for i in range(2 * nb + 2 * nl + 3 * n_Pg - 2, 2 * nb + 2 * nl + 4 * n_Pg - 2) if
                   i not in active_set]

    return inact_Umax, inact_Umin, inact_f, inact_t, inact_Pgmax, inact_Pgmin, inact_Qgmax, inact_Qgmin


def update_gh(x0, xmin, xmax, dgn, dhn, h, l=None, u=None, A=None, ieq_idxs=None):
    '''
    update the equality constraints and inequality constraints
    :param x0:  [theta, v, Pg, Qg]
    :param xmin: lower bound for x0
    :param xmax: upper bound for x0
    :param dgn:  original dg
    :param dhn:  original dh
    :param flow_max: max current flow
    :param l:  lower bound for Ax
    :param u:  upper bound for Ax
    :param A:  matrix A (extra linear constraints)
    :return: dg (updated), dh (updated), be, bi
    '''

    # 在这个位置，根据pips.py， 添加线性约束，再将非线性与线性结合起来，形成一个新的模型
    ## 1. 添加线性约束
    ### 1.1 获得x0的维度nx
    nx = x0.shape[0]  # nx includes p,q,v,theta
    ## 2. 将非线性与线性结合起来，形成一个新的模型
    # add var limits to linear constraints
    eyex = eye(nx, nx, format="csr")
    AA = eyex if A is None else vstack([eyex, A], "csr")
    n_h_ieq = h.shape[0]  # nl是非线性不等式约束（电流约束）的个数，也是线数量的2倍
    if l is None:
        ll = xmin  # xmin是一个nx维的向量，表示x的下界
        uu = xmax  # 同上
    else:
        ll = r_[xmin, l]  # r_是按列合并, 维度不变
        uu = r_[xmax, u]  # 同上

    # split up linear constraints
    ieq = find(absolute(uu - ll) <= EPS)  # 约束条件的上下界相等的索引
    igt = find((uu >= 1e10) & (ll > -1e10))  # 约束条件的上界为无穷大，下界为有穷大的索引
    ilt = find((ll <= -1e10) & (uu < 1e10))  # 约束条件的上界为有穷大，下界为无穷大的索引
    ibx = find((absolute(uu - ll) > EPS) & (uu < 1e10) & (ll > -1e10))  # 约束条件的上下界都为有穷大的索引
    # zero-sized sparse matrices unsupported
    Ae = AA[ieq, :] if len(ieq) else None  # 等式约束的系数
    if len(ilt) or len(igt) or len(ibx):
        idxs = [(1, ilt), (-1, igt), (1, ibx), (-1, ibx)]
        Ai = vstack([sig * AA[idx, :] for sig, idx in idxs if len(idx)], 'csr')  # 不等式约束的系数
    else:  # 只有等式约束 ieq 的情况
        Ai = None
    be = uu[ieq]  # 等式约束的常数项 TODO: 是否要补充上其他等式约束的常数项？
    if dhn is None:  # 如果没有非线性不等式约束
        bi = None
    else:
        bi = r_[dhn @ x0 - h, uu[ilt], -ll[igt], uu[ibx], -ll[ibx]]  # 不等式约束的常数项

    # evaluate cost f(x0) and constraints g(x0), h(x0)
    x = x0
    # gn, hn = self.raw['g'][0], self.raw['g'][1]  # nonlinear constraints
    # dgn, dhn = self.raw['dg'][0], self.raw['dg'][1]
    # h = hn if Ai is None else r_[hn.reshape(len(hn), ), Ai * x - bi]  # inequality constraints
    # g = gn if Ae is None else r_[gn, Ae * x - be]  # equality constraints

    if (dhn is None) and (Ai is None):  # 如果没有不等式约束
        dh = None
    elif dhn is None:  # 如果没有非线性不等式约束
        dh = Ai.T
    elif Ai is None:  # 如果没有线性不等式约束
        dh = dhn
    else:  # 如果有非线性不等式约束和线性不等式约束
        dh = hstack([dhn.T, Ai.T])

    if (dgn is None) and (Ae is None):  # 如果没有等式约束
        dg = None
    elif dgn is None:  # 如果没有非线性等式约束
        dg = Ae.T
    elif Ae is None:  # 如果没有线性等式约束
        dg = dgn
    else:  # 如果有非线性等式约束和线性等式约束
        dg = hstack([dgn.T, Ae.T])

    if ieq_idxs is None:
        return dg, dh, be, bi
    else:  # 返回不等式约束的索引，包括线电流的不等式约束
        ieq_idxs = {'n_h_ieq': arange(n_h_ieq),  # 非线性不等式约束的索引，来自于上面的公式
                    'ilt': ilt,  # 对x的线性不等式约束的索引； x[ilt] <= uu[ilt]
                    'igt': igt,  # 对x的线性不等式约束的索引； x[igt] >= ll[igt]
                    'ibx u-bound': ibx,  # 对x的线性不等式约束的索引； x[ibx] <= uu[ibx]
                    'ibx l-bound': ibx}  # 对x的线性不等式约束的索引； x[ibx] >= ll[ibx]
        return ieq_idxs


def clear_om(om, net, update_gh=update_gh):
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
    ll[xmin == -Inf] = -1e11  ## replace Inf with numerical proxies
    uu[xmax == Inf] = 1e11
    Va_refs = bus[bus[:, BUS_TYPE] == REF, VA] * (pi / 180)
    # print(f"Va_refs: {Va_refs}")
    ll[vv["i1"]["Va"]:vv["iN"]["Va"]] = -np.ones_like(bus[:, VA]) * (pi / 2)  # Va lower bound 赋值
    uu[vv["i1"]["Va"]:vv["iN"]["Va"]] = np.ones_like(bus[:, VA]) * (pi / 2)  # Va upper bound 赋值
    ## deal with the Va_refs
    ll[vv["i1"]["Va"]:vv["iN"]["Va"]][bus[:, BUS_TYPE] == REF] = Va_refs  # Va_refs lower bound 赋值
    uu[vv["i1"]["Va"]:vv["iN"]["Va"]][bus[:, BUS_TYPE] == REF] = Va_refs  # Va_refs upper bound 赋值
    ## deal with reactive power, whose ll and uu is 1e9
    # ll[ll < -1e4] = -100
    # uu[uu > 1e4] = 100
    v_max = uu[vv["i1"]["Vm"]:vv["iN"]["Vm"]][-1]

    # 赋值
    x0[vv["i1"]["Vm"]:vv["iN"]["Vm"]] = bus[:, VM]

    Va_refs = bus[bus[:, BUS_TYPE] == REF, VA] * (pi / 180)
    x0[vv["i1"]["Va"]:vv["iN"]["Va"]] = bus[:, VA] * (pi / 180)  # 赋值

    ## 赋值
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

    ## 不等式约束： 线电压的功率
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

    para_results = [Pd, Qd, slack_v, lb_update, V_ll, V_uu, Pg_lower, Pg_upper, Qg_lower, Qg_upper, nb, nl, gencost,
                    Ybus, Yf, Yt, ngon, Cg, baseMVA, ipol, First_Or_Con, flow_max, x0]

    update_dgh = lambda dgn, dhn: update_gh(x0, xmin, xmax, dgn, dhn, h, l=None, u=None, A=None)
    ieq_idxs = update_gh(x0, xmin, xmax, dgn=None, dhn=None, h=h, l=None, u=None, A=None, ieq_idxs=1)
    # ieq_idxs = update_gh(x0, xmin, xmax, dgn, dhn, h, l=None, u=None, A=None, ieq_idxs=None)

    return para_results, update_dgh, ieq_idxs


def opt_modelling(params, active_set=None, ieq_idxs=None):
    Pd_original, Qd, slack_v, lb_update, V_ll, V_uu, Pg_lower, Pg_upper, Qg_lower, Qg_upper, nb, nl, gencost, Ybus, Yf, Yt, ngon, \
    Cg, baseMVA, ipol, First_Or_Con, flow_max, _ = params

    orig_equa = 6 * nb + 4 * nl + 2  # 6*nb:temp1, temp2, P_net,Q_net, G_real, G_imag; 4*nl: line_temp_1~4; 2: slack U W.

    # ieq_idxs = {'n_h_ieq': arange(n_h_ieq),  # 非线性不等式约束的索引，来自于上面的公式
    #             'ilt': ilt + n_h_ieq,
    #             'igt': igt + n_h_ieq,
    #             'ibx u-bound': ibx + n_h_ieq,
    #             'ibx l-bound': ibx + n_h_ieq}
    # n_h_ieq = ieq_idxs['n_h_ieq']  # 非线性不等式约束的索引, 线电流的约束
    # ilt = ieq_idxs['ilt']  # 线性不等式约束的索引， (a,b]
    # igt = ieq_idxs['igt']  # 线性不等式约束的索引   [a,b)
    # ibx_u = ieq_idxs['ibx u-bound']  # 线性不等式约束的索引  [a,b]的上界
    # ibx_l = ieq_idxs['ibx l-bound']  # 线性不等式约束的索引  [a,b]的下界
    n_h_ieq = arange(2 * nl)
    # create the model; set the model type: NonConvex.
    model = gp.Model("PowerGeneration")
    model.setParam('NonConvex', 2)
    # 将输出设置为 0，关闭 verbose 输出
    model.Params.OutputFlag = 0

    if active_set is None:
        Pd = Pd_original
        active_set = []
    else:
        Pd = model.addMVar(nb, vtype=GRB.CONTINUOUS, name='Changeable variable: theta.')
        model.addConstr(0.8 * Pd_original <= Pd, name='Range of theta.')
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

    # 不等式约束从这里开始。02.21
    # 处理line的不等式约束，共添加 2* nl 个约束
    Line_c_1_ineq = model.addConstrs(
        (Line_temp_1[i] ** 2 + Line_temp_2[i] ** 2 <= np.real(flow_max[i]) for i in range(nl) if
         i not in active_set),
        'Branch Con 1')  # p.u.
    Line_c_1_eq = model.addConstrs(
        (Line_temp_1[i] ** 2 + Line_temp_2[i] ** 2 == np.real(flow_max[i]) for i in range(nl) if
         i in active_set),
        'Branch Con 1')  # p.u.

    Line_c_2_ineq = model.addConstrs(
        (Line_temp_3[i] ** 2 + Line_temp_4[i] ** 2 <= np.real(flow_max[i]) for i in range(nl) if
         i + len(n_h_ieq / 2) not in active_set),
        'Branch Con 2')  # p.u.
    Line_c_2_eq = model.addConstrs(
        (Line_temp_3[i] ** 2 + Line_temp_4[i] ** 2 == np.real(flow_max[i]) for i in range(nl) if
         i + len(n_h_ieq / 2) in active_set),
        'Branch Con 2')  # p.u.

    # 开始添加不等式约束； (nb - 1) * 2 ， 注意: "减一"; 此处的不等式约束是对于 Vm 的不等式约束
    NU_upper_ineq = model.addConstrs(
        (V_re[i] ** 2 + V_im[i] ** 2 <= V_uu ** 2 for i in range(1, nb) if i + len(n_h_ieq) - 1 not in active_set),
        'V upper bound')  # 对 range 起始值的处理，是处理树状分层搜索的突破口；有一"减"，对应的有一"加"。
    NU_upper_eq = model.addConstrs(
        (V_re[i] ** 2 + V_im[i] ** 2 == V_uu ** 2 for i in range(1, nb) if i + len(n_h_ieq) - 1 in active_set),
        'V upper bound')  # 对 range 起始值的处理，是处理树状分层搜索的突破口；有一"减"，对应的有一"加"。
    # 下界约束
    NU_lower_ineq = model.addConstrs(
        (V_re[i] ** 2 + V_im[i] ** 2 >= V_ll ** 2 for i in range(1, nb) if i + len(n_h_ieq) + nb - 2 not in active_set),
        'V lower bound')
    NU_lower_eq = model.addConstrs(
        (V_re[i] ** 2 + V_im[i] ** 2 == V_ll ** 2 for i in range(1, nb) if i + len(n_h_ieq) + nb - 2 in active_set),
        'V lower bound')

    # 添加Pg的上下限约束，一共 2* len(Pg)
    # 针对有功发电的上下界进行处理，暂时不管无功出力
    Pg_c_ineq = model.addConstrs(
        Pg[i] <= Pg_upper[i] for i in range(len(Pg_upper)) if i + nb * 2 + nl * 2 - 2 not in active_set)
    Pg_c_eq = model.addConstrs(
        Pg[i] == Pg_upper[i] for i in range(len(Pg_upper)) if i + nb * 2 + nl * 2 - 2 in active_set)
    Pg_lower_eq = model.addConstrs(
        Pg[i] == Pg_lower[i] for i in range(len(Pg_upper)) if
        i + nb * 2 + nl * 2 - 2 + len(Pg_upper) in active_set)
    Pg_lower_ineq = model.addConstrs(
        Pg[i] >= Pg_lower[i] for i in range(len(Pg_upper)) if
        i + nb * 2 + nl * 2 - 2 + len(Pg_upper) not in active_set)

    # set the objective of the model
    # The "gencost" changes with different cases. Keep in Mind!
    # model.setObjective(gencost[ipol, First_Or_Con]@Pg**2+gencost[ipol, First_Or_Con+1]@Pg, GRB.MINIMIZE)
    if len(active_set) == 0:  # 如果没有激活不等式约束，那么就是一个标准的OPF问题
        # 设置目标函数
        model.setObjective(gencost[ipol, First_Or_Con] @ Pg, GRB.MINIMIZE)
        # 计算优化结果
        model.optimize()

        # 处理优化结果，获得发电的有功和无功，以及优化的目标函数值
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
    求解非凸的OPF问题。

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

    def __init__(self, net=None, lb_loads=0.8, ub_loads=1.2):
        """Initialized the Non-convex OPF problem from a net object."""

        if net is not None:

            self.net = net
            self.om, self.ppopt, self.raw = pp.runopp(net, delta=1e-16, RETURN_RAW_DER=1)  # solve the OPF problem
        else:
            raise Exception("The input is not a net object built on the PandaPower.")
            # print()
        # print(f"The constraint: {self.raw['g']}")
        # print(f"The constraint: {self.raw['dg']}")

        res_cost = self.net.res_cost
        print(f"res_line:\n {self.net.res_line}")

        # get parameters from net obj
        self.parameters_grid, update_dgdh, ieq_idxs = clear_om(om=self.om, net=self.net)

        # ppopt = self.om.get_ppc()
        # print(f"ppopt: {ppopt['OPF_FLOW_LIM']}")

        # parameters_grid: Pd, Qd, slack_v, lb_update, V_ll, V_uu, Pg_lower, Pg_upper, Qg_lower, Qg_upper, nb, nl, gencost, Ybus, Yf, Yt, ngon,\
        # Cg, baseMVA,ipol, First_Or_Con
        nb, nl, self.x_0 = self.parameters_grid[10], self.parameters_grid[11], self.parameters_grid[-1]

        dg, dh, self.Lxx = self.raw['dg'][0], self.raw['dg'][1], self.raw['Lxx']  # dg, dh可能在储存的时候就已经被转置了；Lxx是对称的，所以不用怕
        self.dg, self.dh, self.be, self.bi = update_dgdh(dg, dh)  # 更新dg, dh: 包含了本来的等式和不等式，及线性等式和不等式

        self.nb, self.nl, self.n_Pg = nb, nl, len(self.parameters_grid[6])
        self.model_grid, self.Pg_star, self.Qg_star, optimized_obj = opt_modelling(params=self.parameters_grid,
                                                                                   ieq_idxs=ieq_idxs)
        self.equality_indices = [*range(self.dg.shape[1])]  # 等式约束的索引

        # 生成对负载的约束（lower bound, upper bound）; A_t*loads <= b_t
        Pd = self.parameters_grid[0]
        Qd = self.parameters_grid[1]
        idx_load = find(Pd != 0)
        n_load = len(idx_load)
        self.A_t = np.vstack([np.eye(n_load * 2), -np.eye(n_load * 2)])  # 生成A_t矩阵; A_t*loads <= b_t
        self.b_t = np.vstack([np.eye(n_load * 2) * ub_loads, -np.eye(n_load * 2) * lb_loads]) @ np.vstack(
            [Pd[idx_load], Qd[idx_load]])  # 生成b_t=[ub*I, -lb*I].T*loads_0

        self.A, self.b = self.cal_ieq_params()  # 生成A, b矩阵, 作为不等式约束的参数
        self.F = zeros((self.A.shape[0], n_load))  # 生成F矩阵, Ax-F*loads <= b

        self.solver = Solver()

        if abs(res_cost / optimized_obj) - 1 <= 0.001:
            print('*' * 75)
            print("The result of GUROBI-model is exact! Very GOOD!")
        else:
            print(f'The solving error is {abs(res_cost / optimized_obj)}')

    def num_x(self) -> int:
        """Returns number of parameters."""
        return len(self.x_0)

    def num_t(self) -> int:
        """Returns number of uncertain variables."""
        return self.nb  # the number of Pd

    def num_constraints(self) -> int:
        """Returns number of constraints: linear constraints + quadratic constraints"""
        return self.dg.shape[1] + self.dh.shape[1]

    def num_inequality_constraints(self) -> int:
        return self.dh.shape[1]  #

    def num_equality_constraints(self) -> int:
        return len(self.equality_indices)

    def evaluate_objective(self, x: numpy.ndarray, theta_point: numpy.ndarray):
        pass

    def optimal_control_law(self, active_set: List[int] = None) -> Tuple:
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
            mu_ = self.raw['lmbda']['mu']
            active_set = find(mu_ > 0)

        mu = self.raw['lmbda']['mu']
        labda = self.raw['lmbda']['lam']
        n_c = self.dg.shape[1] + len(active_set)  # 等式约束和激活的不等式约束之和

        # 过滤Lxx
        #
        # 将稀疏矩阵转换为csr_matrix
        csr_dh, csr_dg = sparse(self.dh), sparse(self.dg)
        # 将稀疏矩阵转换为密集矩阵，并进行切片
        dh_iq, dg_eq = csr_dh.toarray(), csr_dg.toarray()  # dh_iq, dg_eq 分别指的是不等式约束和等式约束的导数
        labda_b = np.broadcast_to(labda.reshape((len(labda), 1)), (dg_eq.T.shape))
        mu_b = np.broadcast_to(mu.reshape((len(mu), 1)), (dh_iq.T.shape))
        M = vstack([
            hstack([self.Lxx, dg_eq, dh_iq[:, active_set]]),
            hstack([vstack([dg_eq.T, -(mu_b * dh_iq.T)[active_set]]), zeros((n_c, n_c))])
        ])  # M 的维度是 (n_x + n_c, n_x + n_c), z = [x, mu, labda[active_set]]; n_z = n_x + n_c #TODO 0519 加M的公式(次要）

        inverse_M = pinv(M.toarray())

        # print(f"the  M is {M}; \n The inverse_M is {inverse_M}")

        # 计算 N; N 的维度是 (n_x + n_c, n_pl+n_ql)
        Pd = self.parameters_grid[0]
        Qd = self.parameters_grid[1]
        idx_load = find(Pd != 0)
        n_load = len(idx_load)

        n_x = self.Lxx.shape[0]
        # L_sl_x 的形状是 (2*n_load, n_x), 因为有 2*n_load 个负载（有功加无功），n_x个决策变量
        L_sl_x = zeros((2 * n_load, n_x))
        # gh_sl 的形状是 (2*n_load, n_c), 因为有 2*n_load 个负载（有功加无功），n_c个约束
        gh_sl = zeros((2 * n_load, n_c))
        for i in range(n_load):
            gh_sl[i, idx_load[i]] = 1
            gh_sl[i + n_load, idx_load[i] + self.nb] = 1

        labda_b_T = np.broadcast_to(labda.reshape((1, len(labda))), (2 * n_load, len(labda)))
        mu_b_T = np.broadcast_to(mu.reshape((1, len(mu))), (2 * n_load, len(mu)))
        lab_mu = r_[ones([len(labda), 2 * n_load]), mu_b_T[:, active_set].T].T
        GH_sl = - lab_mu * gh_sl  # GH_sl 的形状是 (2*n_load, n_c)

        N = np.hstack([L_sl_x, GH_sl]).T  # N 的形状是 (2*n_load, n_x + n_c).T = (n_x + n_c, 2*n_load)

        A_x = (inverse_M @ N)[:n_x]  # A_x 的形状是 (n_x, 2*n_load)
        A_l = (inverse_M @ N)[n_x:]  # A_l 的形状是 (n_c, 2*n_load)

        x_0 = self.x_0
        b_x = x_0 - A_x @ r_[Pd[idx_load], Qd[idx_load]]
        b_l = r_[labda, mu[active_set]] - A_x @ r_[Pd[idx_load], Qd[idx_load]]

        return A_x, b_x, A_l, b_l

        # print(f"the N is {N}; \n Its shape is {N.shape}")

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

    def check_optimality(self, active_set, theta=None):
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
        if len(active_set) != self.num_x():  # 如果等式约束数量不等于参数数量num_x， 就不是最优？:可能和A有关系，可能与max_depth有关系
            # return False
            pass

        zeros = lambda x, y: numpy.zeros((x, y))
        # num_x = self.num_x()

        num_constraints = self.num_constraints()
        num_active = len(active_set)
        # num_theta_c = self.A_t.shape[0]
        n_eq = len(self.equality_indices)  # 等式约束的数量
        # num_activated = len(active_set) - len(self.equality_indices)

        # inactive = [i for i in range(num_constraints) if i not in active_set]

        # num_inactive = num_constraints - num_active
        num_inequa = num_constraints - n_eq

        # num_theta = self.num_t()

        # The "gencost" changes with different cases. Keep in Mind!
        # model.setObjective(gencost[ipol, First_Or_Con]@Pg**2+gencost[ipol, First_Or_Con+1]@Pg, GRB.MINIMIZE)
        # model.setObjective(gencost[ipol, First_Or_Con] @ Pg, GRB.MINIMIZE); while 5 bus
        # [Pd, Qd, slack_v, lb_update, V_ll,            ; 5
        # V_uu, Pg_lower, Pg_upper, Qg_lower, Qg_upper, ; 5
        # nb, nl, gencost, Ybus, Yf,                    ; 5
        # Yt, ngon, Cg, baseMVA, ipol,                  ; 5
        # First_Or_Con, flow_max] = param.              ; 5
        Pd_original, Qd, slack_v, lb_update, V_ll, V_uu, Pg_lower, Pg_upper, Qg_lower, Qg_upper, nb, nl, gencost, Ybus, Yf, \
        Yt, ngon, Cg, baseMVA, ipol, First_Or_Con, flow_max, _ = self.parameters_grid
        # gencost, ipol, First_Or_Con = self.parameters_grid[12], self.parameters_grid[19], self.parameters_grid[20]
        Fx = []
        Fx.append(zeros(nb * 2, 1))  # add zeros for U, W, where V = U+jW.
        F_Pg = gencost[ipol, First_Or_Con].reshape([-1, 1])
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
            Pd = model.addMVar(nb, vtype=GRB.CONTINUOUS, name='Changeable variable: theta.')  # 默认lb = 0，是可以的
        elif len(theta) >= 0:
            Pd = model.addMVar(nb, lb=theta, ub=theta, vtype=GRB.CONTINUOUS, name='fixed theta for Pd')
            print('-' * 75)
            print(f"Checking {theta}, the point is in the CR or not.")
        else:
            raise Exception(f"Theta is not required loads, it is {theta}!")

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
        muu_ = model.addMVar(2 * nb, lb=-infi, vtype=GRB.CONTINUOUS, name='mu')  # 为什么没有 lambda? 在后面。
        # lambda_ = model.addMVar(2*(nb-1)+2*nl+ngon*2, lb=0, vtype=GRB.CONTINUOUS, name='lambda')
        t = model.addMVar((1, 1), lb=1e-3, ub=10, vtype=GRB.CONTINUOUS,
                          name='t')  # ub = 10 为了让t别太大了,lb=1e-3 t经常在0的位置算，耗费时间
        s_j = model.addVars(num_inequa, lb=0, vtype=GRB.CONTINUOUS, name='S_j')
        R_G_Us = model.addMVar((nb, nb), lb=-infi, vtype=GRB.CONTINUOUS, name='R{G^s_U}')
        R_G_Ws = model.addMVar((nb, nb), lb=-infi, vtype=GRB.CONTINUOUS, name='R{G^s_W}')
        H_f_U = model.addMVar((nl, nb), lb=-infi, vtype=GRB.CONTINUOUS, name='H^f_U')
        H_f_W = model.addMVar((nl, nb), lb=-infi, vtype=GRB.CONTINUOUS, name='H^f_W')
        H_t_U = model.addMVar((nl, nb), lb=-infi, vtype=GRB.CONTINUOUS, name='H^t_U')
        H_t_W = model.addMVar((nl, nb), lb=-infi, vtype=GRB.CONTINUOUS, name='H^t_W')

        # the parameters of branches
        G = np.real(Ybus)
        B = np.imag(Ybus)
        # the parameters for currents in branches
        G_yf, B_yf = np.real(Yf), np.imag(Yf)
        G_yt, B_yt = np.real(Yt), np.imag(Yt)

        Real_G_s_U = model.addConstr(
            G @ V_re * np.eye(nb) - B @ V_im * np.eye(nb) + V_re * np.eye(nb) @ G + V_im * np.eye(nb) @ B == R_G_Us,
            'Temp Constrs:R_G_Us ')
        Real_G_s_W = model.addConstr(
            B @ V_re * np.eye(nb) + G @ V_im * np.eye(nb) + V_re * np.eye(nb) @ B - V_im * np.eye(nb) @ G == R_G_Ws,
            'Temp Constrs: R_G_Ws')

        # 是不是还有Imag_G_s_U,Imag_G_s_W? 暂时不用，因为mu=0 就可以掩盖掉了

        temp_A = model.addMVar((nl, nl), lb=-infi, vtype=GRB.CONTINUOUS, name='[Cal{ A }]')
        temp_B = model.addMVar((nl, nl), lb=-infi, vtype=GRB.CONTINUOUS, name='[Cal{ B }]')

        ## 需要根据active set变化
        # 生成替代者 A 和 B
        Temp_A_c = model.addConstr(G_yf @ V_re * np.eye(nl) - B_yf @ V_im * np.eye(nl) == temp_A, 'Temp Constr: Cal A')
        Temp_B_c = model.addConstr(B_yf @ V_re * np.eye(nl) + G_yf @ V_im * np.eye(nl) == temp_B, 'Temp Constr: Cal B')

        # 生成λ.T*dG/dU,λ.T*dG/dW, λ.T*dG/dPg, λ.T*dG/dQg
        mu_dG_dU = model.addMVar((1, nb), lb=-infi, vtype=GRB.CONTINUOUS, name='μ.T x 等式约束对U的导数')
        mu_dG_dW = model.addMVar((1, nb), lb=-infi, vtype=GRB.CONTINUOUS, name='μ.T x 等式约束对W的导数')
        mu_dG_dPg = model.addMVar((1, ngon), lb=-infi, vtype=GRB.CONTINUOUS, name='μ.T x 等式约束对Pg的导数')
        mu_dG_dQg = model.addMVar((1, len(Pg_upper)), lb=-infi, vtype=GRB.CONTINUOUS, name='μ.T x 等式约束对Qg的导数')

        # 因为mu_ 是vector，所以mu_[:nb]可以取值，它和单列的向量是不一样的。
        model.addConstr(muu_.T[:nb] @ R_G_Us + muu_.T[nb:] @ -R_G_Ws == mu_dG_dU,
                        name='μ.T x dG/dU')  # I{G^s_U} = -R{G^s_W}
        model.addConstr(muu_.T[:nb] @ R_G_Ws + muu_.T[nb:] @ R_G_Us == mu_dG_dW,
                        name='μ.T x dG/dW')  # I{G^s_W} = R{G^s_U}
        model.addConstr(muu_.T[:nb] @ -Cg + muu_.T[nb:] @ zeros(nb, len(Pg_upper)) == mu_dG_dPg,
                        name='μ.T x dG/dPg')  # 注释了这句话！！
        model.addConstr(muu_.T[:nb] @ zeros(nb, len(Pg_upper)) + muu_.T[nb:] @ -Cg == mu_dG_dQg, name='μ.T x dG/dQg')

        # 下面两式的每一行就是一条线路（分别对应实部和虚部），选择约束时只需要用 H[i]，选择某一条线路即可，用for循环处理
        H_f_U_Constr = model.addConstr(2 * temp_A @ G_yf + 2 * temp_B @ B_yf == H_f_U, "生成了等式H^2_U")
        H_f_W_Constr = model.addConstr(-2 * temp_A @ B_yf + 2 * temp_B @ G_yf == H_f_W, "需要根据活跃约束集来变化")
        H_t_U_Constr = model.addConstr(2 * temp_A @ G_yt + 2 * temp_B @ B_yt == H_t_U, "生成了等式H^2_U")
        H_t_W_Constr = model.addConstr(-2 * temp_A @ B_yt + 2 * temp_B @ G_yt == H_t_W, "需要根据活跃约束集来变化")

        # shape_tran = np.array([[0,1,0,0], [0,0,1,0], [0,0,0,1]])  ##TODO change with the topology
        shape_tran = np.array([[0, 1, 0], [0, 0, 1]])
        H_vmax_U = 2 * V_re[1:] * np.eye(nb - 1) @ shape_tran
        H_vmax_W = 2 * V_im[1:] * np.eye(nb - 1) @ shape_tran
        H_vmin_U = -H_vmax_U
        H_vmin_W = -H_vmax_W

        # 在构造这个约束模型的时候，Active set 同时体现在拉格朗日函数和约束的写法上。

        # 生成μ.T*dH/dU,μ.T*dH/dW, μ.T*dH/dPg, μ.T*dH/dQg ; 这部分的重点是体现active set

        lambda_dH_dU = model.addMVar((1, nb), lb=-infi, vtype=GRB.CONTINUOUS, name='μ.T x 活跃不等式约束对U的导数')
        lambda_dH_dW = model.addMVar((1, nb), lb=-infi, vtype=GRB.CONTINUOUS, name='μ.T x 活跃不等式约束对W的导数')
        lambda_dH_dPg = model.addMVar((1, len(Pg_upper)), lb=-infi, vtype=GRB.CONTINUOUS, name='μ.T x 活跃不等式约束对Pg的导数')
        lambda_dH_dQg = model.addMVar((1, len(Pg_upper)), lb=-infi, vtype=GRB.CONTINUOUS, name='μ.T x 活跃不等式约束对Qg的导数')

        #  dH/dPg 不等于 0 , 等于I 和-I ； dH/dQg 等于 0 ###todo
        n_Pg = len(Pg_upper)

        # 不等式顺序：2(nb-1)个 V**2 约束 ； 2 nl 个线电流约束； n_Pg 个发电上限约束；n_Pg 个发电下限约束
        act_Vmax, act_Vmin, act_f, act_t, act_Pgmax, act_Pgmin, act_Qgmax, act_Qgmin \
            = create_act_index(n_eq, nb, nl, n_Pg, active_set)

        lambda_vmax = [infi if num is True else 0 for num in act_Vmax]
        lambda_Vmax = model.addMVar((nb - 1), lb=0, ub=lambda_vmax, vtype=GRB.CONTINUOUS, name='lambda Vmax')
        lambda_vmin = [infi if num is True else 0 for num in act_Vmin]
        lambda_Vmin = model.addMVar(nb - 1, lb=0, ub=lambda_vmin, vtype=GRB.CONTINUOUS,
                                    name='lambda Vmin')
        lambda_f = [infi if num is True else 0 for num in act_f]
        lambda_F = model.addMVar(nl, lb=0, ub=lambda_f, vtype=GRB.CONTINUOUS,
                                 name='lambda F')
        lambda_t = [infi if num is True else 0 for num in act_t]
        lambda_T = model.addMVar(nl, lb=0, ub=lambda_t, vtype=GRB.CONTINUOUS,
                                 name='lambda T')
        lambda_Pgmax = [infi if num is True else 0 for num in act_Pgmax]
        lambda_PGmax = model.addMVar(n_Pg, lb=0, ub=lambda_Pgmax, vtype=GRB.CONTINUOUS,
                                     name='lambda PGmax')
        lambda_Qgmax = [infi if num is True else 0 for num in act_Qgmax]
        lambda_QGmax = model.addMVar(n_Pg, lb=0, ub=lambda_Qgmax, vtype=GRB.CONTINUOUS,  # n_Qg=n_Pg
                                     name='lambda QGmax')
        lambda_Pgmin = [infi if num is True else 0 for num in act_Pgmin]
        lambda_PGmin = model.addMVar(n_Pg, lb=0, ub=lambda_Pgmin, vtype=GRB.CONTINUOUS,  # n_Qg=n_Pg
                                     name='lambda PGmin')
        lambda_Qgmin = [infi if num is True else 0 for num in act_Qgmin]
        lambda_QGmin = model.addMVar(n_Pg, lb=0, ub=lambda_Qgmin, vtype=GRB.CONTINUOUS,  # n_Qg=n_Pg
                                     name='lambda QGmin')

        model.addConstr(lambda_F @ H_f_U + lambda_T @ H_t_U
                        + lambda_Vmax @ H_vmax_U + lambda_Vmin @ H_vmin_U == lambda_dH_dU,
                        name='λ.T x dH/dU')  #
        model.addConstr(lambda_F @ H_f_W + lambda_T @ H_t_W
                        + lambda_Vmax @ H_vmax_W + lambda_Vmin @ H_vmin_W == lambda_dH_dW,
                        name='λ.T x dH/dW')  #
        model.addConstr(lambda_PGmax @ np.eye(n_Pg) - lambda_PGmin @ np.eye(n_Pg) == lambda_dH_dPg,  # 就是box约束的求导
                        name='λ.T x dH/dPg')
        model.addConstr(lambda_QGmax @ np.eye(n_Pg) - lambda_QGmin @ np.eye(n_Pg) == lambda_dH_dQg,  # 就是box约束的求导；
                        # n_Qg=n_Pg
                        name='λ.T x dH/dQg')

        # dF/dX + (λ.T x dG/dX).T + (μ.T x dH/dX).T = 0
        dL_dU = model.addConstr(zeros(nb, 1) + mu_dG_dU.T + lambda_dH_dU.T == 0, 'd Largane/dU = 0')
        dL_dW = model.addConstr(zeros(nb, 1) + mu_dG_dW.T + lambda_dH_dW.T == 0, 'd Largane/dW = 0')
        dL_dPg = model.addConstr(F_Pg + mu_dG_dPg.T + lambda_dH_dPg.T == 0, 'd Largane/dPg = 0')  # mu_dH_dPg = 0
        # dL_dQg = model.addConstr(zeros(len(Qg_upper), 1) + mu_dG_dQg.T + lambda_dH_dQg.T == 0,
        #                          'd Largane/dQg = 0')  # mu_dH_dQg = 0
        # 22.22.47 应该有 dL_dQg
        ## 处理其他约束: 等式约束

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

        # 不等式约束;  未激活的一组
        # sj 与active set 一同处理
        inact_Vmax, inact_Vmin, inact_f, inact_t, inact_Pgmax, inact_Pgmin, inact_Qgmax, inact_Qgmin = \
            create_inact_index(n_eq, nb, nl, n_Pg, active_set)

        # 下式左侧是否应该乘上baseMVA
        Line_c_1_ineq = model.addConstrs(
            (Line_temp_1[i] ** 2 + Line_temp_2[i] ** 2 + s_j[i] == np.real(
                flow_max[i]) for i in range(nl) if i in inact_f[0:nl]),
            'Branch Current from constraints')  # p.u.
        Line_c_2_ineq = model.addConstrs(
            (Line_temp_3[i] ** 2 + Line_temp_4[i] ** 2 + s_j[i + len(inact_f)] ==
             np.real(flow_max[i]) for i in range(nl) if i + nl in inact_t), 'Branch Current to constraints')  # p.u.

        V_upper_ineq = model.addConstrs(
            (V_re[i] ** 2 + V_im[i] ** 2 + s_j[i - 1 + len(inact_f) + + len(inact_t)] == V_uu ** 2 for i in range(1, nb)
             if i-1 + 2 * nl in inact_Vmax),
            'V upper bound')  # 不活跃为啥要相等？ 因为加了 sj.; 该约束加了 nb-1 个，因为跳过了第 0 个，所以用 i-1 检测第一个约束
        #
        V_lower_ineq = model.addConstrs(
            (V_ll ** 2 - V_re[i] ** 2 - V_im[i] ** 2 + s_j[
                i + len(inact_Vmax) + len(inact_Pgmax) + len(inact_Qgmax) + len(inact_f) + len(inact_t) - 1] == 0 for i in range(1, nb) if
             i-1 + 2 * nl + (nb - 1) + 2*n_Pg in inact_Vmin), # 应该有个减 1?
            'V lower bound')

        # 针对有功发电的上下界进行处理，暂时不管无功出力
        Pg_upper_ineq = model.addConstrs(
            Pg[i] + s_j[i + len(inact_Vmax) + len(inact_f) + len(inact_t)] == Pg_upper[i] for i in
            range(n_Pg) if i + 2 * nl + nb - 1 in inact_Pgmax)
        Pg_lower_ineq = model.addConstrs(
            Pg_lower[i] - Pg[i] + s_j[
                i + len(inact_Vmax) + len(inact_Vmin) + len(inact_Qgmax) + len(inact_f) + len(inact_t) + len(inact_Pgmax)] == 0 for i in
            range(n_Pg) if i + 2 * nb + 2 * nl - 2 + 2*n_Pg in inact_Pgmin)

        # Qg_upper_ineq = model.addConstrs(
        #     Qg[i] + s_j[i + len(inact_Vmax) + len(inact_Pgmax) + len(inact_f) + len(inact_t)] == Qg_upper[i] for i in
        #     range(n_Pg) if i + 2 * nl + nb - 1 + n_Pg in inact_Qgmax)
        # Qg_lower_ineq = model.addConstrs(
        #     Qg_lower[i] - Qg[i] + s_j[
        #         i + len(inact_Vmax) + len(inact_Vmin) + len(inact_Qgmax) + len(inact_Pgmin) + len(inact_f) + len(inact_t) + len(
        #             inact_Pgmax)] == 0 for i in
        #     range(n_Pg) if i + 2 * nb + 2 * nl - 2 + 3*n_Pg in inact_Qgmin)

        # 激活约束部分： 要做等式处理
        # 下式左侧是否应该乘上baseMVA
        Line_c_1_eq = model.addConstrs(
            (Line_temp_1[i] ** 2 + Line_temp_2[i] ** 2 == np.real(
                flow_max[i]) for i in range(nl) if i in active_set),
            'Branch current from constraints')  # p.u.
        Line_c_2_eq = model.addConstrs(
            (Line_temp_3[i] ** 2 + Line_temp_4[i] ** 2 == np.real(flow_max[i]) for i in range(nl) if
             i + nl in active_set), 'Branch Con 2')  # p.u.

        V_upper_eq = model.addConstrs(
            (V_re[i] ** 2 + V_im[i] ** 2 == V_uu ** 2 for i in range(1, nb) if i-1 + 2 * nl in active_set),
            'V upper bound')  # 不活跃为啥要相等？ 因为加了 sj.
        #
        V_lower_eq = model.addConstrs(
            (V_ll ** 2 - V_re[i] ** 2 - V_im[i] ** 2 == 0 for i in range(1, nb) if
             (i - 1) + 2 * nl + (nb - 1) + 2*n_Pg in active_set),
            'V lower bound')

        # 针对有功发电的上下界进行处理
        Pg_upper_eq = model.addConstrs(
            Pg[i] == Pg_upper[i] for i in
            range(n_Pg) if i + nb + 2 * nl - 1 in active_set)
        Pg_lower_eq = model.addConstrs(
            Pg_lower[i] - Pg[i] == 0 for i in
            range(n_Pg) if i + 2 * nb + 2 * nl - 2 + 2*n_Pg in active_set)

        # 针对无功发电的上下界进行处理
        Qg_upper_eq = model.addConstrs(
            Qg[i] == Qg_upper[i] for i in
            range(n_Pg) if i + nb + 2 * nl - 1 + n_Pg in active_set)
        Qg_lower_eq = model.addConstrs(
            Qg_lower[i] - Qg[i] == 0 for i in
            range(n_Pg) if i + 2 * nb + 2 * nl - 2 + 3 * n_Pg in active_set)

        # 接下来处理关于λ的不等式
        model.addConstrs(t <= lambda_Vmax[act_Vmax] for i in lambda_vmax if i >= 1)  # 这样是正确的
        model.addConstrs(t <= lambda_Vmin[act_Vmin] for i in lambda_vmin if i >= 1)  #
        model.addConstrs(t <= lambda_F[act_f] for i in lambda_f if i >= 1)  #
        model.addConstrs(t <= lambda_T[act_t] for i in lambda_t if i >= 1)  #
        model.addConstrs(t <= lambda_PGmax[act_Pgmax] for i in lambda_Pgmax if i >= 1)  #
        model.addConstrs(t <= lambda_PGmin[act_Pgmin] for i in lambda_Pgmin if i >= 1)  #
        model.addConstrs(t <= lambda_QGmax[act_Qgmax] for i in lambda_Qgmax if i >= 1)  #
        model.addConstrs(t <= lambda_QGmin[act_Qgmin] for i in lambda_Qgmin if i >= 1)  #

        model.addConstrs(t <= s_j[i] for i in range(num_inequa))

        # 给 θ一个范围，一般是线性的  TODO 范围要可以定制的
        model.addConstr(Pd <= 1.2 * Pd_original, name='the upper range of Pd.')
        model.addConstr(0.8 * Pd_original <= Pd, name='the lower range of Pd.')

        # 将优化目标设置成 max t, 并 求解 优化问题，
        model.setObjective(t, GRB.MAXIMIZE)
        model.setParam('TimeLimit', 3000)
        # model.setParam('BestBdStop', 0.01)
        model.optimize()

        # get gurobi status
        status = model.status
        print(f"status of GUROBI: {status}")

        # if not solved return None
        if status != GRB.OPTIMAL and status != GRB.SUBOPTIMAL:
            return None

        else:  # create the Solver return object； 这个 SolverOutput 从其函数体来看，在这个位置没有任何意义

            sol = {
                'U': V_re.X,
                'W': V_im.X,
                'Pg': Pg.X,
                'Qg': Qg.X,
                'theta': Pd.X,  # 要设置成变量
                'mu': muu_.X,
                'lambda': [lambda_F.X, lambda_T.X, lambda_Vmax.X, lambda_PGmax.X, lambda_Vmin.X, lambda_PGmin.X],
                # 'slack': s_j, # 如果还是显示*Awaiting Model update, 就需要考虑是否没有用到这些约束！
                'F_Pg': F_Pg.tolist(),
                'mu_dG_dPg': mu_dG_dPg.X,
                'lambda_dH_dPg': lambda_dH_dPg.T.X,
                'slack_ele': [s_j[0].X, s_j[1].X, s_j[2].X, s_j[3].X, s_j[4].X, s_j[5].X, s_j[6].X, s_j[7].X, s_j[8].X,
                              s_j[9].X, s_j[10].X, s_j[11].X, s_j[12].X, s_j[13].X, s_j[14].X, s_j[15].X],
                't': t.X,
                'equality_indices': active_set
            }  # F_Pg + mu_dG_dPg.T + lambda_dH_dPg.T == 0
            return sol

    def cal_ieq_params(self):
        '''
        计算不等式约束的参数
        :return: parameters of ineqaulity constraints: A, b

        .. math::

            \begin{align*}
            Ax <= b \\
            h(x) = h(x_0) + \na h(x_0)(x-x_0) <= 0 \\
            \na h(x_0)*x <= \na h(x_0)*x_0 - h(x_0) \\
            A = \na h(x_0)； b = \na h(x_0)*x_0 - h(x_0) \\
            \end{align*}
        '''

        # g, h = self.raw['g'][0], self.raw['g'][1]
        A = sparse(self.dh).T
        b = self.bi

        return A, b
