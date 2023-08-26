# Copyright (c) 1996-2015 PSERC. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""Evaluates nonlinear constraints and their Jacobian for OPF.
"""
import torch
# from numpy import zeros, ones, conj, exp, r_, Inf, arange
from torch import zeros, ones, conj, exp, tensor, arange, cat
# from cupy import concatenate as r_ # concatenate is similar with np.r_, but the specifics will depend on your use case
Inf = tensor(float('inf'))
# from scipy.sparse import lil_matrix, vstack, hstack, csr_matrix as sparse
device = "cuda" if torch.cuda.is_available() else "cpu"

from pypower_pych_gpu.torch_utils import vstack, hstack
from pypower_pych_gpu.torch_utils import sparse
from pypower_pych_gpu.idx_gen import GEN_BUS, PG, QG
from pypower_pych_gpu.idx_brch import F_BUS, T_BUS, RATE_A

# 所有的函数都需要转换成torch的
from pypower_pych_gpu.makeSbus import makeSbus
from pypower_pych_gpu.dSbus_dV import dSbus_dV
from pypower_pych_gpu.dIbr_dV import dIbr_dV
from pypower_pych_gpu.dSbr_dV import dSbr_dV
from pypower_pych_gpu.dAbr_dV import dAbr_dV


def opf_consfcn(x, pack_para, Ybus, Yf, Yt, ppopt, il=None, *args):
    """Evaluates nonlinear constraints and their Jacobian for OPF.

    Constraint evaluation function for AC optimal power flow, suitable
    for use with L{pips}. Computes constraint vectors and their gradients.

    @param x: optimization vector
    @param om: OPF model object
    @param Ybus: bus admittance matrix
    @param Yf: admittance matrix for "from" end of constrained branches
    @param Yt: admittance matrix for "to" end of constrained branches
    @param ppopt: PYPOWER options vector
    @param il: (optional) vector of branch indices corresponding to
    branches with flow limits (all others are assumed to be
    unconstrained). The default is C{range(nl)} (all branches).
    C{Yf} and C{Yt} contain only the rows corresponding to C{il}.

    @return: C{h} - vector of inequality constraint values (flow limits)
    limit^2 - flow^2, where the flow can be apparent power real power or
    current, depending on value of C{OPF_FLOW_LIM} in C{ppopt} (only for
    constrained lines). C{g} - vector of equality constraint values (power
    balances). C{dh} - (optional) inequality constraint gradients, column
    j is gradient of h(j). C{dg} - (optional) equality constraint gradients.

    @see: L{opf_costfcn}, L{opf_hessfcn}

    @author: Carlos E. Murillo-Sanchez (PSERC Cornell & Universidad
    Autonoma de Manizales)
    @author: Ray Zimmerman (PSERC Cornell)
    """
    ##----- initialize -----

    ## unpack data

    baseMVA, bus, gen, branch = pack_para[1], pack_para[2], pack_para[3], pack_para[4]
    vv = pack_para[7]

    ## problem dimensions
    nb = bus.shape[0]          ## int: number of buses
    nl = branch.shape[0]       ## int: number of branches
    ng = gen.shape[0]          ## int: number of dispatchable injections
    nxyz = len(x)              ## int: total number of control vars of all types

    ## set default constrained lines
    if il is None:
        il = arange(nl).to(device)         ## all lines have limits by default
    nl2 = len(il)              ## int: number of constrained lines
    #
    # grab Pg & Qg
    Pg = x[vv["i1"]["Pg"]:vv["iN"]["Pg"]]  ## active generation in p.u.
    Qg = x[vv["i1"]["Qg"]:vv["iN"]["Qg"]]  ## reactive generation in p.u.

    ## put Pg & Qg back in gen
    gen[:, PG] = Pg * baseMVA  ## active generation in MW
    gen[:, QG] = Qg * baseMVA  ## reactive generation in MVAr

    ## rebuild Sbus
    Sbus = makeSbus(baseMVA, bus, gen) ## net injected power in p.u.

    ## ----- evaluate constraints -----
    ## reconstruct V
    Va = x[vv["i1"]["Va"]:vv["iN"]["Va"]]
    Vm = x[vv["i1"]["Vm"]:vv["iN"]["Vm"]]
    V = Vm * exp(1j * Va)

    ## evaluate power flow equations
    mis = V * (Ybus.matmul(V)).conj() - Sbus.squeeze()
    # print(f"V: {V}; Ybus: {Ybus}; Ybus*V: {Ybus * V}")

    ##----- evaluate constraint function values -----
    ## first, the equality constraints (power flow)
    g = cat([ mis.real,            ## active power mismatch for all buses)
            mis.imag ] )          ## reactive power mismatch for all buses

    ## then, the inequality constraints (branch flow limits)
    if nl2 > 0:
        flow_max = (branch[il, RATE_A] / baseMVA)**2
        flow_max[flow_max == 0] = Inf
        if ppopt['OPF_FLOW_LIM'] == 2:       ## current magnitude limit, |I|
            If = Yf.mm(V.reshape(-1, 1)).squeeze()
            It = Yt.mm(V.reshape(-1, 1)).squeeze()
            h =  cat([ If * If.conj() - flow_max,     ## branch I limits (from bus)
                    It * It.conj() - flow_max ]).real   ## branch I limits (to bus)
        else:
            ## compute branch power flows
            ## complex power injected at "from" bus (p.u.)
            Sf = V[ branch[il, F_BUS].int() ] * (Yf.mm(V.reshape(-1, 1))).conj()
            ## complex power injected at "to" bus (p.u.)
            St = V[ branch[il, T_BUS].int() ] * (Yt.mm(V.reshape(-1, 1))).conj()
            if ppopt['OPF_FLOW_LIM'] == 1:   ## active power limit, P (Pan Wei)
                h = cat([ (Sf.real)**2 - flow_max,   ## branch P limits (from bus)
                        (St.real)**2 - flow_max ])  ## branch P limits (to bus)
            else:                ## apparent power limit, |S|
                h = cat([ Sf * Sf.conj() - flow_max, ## branch S limits (from bus)
                        St * St.conj() - flow_max ]).real  ## branch S limits (to bus)
    else:
        h = zeros((0, 1), dtype=torch.float64,  device=device)

    ##----- evaluate partials of constraints -----
    ## index ranges
    iVa = arange(vv["i1"]["Va"], vv["iN"]["Va"]).to(device)
    iVm = arange(vv["i1"]["Vm"], vv["iN"]["Vm"]).to(device)
    iPg = arange(vv["i1"]["Pg"], vv["iN"]["Pg"]).to(device)
    iQg = arange(vv["i1"]["Qg"], vv["iN"]["Qg"]).to(device)
    iVaVmPgQg = cat([iVa, iVm, iPg, iQg]).T

    ## compute partials of injected bus powers
    dSbus_dVm, dSbus_dVa = dSbus_dV(Ybus, V)           ## w.r.t. V
    ## Pbus w.r.t. Pg, Qbus w.r.t. Qg
    # neg_Cg = sparse((-ones(ng), (gen[:, GEN_BUS], arange(ng))), (nb, ng))
    neg_Cg = torch.sparse_coo_tensor(torch.stack([gen[:, GEN_BUS].int(), arange(ng).to(device)], dim=0), values=-ones(ng, dtype=torch.float64, device=device),  size=(nb, ng)).to_sparse(layout=torch.sparse_csr).to_dense()

    ## construct Jacobian of equality constraints (power flow) and transpose it
    dg = zeros((2 * nb, nxyz), dtype=torch.float64, device=device)
    blank = zeros((nb, ng), dtype=torch.float64, device=device)
    dg[:, iVaVmPgQg] = torch.cat([
            ## P mismatch w.r.t Va, Vm, Pg, Qg
            torch.cat([dSbus_dVa.to_dense().real, dSbus_dVm.to_dense().real, neg_Cg, blank], dim=1),
            ## Q mismatch w.r.t Va, Vm, Pg, Qg
            torch.cat([dSbus_dVa.to_dense().imag, dSbus_dVm.to_dense().imag, blank, neg_Cg], dim=1)
        ], dim=0)
    dg = dg.T

    if nl2 > 0:
        ## compute partials of Flows w.r.t. V
        if ppopt['OPF_FLOW_LIM'] == 2:     ## current
            dFf_dVa, dFf_dVm, dFt_dVa, dFt_dVm, Ff, Ft = \
                    dIbr_dV(branch[il, :], Yf, Yt, V)
        else:                  ## power
            dFf_dVa, dFf_dVm, dFt_dVa, dFt_dVm, Ff, Ft = \
                    dSbr_dV(branch[il, :], Yf, Yt, V)
        if ppopt['OPF_FLOW_LIM'] == 1:     ## real part of flow (active power)
            dFf_dVa = dFf_dVa.real
            dFf_dVm = dFf_dVm.real
            dFt_dVa = dFt_dVa.real
            dFt_dVm = dFt_dVm.real
            Ff = Ff.real
            Ft = Ft.real

        ## squared magnitude of flow (of complex power or current, or real power)
        df_dVa, df_dVm, dt_dVa, dt_dVm = \
                dAbr_dV(dFf_dVa, dFf_dVm, dFt_dVa, dFt_dVm, Ff, Ft)

        ## construct Jacobian of inequality constraints (branch limits)
        ## and transpose it.
        dh = zeros((2 * nl2, nxyz), dtype=torch.float64, device=device)
        dh[:, cat([iVa, iVm]).T] = vstack([
                hstack([df_dVa, df_dVm]),    ## "from" flow limit
                hstack([dt_dVa, dt_dVm])     ## "to" flow limit
            ], "csr")
        dh = dh.T
    else:
        dh = None

    return h, g, dh, dg
