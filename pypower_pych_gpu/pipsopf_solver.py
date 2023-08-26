# -*- coding: utf-8 -*-

# Copyright 1996-2015 PSERC. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# Copyright (c) 2016-2022 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


"""Solves AC optimal power flow using PIPS.
"""
import numpy, torch
from torch import ones, zeros, pi, exp, conj, cat, tensor
Inf = tensor(float('inf'))
from pypower_pych_gpu.torch_utils import find
from pypower_pych_gpu.csr_real_imag import csr_imag
from pypower_pych_gpu.idx_brch import F_BUS, T_BUS, RATE_A, PF, QF, PT, QT, MU_SF, MU_ST
from pypower_pych_gpu.idx_bus import BUS_TYPE, REF, VM, VA, MU_VMAX, MU_VMIN, LAM_P, LAM_Q
from pypower_pych_gpu.idx_cost import MODEL, PW_LINEAR, NCOST
from pypower_pych_gpu.idx_gen import GEN_BUS, PG, QG, VG, MU_PMAX, MU_PMIN, MU_QMAX, MU_QMIN
from pypower_pych_gpu.makeYbus import makeYbus
from pypower_pych_gpu.opf_consfcn import opf_consfcn
from pypower_pych_gpu.opf_costfcn import opf_costfcn

device = "cuda" if torch.cuda.is_available() else "cpu"

from pypower_pych_gpu.opf_hessfcn import opf_hessfcn
from pypower_pych_gpu.pips import pips

import warnings
warnings.filterwarnings("ignore")



def pipsopf_solver_torch(om, ppopt, x0_init=None, out_opt=None):
    """Solves AC optimal power flow using PIPS in GPUs only.

    Inputs are an OPF model object, a PYPOWER options vector and
    a dict containing keys (can be empty) for each of the desired
    optional output fields.

    outputs are a C{results} dict, C{success} flag and C{raw} output dict.

    C{results} is a PYPOWER case dict (ppc) with the usual baseMVA, bus
    branch, gen, gencost fields, along with the following additional
    fields:
        - C{order}      see 'help ext2int' for details of this field
        - C{x}          final value of optimization variables (internal order)
        - C{f}          final objective function value
        - C{mu}         shadow prices on ...
            - C{var}
                - C{l}  lower bounds on variables
                - C{u}  upper bounds on variables
            - C{nln}
                - C{l}  lower bounds on nonlinear constraints
                - C{u}  upper bounds on nonlinear constraints
            - C{lin}
                - C{l}  lower bounds on linear constraints
                - C{u}  upper bounds on linear constraints

    C{success} is C{True} if solver converged successfully, C{False} otherwise

    C{raw} is a raw output dict in form returned by MINOS
        - xr     final value of optimization variables
        - pimul  constraint multipliers
        - info   solver specific termination code
        - output solver specific output information

    @see: L{opf}, L{pips}

    @author: Ray Zimmerman (PSERC Cornell)
    @author: Carlos E. Murillo-Sanchez (PSERC Cornell & Universidad
    Autonoma de Manizales)
    @author: Richard Lincoln
    """
    ##----- initialization -----
    ## optional output
    if out_opt is None:
        out_opt = {}

    ## options
    verbose = ppopt['VERBOSE']
    feastol = ppopt['PDIPM_FEASTOL']
    gradtol = ppopt['PDIPM_GRADTOL']
    comptol = ppopt['PDIPM_COMPTOL']
    costtol = ppopt['PDIPM_COSTTOL']
    max_it  = ppopt['PDIPM_MAX_IT']
    max_red = ppopt['SCPDIPM_RED_IT']
    init = ppopt['INIT']
    step_control = (ppopt['OPF_ALG'] == 565)  ## OPF_ALG == 565, PIPS-sc
    if feastol == 0:
        feastol = ppopt['OPF_VIOLATION']
    opt = {  'feastol': feastol,
             'gradtol': gradtol,
             'comptol': comptol,
             'costtol': costtol,
             'max_it': max_it,
             'max_red': max_red,
             'step_control': step_control,
             'cost_mult': 1,
             'verbose': verbose  }

    ## unpack data
    ppc = om.get_ppc()
    baseMVA, bus, gen, branch, gencost = \
        tensor(ppc["baseMVA"]).to(device), tensor(ppc["bus"]).to(device), tensor(ppc["gen"]).to(device), \
                            tensor(ppc["branch"]).to(device), tensor(ppc["gencost"]).to(device)
    vv, _, nn, _ = om.get_idx() # dict
    # vv, nn = tensor(vv), tensor(nn)
    ## problem dimensions
    nb = bus.shape[0]          ## int: number of buses
    nl = branch.shape[0]       ## int: number of branches
    ny = om.getN('var', 'y')   ## int: number of piece-wise linear costs

    ## linear constraints
    A, l, u = om.linear_constraints()
    if isinstance(A, numpy.ndarray):
        A = tensor(A).to(device)
    if isinstance(l, numpy.ndarray):
        l = tensor(l).to(device)
    if isinstance(u, numpy.ndarray):
        u = tensor(u).to(device)


    ## bounds on optimization vars
    x0, xmin, xmax = om.getv()
    x0, xmin, xmax = tensor(x0).to(device), tensor(xmin).to(device), tensor(xmax).to(device)

    ## build admittance matrices
    Ybus, Yf, Yt = makeYbus(baseMVA, bus, branch)
    if 'torch' not in str(type(Ybus)):
        raise ValueError('Ybus must be a torch array')
    if 'torch' not in str(type(Yf)):
        raise ValueError('Yf must be a torch array')
    if 'torch' not in str(type(Yt)):
        raise ValueError('Yt must be a torch array')

    ## try to select an interior initial point if init is not available from a previous powerflow
    if init != "pf":
        ll, uu = xmin.clone(), xmax.clone()
        ll[xmin == -Inf] = -1e10   ## replace Inf with numerical proxies
        uu[xmax ==  Inf] =  1e10
        x0 = (ll + uu) / 2
        Varefs = bus[bus[:, BUS_TYPE] == REF, VA] * (pi / 180)
        ## angles set to first reference angle
        x0[vv["i1"]["Va"]:vv["iN"]["Va"]] = Varefs[0]
        if ny > 0:
    #         ipwl = find(gencost[:, MODEL] == PW_LINEAR)
    # #         PQ = r_[gen[:, PMAX], gen[:, QMAX]]
    # #         c = totcost(gencost[ipwl, :], PQ[ipwl])
    #         c = gencost.flatten('F')[sub2ind(gencost.shape, ipwl, NCOST+2*gencost[ipwl, NCOST])]    ## largest y-value in CCV data
    #         x0[vv["i1"]["y"]:vv["iN"]["y"]] = max(c) + 0.1 * abs(max(c))
    # #        x0[vv["i1"]["y"]:vv["iN"]["y"]] = c + 0.1 * abs(c)
            pass

    if x0_init is not None:
        x0 = x0_init

    ## find branches with flow limits
    il = find((branch[:, RATE_A] != 0) & (branch[:, RATE_A].real < 1e10))
    Yf_on, Yt_on = (Yf.to_dense()[il, :]).to_sparse(layout=torch.sparse_csr), (Yt.to_dense()[il, :]).to_sparse(layout=torch.sparse_csr)
    nl2 = len(il)           ## number of constrained lines


    cp = om.get_cost_params()
    N, Cw, H, dd, rh, kk, mm = \
        cp["N"], cp["Cw"], cp["H"], cp["dd"], cp["rh"], cp["kk"], cp["mm"]
    cp = (N, Cw, H, dd, rh, kk, mm)
    pack_para = (ppc, baseMVA, bus, gen, branch, gencost, il, vv, nn, ny, cp)

    ##-----  run opf  -----
    f_fcn = lambda x, return_hessian=False: opf_costfcn(x, pack_para, return_hessian)
    gh_fcn = lambda x: opf_consfcn(x, pack_para, Ybus, Yf_on, Yt_on, ppopt, il)
    hess_fcn = lambda x, lmbda, cost_mult: opf_hessfcn(x, lmbda, pack_para, Ybus, Yf_on, Yt_on, ppopt, il, cost_mult)

    solution = pips(f_fcn, x0, A, l, u, xmin, xmax, gh_fcn, hess_fcn, opt)
    x, f, info, lmbda, output = solution["x"], solution["f"], \
            solution["eflag"], solution["lmbda"], solution["output"]

    success = (info > 0)

    ## update solution data
    Va = x[vv["i1"]["Va"]:vv["iN"]["Va"]]
    Vm = x[vv["i1"]["Vm"]:vv["iN"]["Vm"]]
    Pg = x[vv["i1"]["Pg"]:vv["iN"]["Pg"]]
    Qg = x[vv["i1"]["Qg"]:vv["iN"]["Qg"]]

    V = Vm * exp(1j * Va)

    ##-----  calculate return values  -----
    ## update voltages & generator outputs
    bus[:, VA] = Va * 180 / pi
    bus[:, VM] = Vm
    gen[:, PG] = Pg * baseMVA
    gen[:, QG] = Qg * baseMVA
    gen[:, VG] = Vm[ gen[:, GEN_BUS].int() ]

    ## compute branch flows
    Sf = V[ branch[:, F_BUS].int() ] * (Yf.matmul(V)).conj()  ## cplx pwr at "from" bus, p["u"].
    St = V[ branch[:, T_BUS].int() ] * (Yt.matmul(V)).conj()   ## cplx pwr at "to" bus, p["u"].
    branch[:, PF] = Sf.real * baseMVA
    branch[:, QF] = Sf.imag * baseMVA
    branch[:, PT] = St.real * baseMVA
    branch[:, QT] = St.imag * baseMVA

    ## line constraint is actually on square of limit
    ## so we must fix multipliers
    muSf = zeros(nl, dtype=torch.complex128, device=device)
    muSt = zeros(nl, dtype=torch.complex128, device=device)
    if len(il) > 0:
        muSf[il] = \
            2 * lmbda["ineqnonlin"][:nl2] * branch[il, RATE_A] / baseMVA
        muSt[il] = \
            2 * lmbda["ineqnonlin"][nl2:nl2+nl2] * branch[il, RATE_A] / baseMVA

    ## update Lagrange multipliers
    bus[:, MU_VMAX]  = lmbda["upper"][vv["i1"]["Vm"]:vv["iN"]["Vm"]]
    bus[:, MU_VMIN]  = lmbda["lower"][vv["i1"]["Vm"]:vv["iN"]["Vm"]]
    gen[:, MU_PMAX]  = lmbda["upper"][vv["i1"]["Pg"]:vv["iN"]["Pg"]] / baseMVA
    gen[:, MU_PMIN]  = lmbda["lower"][vv["i1"]["Pg"]:vv["iN"]["Pg"]] / baseMVA
    gen[:, MU_QMAX]  = lmbda["upper"][vv["i1"]["Qg"]:vv["iN"]["Qg"]] / baseMVA
    gen[:, MU_QMIN]  = lmbda["lower"][vv["i1"]["Qg"]:vv["iN"]["Qg"]] / baseMVA

    bus[:, LAM_P] = \
        lmbda["eqnonlin"][nn["i1"]["Pmis"]:nn["iN"]["Pmis"]] / baseMVA
    bus[:, LAM_Q] = \
        lmbda["eqnonlin"][nn["i1"]["Qmis"]:nn["iN"]["Qmis"]] / baseMVA
    branch[:, MU_SF] = muSf / baseMVA
    branch[:, MU_ST] = muSt / baseMVA

    ## package up results
    nlnN = om.getN('nln')

    ## extract multipliers for nonlinear constraints
    kl = find(lmbda["eqnonlin"] < 0)
    ku = find(lmbda["eqnonlin"] > 0)
    nl_mu_l = zeros(nlnN, dtype=torch.float64, device=device)
    nl_mu_u = cat([zeros(2*nb, dtype=torch.float64, device=device), muSf, muSt])
    nl_mu_l[kl] = -lmbda["eqnonlin"][kl]
    nl_mu_u[ku] =  lmbda["eqnonlin"][ku].to(torch.complex128)

    mu = {
      'var': {'l': lmbda["lower"], 'u': lmbda["upper"]},
      'nln': {'l': nl_mu_l, 'u': nl_mu_u},
      'lin': {'l': lmbda["mu_l"], 'u': lmbda["mu_u"]} }

    results = ppc
    results["bus"], results["branch"], results["gen"], \
        results["om"], results["x"], results["mu"], results["f"] = \
            bus, branch, gen, om, x, mu, f

    pimul = cat([
        results["mu"]["nln"]["l"] - results["mu"]["nln"]["u"],
        results["mu"]["lin"]["l"] - results["mu"]["lin"]["u"],
        -ones(int(ny > 0), dtype=torch.float64, device=device),
        results["mu"]["var"]["l"] - results["mu"]["var"]["u"],
    ])
    raw = {'xr': x, 'pimul': pimul, 'info': info, 'output': output}

    return results, success, raw
