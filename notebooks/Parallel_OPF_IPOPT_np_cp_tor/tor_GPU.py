import warnings
warnings.filterwarnings("ignore")

import time
import sys, os
path_current = '/home/huzuntao/PycharmProjects/MPP_Powersystem/'
path_ = os.getcwd()
if path_current not in sys.path:
    sys.path.insert(1, '/home/huzuntao/PycharmProjects/MPP_Powersystem/')
elif path_ not in sys.path:
    sys.path.insert(1, path_)

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

# from pandapower.pypower.util import sub2ind

from pypower_pych_gpu.opf_hessfcn import opf_hessfcn
from pypower_pych_gpu.pips import pips

import warnings
warnings.filterwarnings("ignore")

# pip.py
import torch
# from numpy import array, Inf, any, isnan, ones, r_, finfo, \
#     zeros, dot, absolute, log, flatnonzero as find
# from numpy.linalg import norm

# from scipy.sparse import vstack, hstack, eye, csr_matrix as sparse
# from scipy.sparse.linalg import spsolve
from torch import tensor, any, isnan, ones, cat, finfo, \
    zeros, dot, absolute, log, arange
from pypower_pych_gpu.torch_utils import find
from torch import asarray
from torch.linalg import norm
Inf = tensor(float('inf'))

from pypower_pych_gpu.torch_utils import vstack, hstack
from torch import eye
from pypower_pych_gpu.torch_utils import sparse, issparse
# from cupyx.scipy.sparse.linalg import spsolve
from torch.linalg import solve

from pandapower.pypower.pipsver import pipsver

EPS = finfo(float).eps
    # Define the device
device = "cuda" if torch.cuda.is_available() else "cpu"
# print("Using {} device".format(device))

import pandapower as pp
net = pp.create_empty_network()

# create buses
bus1 = pp.create_bus(net, vn_kv=110.)
bus2 = pp.create_bus(net, vn_kv=110.)
bus3 = pp.create_bus(net, vn_kv=110.)
bus4 = pp.create_bus(net, vn_kv=110.)
bus5 = pp.create_bus(net, vn_kv=110.)
bus6 = pp.create_bus(net, vn_kv=110.)

# create 110 kV lines
pp.create_line(net, bus4, bus5, length_km=90., std_type='149-AL1/24-ST1A 110.0')
pp.create_line(net, bus3, bus4, length_km=90., std_type='149-AL1/24-ST1A 110.0')
pp.create_line(net, bus2, bus3, length_km=90., std_type='149-AL1/24-ST1A 110.0')
pp.create_line(net, bus1, bus2, length_km=70., std_type='149-AL1/24-ST1A 110.0')
pp.create_line(net, bus6, bus3, length_km=70., std_type='149-AL1/24-ST1A 110.0')

# create loads
p_load_1 = 10
p_load_2 = 30
pp.create_load(net, bus2, p_mw=p_load_1, controllable=False)
pp.create_load(net, bus4, p_mw=p_load_2/2, controllable=False)
pp.create_load(net, bus5, p_mw=p_load_2/2, controllable=False)
pp.create_load(net, bus6, p_mw=p_load_2/2, controllable=False)
# create generators
eg = pp.create_ext_grid(net, bus1, min_p_mw=0, max_p_mw=1000, vm_pu=1.05)
g0 = pp.create_gen(net, bus3, p_mw=80, min_p_mw=0, max_p_mw=80, vm_pu=1.00, controllable=True)

costeg = pp.create_poly_cost(net, 0, 'ext_grid', cp1_eur_per_mw=20)
costgen1 = pp.create_poly_cost(net, 0, 'gen', cp1_eur_per_mw=10)
costgen2 = pp.create_poly_cost(net, 1, 'gen', cp1_eur_per_mw=10)

net.bus["min_vm_pu"] = 0.96
net.bus["max_vm_pu"] = 1.04
net.line["max_loading_percent"] = 100
om, ppopt, raw = pp.runopp(net, delta=1e-16, RETURN_RAW_DER=1)


def solve_opt_problem(id=None):

    if id is None:
        id = 0
    print("id: ", id)
    global bus, A, Ybus, il, Yf_on, Yt_on, H, mm, item, pack_para, f
    x0_init = None
    out_opt = None
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
    max_it = ppopt['PDIPM_MAX_IT']
    max_red = ppopt['SCPDIPM_RED_IT']
    init = ppopt['INIT']
    step_control = (ppopt['OPF_ALG'] == 565)  ## OPF_ALG == 565, PIPS-sc
    if feastol == 0:
        feastol = ppopt['OPF_VIOLATION']
    opt = {'feastol': feastol,
           'gradtol': gradtol,
           'comptol': comptol,
           'costtol': costtol,
           'max_it': max_it,
           'max_red': max_red,
           'step_control': step_control,
           'cost_mult': 1,
           'verbose': verbose}
    ## unpack data TODO: move the data to GPU
    ppc = om.get_ppc()
    baseMVA, bus, gen, branch, gencost = tensor(ppc["baseMVA"], device=device), tensor(ppc["bus"],
                                                                                       device=device), tensor(
        ppc["gen"], device=device), \
        tensor(ppc["branch"], device=device), tensor(ppc["gencost"], device=device)
    vv, _, nn, _ = om.get_idx()
    # vv, nn = tensor(vv), tensor(nn)
    ## problem dimensions
    nb = bus.shape[0]  ## int: number of buses
    nl = branch.shape[0]  ## int: number of branches
    ny = om.getN('var', 'y')  ## int: number of piece-wise linear costs
    ## linear constraints
    A, l, u = om.linear_constraints()
    if isinstance(A, numpy.ndarray):
        A = tensor(A, device=device)
    if isinstance(l, numpy.ndarray):
        l = tensor(l, device=device)
    if isinstance(u, numpy.ndarray):
        u = tensor(u, device=device)
    ## bounds on optimization vars
    x0, xmin, xmax = om.getv()
    x0, xmin, xmax = tensor(x0, device=device), tensor(xmin, device=device), tensor(xmax, device=device)
    ## build admittance matrices
    Ybus, Yf, Yt = makeYbus(baseMVA, bus, branch)
    if ('torch' not in str(type(Ybus))) or ('torch' not in str(type(Yf))) or ('torch' not in str(type(Yt))):
        raise ValueError('Ybus/Yf/Yt must be a torch tensor')
    else:
        Ybus = Ybus.to(device)
        Yf = Yf.to(device)
        Yt = Yt.to(device)
    ## try to select an interior initial point if init is not available from a previous powerflow
    if init != "pf":
        ll, uu = xmin.clone(), xmax.clone()
        ll[xmin == -Inf] = -1e10  ## replace Inf with numerical proxies
        uu[xmax == Inf] = 1e10
        x0 = (ll + uu) / 2
        Varefs = bus[bus[:, BUS_TYPE] == REF, VA] * (pi / 180)
        ## angles set to first reference angle
        x0[vv["i1"]["Va"]:vv["iN"]["Va"]] = Varefs[0]
        if ny > 0:
            pass
    if x0_init is not None:
        x0 = x0_init
    ## find branches with flow limits
    il = find((branch[:, RATE_A] != 0) & (branch[:, RATE_A].real < 1e10))
    Yf_on, Yt_on = (Yf.to_dense()[il, :]).to_sparse(layout=torch.sparse_csr), (Yt.to_dense()[il, :]).to_sparse(
        layout=torch.sparse_csr)
    nl2 = len(il)  ## number of constrained lines
    cp = om.get_cost_params()
    N, Cw, H, dd, rh, kk, mm = \
        cp["N"], cp["Cw"], cp["H"], cp["dd"], cp["rh"], cp["kk"], cp["mm"]
    # print(f"N: {N}, Cw: {Cw}, H: {H}, dd: {dd}, rh: {rh}, kk: {kk}, mm: {mm}")
    # print(f"cp: {cp}")
    cp = [tensor(item).to(device) for item in (N, Cw, H, dd, rh, kk, mm)]
    pack_para = (0, baseMVA, bus, gen, branch, gencost, il, vv, nn, ny, cp)
    if issparse(N) and N.nnz > 0:
        print('N is sparse')
    ##-----  run opf  -----
    f_fcn = lambda x, return_hessian=False: opf_costfcn(x, pack_para, return_hessian)
    gh_fcn = lambda x: opf_consfcn(x, pack_para, Ybus, Yf_on, Yt_on, ppopt, il)
    hess_fcn = lambda x, lmbda, cost_mult: opf_hessfcn(x, lmbda, pack_para, Ybus, Yf_on, Yt_on, ppopt, il, cost_mult)
    if isinstance(f_fcn, dict):  ## problem dict
        p = f_fcn
        f_fcn = p['f_fcn']
        x0 = p['x0']
        if 'opt' in p: opt = p['opt']
        if 'hess_fcn' in p: hess_fcn = p['hess_fcn']
        if 'gh_fcn' in p: gh_fcn = p['gh_fcn']
        if 'xmax' in p: xmax = p['xmax']
        if 'xmin' in p: xmin = p['xmin']
        if 'u' in p: u = p['u']
        if 'l' in p: l = p['l']
        if 'A' in p: A = p['A']
    # print(f"isinstance(f_fcn, dict): {isinstance(f_fcn, dict)}; x0 shape: {x0.shape}")
    nx = x0.shape[0]  # int: number of variables
    nA = A.shape[0] if A is not None else 0  # int: number of original linear constr
    # default argument values
    if l is None or len(l) == 0: l = -Inf * ones(nA, device=device)
    if u is None or len(u) == 0: u = Inf * ones(nA, device=device)
    if xmin is None or len(xmin) == 0: xmin = -Inf * ones(x0.shape[0], dtype=x0.dtype).to(device)
    if xmax is None or len(xmax) == 0: xmax = Inf * ones(x0.shape[0], dtype=x0.dtype).to(device)
    if gh_fcn is None:
        nonlinear = False
        gn = tensor([], device=device)
        hn = tensor([], device=device)
    else:
        nonlinear = True
    if opt is None: opt = {}
    # options
    if "feastol" not in opt:
        opt["feastol"] = 1e-06
    if "gradtol" not in opt:
        opt["gradtol"] = 1e-06
    if "comptol" not in opt:
        opt["comptol"] = 1e-06
    if "costtol" not in opt:
        opt["costtol"] = 1e-06
    if "max_it" not in opt:
        opt["max_it"] = 150
    if "max_red" not in opt:
        opt["max_red"] = 20
    if "step_control" not in opt:
        opt["step_control"] = False
    if "cost_mult" not in opt:
        opt["cost_mult"] = 1
    if "verbose" not in opt:
        opt["verbose"] = 0
    opt["verbose"] = 1
    # initialize history
    hist = []
    # constants
    xi = 0.99995
    sigma = 0.1
    z0 = 1
    alpha_min = 1e-8
    rho_min = 0.95
    rho_max = 1.05
    mu_threshold = 1e-5
    # initialize
    i = 0  # iteration counter
    converged = False  # flag
    eflag = False  # exit flag
    # add var limits to linear constraints
    eyex = eye(nx, nx, dtype=torch.float64, device=device)
    # print(f"A: {A}; eyex shape: {eyex.shape}; nx : {nx}")
    AA = eyex if A is None else vstack([eyex, A], "csr")
    ll = cat([xmin, l])
    uu = cat([xmax, u])
    # split up linear constraints
    ieq = find(absolute(uu - ll) <= EPS)
    igt = find((uu >= 1e10) & (ll > -1e10))
    ilt = find((ll <= -1e10) & (uu < 1e10))
    ibx = find((absolute(uu - ll) > EPS) & (uu < 1e10) & (ll > -1e10))
    # zero-sized sparse matrices unsupported
    # print(f'ieq_: {ieq};AA shape: {AA.shape}')
    Ae = AA[ieq, :] if len(ieq) else None
    if len(ilt) or len(igt) or len(ibx):
        idxs = [(1, ilt), (-1, igt), (1, ibx), (-1, ibx)]
        Ai = vstack([sig * AA[idx, :] for sig, idx in idxs if len(idx)], 'csr')
    else:
        Ai = None
    be = uu[ieq]
    bi = cat([uu[ilt], -ll[igt], uu[ibx], -ll[ibx]])
    x = x0
    f, df = f_fcn(x)  # cost
    f = f * opt["cost_mult"]
    df = df * opt["cost_mult"]
    if nonlinear:
        hn, gn, dhn, dgn = gh_fcn(x)  # nonlinear constraints
        h = hn if Ai is None else cat([hn, Ai.to(torch.float64).matmul(x) - bi])  # inequality constraints
        g = gn if Ae is None else cat([gn, Ae.to(torch.float64).matmul(x) - be])
        if (dhn is None) and (Ai is None):
            dh = None
        elif dhn is None:
            dh = Ai.T
        elif Ai is None:
            dh = dhn
        else:
            dh = hstack([dhn, Ai.T])

        if (dgn is None) and (Ae is None):
            dg = None
        elif dgn is None:
            dg = Ae.T
        elif Ae is None:
            dg = dgn
        else:
            dg = hstack([dgn, Ae.T])
    else:
        h = -bi if Ai is None else Ai * x - bi  # inequality constraints
        g = -be if Ae is None else Ae * x - be  # equality constraints
        dh = None if Ai is None else Ai.T  # 1st derivative of inequalities
        dg = None if Ae is None else Ae.T  # 1st derivative of equalities
    # some dimensions
    neq = g.shape[0]  # int: number of equality constraints
    niq = h.shape[0]  # int: number of inequality constraints
    neqnln = gn.shape[0]  # int: number of nonlinear equality constraints
    niqnln = hn.shape[0]  # int: number of nonlinear inequality constraints
    nlt = len(ilt)  # int: number of upper bounded linear inequalities
    ngt = len(igt)  # int: number of lower bounded linear inequalities
    nbx = len(ibx)  # int: number of doubly bounded linear inequalities
    # initialize gamma, lam, mu, z, e
    gamma = 1  # barrier coefficient
    lam = zeros(neq, dtype=torch.float64, device=device)
    z = z0 * ones(niq, dtype=torch.float64, device=device)
    mu = z0 * ones(niq, dtype=torch.float64, device=device)
    k = find(h < -z0)
    z[k] = -h[k].double()
    k = find((gamma / z) > z0)
    mu[k] = gamma / z[k]
    e = ones(niq, dtype=torch.float64, device=device)
    # check tolerance
    f0 = f
    if opt["step_control"]:
        L = f + dot(lam, g) + dot(mu, h + z) - gamma * sum(log(z))
    Lx = df.clone()
    Lx = Lx + dg.matmul(lam) if dg is not None else Lx
    Lx = Lx + dh.matmul(mu) if dh is not None else Lx
    maxh = zeros(1, device=device) if len(h) == 0 else max(h)
    gnorm = norm(g, Inf) if len(g) else 0.0
    lam_norm = norm(lam, Inf) if len(lam) else 0.0
    mu_norm = norm(mu, Inf) if len(mu) else 0.0
    znorm = norm(z, Inf) if len(z) else 0.0
    feascond = \
        max([gnorm, maxh]) / (1 + max([norm(x, Inf), znorm]))
    gradcond = \
        norm(Lx, Inf) / (1 + max([lam_norm, mu_norm]))
    compcond = dot(z, mu) / (1 + norm(x, Inf))
    costcond = absolute(f - f0) / (1 + absolute(f0))
    # save history
    hist.append({'feascond': feascond, 'gradcond': gradcond,
                 'compcond': compcond, 'costcond': costcond, 'gamma': gamma,
                 'stepsize': 0, 'obj': f / opt["cost_mult"], 'alphap': 0, 'alphad': 0})
    if opt["verbose"]:  # pragma: no cover
        s = '-sc' if opt["step_control"] else ''
        v = pipsver('all')
        print('Python Interior Point Solver - PIPS%s, Version %s, %s' %
              (s, v['Version'], v['Date']))
        if opt['verbose'] > 1:
            print(" it    objective   step size   feascond     gradcond     "
                  "compcond     costcond  ")
            print("----  ------------ --------- ------------ ------------ "
                  "------------ ------------")
            print("%3d  %12.8g %10s %12g %12g %12g %12g" %
                  (i, (f / opt["cost_mult"]), "",
                   feascond, gradcond, compcond, costcond))
    if feascond < opt["feastol"] and gradcond < opt["gradtol"] and \
            compcond < opt["comptol"] and costcond < opt["costtol"]:
        converged = True
        if opt["verbose"]:
            print("Converged!")
    # main loop
    # do Newton iterations
    while (not converged) and (i < opt["max_it"]):
        # update iteration counter
        i += 1
        print(f"max it: {opt['max_it']}; i: {i}")

        # compute update step
        lmbda = {"eqnonlin": lam[range(neqnln)],
                 "ineqnonlin": mu[range(niqnln)]}
        if nonlinear:
            if hess_fcn is None:
                print("pips: Hessian evaluation via finite differences "
                      "not yet implemented.\nPlease provide "
                      "your own hessian evaluation function.")
            Lxx = hess_fcn(x, lmbda, opt["cost_mult"])
        else:
            _, _, d2f = f_fcn(x, True)  # cost
            Lxx = d2f * opt["cost_mult"]
        rz = arange(len(z)).to(device)
        zinvdiag = sparse((1.0 / z, (rz, rz))) if len(z) else None
        rmu = arange(len(mu)).to(device)
        mudiag = sparse((mu, (rmu, rmu))) if len(mu) else None
        dh_zinv = None if dh is None else dh.matmul(zinvdiag)
        M = Lxx if dh is None else Lxx + dh_zinv.matmul(mudiag).matmul(dh.T)
        N = Lx if dh is None else Lx + dh_zinv.matmul((mudiag.matmul(h.to(torch.float64)) + gamma * e))

        Ab = M.to_sparse_csr if dg is None else vstack([
            hstack([M, dg]),
            hstack([dg.T, zeros((neq, neq), dtype=torch.float64, device=device)])
        ])
        bb = cat([-N, -g])

        dxdlam = solve(Ab.to_dense(), bb).double()

        if any(isnan(dxdlam)):
            if opt["verbose"]:
                print('\nNumerically Failed\n')
            eflag = -1
            break

        dx = dxdlam[:nx]
        dlam = dxdlam[nx:nx + neq]
        dz = -h - z if dh is None else -h.to(torch.float64) - z - dh.T.matmul(dx)
        dmu = -mu if dh is None else -mu + zinvdiag.matmul((gamma * e - mudiag.matmul(dz)))

        # do the update
        k = find(dz < 0.0)
        alphap = min([xi * min(z[k] / -dz[k]), 1]) if len(k) else 1.0
        k = find(dmu < 0.0)
        alphad = min([xi * min(mu[k] / -dmu[k]), 1]) if len(k) else 1.0
        x = x + alphap * dx
        z = z + alphap * dz
        lam = lam + alphad * dlam
        mu = mu + alphad * dmu
        if niq > 0:
            gamma = sigma * dot(z, mu) / niq

        # evaluate cost, constraints, derivatives
        f, df = f_fcn(x)  # cost
        f = f * opt["cost_mult"]
        df = df * opt["cost_mult"]
        if nonlinear:
            hn, gn, dhn, dgn = gh_fcn(x)  # nln constraints
            #            g = gn if Ai is None else r_[gn, Ai * x - bi] # ieq constraints
            #            h = hn if Ae is None else r_[hn, Ae * x - be] # eq constraints
            h = hn if Ai is None else cat([hn, Ai.to(torch.float64).matmul(x) - bi])  # inequality constraints
            g = gn if Ae is None else cat([gn, Ae.to(torch.float64).matmul(x) - be])
            # h = hn if Ai is None else cat([hn.reshape(len(hn),), Ai * x - bi]) # ieq constr
            # g = gn if Ae is None else cat([gn, Ae * x - be])  # eq constr

            if (dhn is None) and (Ai is None):
                dh = None
            elif dhn is None:
                dh = Ai.T
            elif Ai is None:
                dh = dhn
            else:
                dh = hstack([dhn, Ai.T])

            if (dgn is None) and (Ae is None):
                dg = None
            elif dgn is None:
                dg = Ae.T
            elif Ae is None:
                dg = dgn
            else:
                dg = hstack([dgn, Ae.T])
        else:
            h = -bi if Ai is None else Ai.matmul(x) - bi  # inequality constraints
            g = -be if Ae is None else Ae.matmul(x) - be  # equality constraints
            # 1st derivatives are constant, still dh = Ai.T, dg = Ae.T

        Lx = df
        Lx = Lx + dg.matmul(lam) if dg is not None else Lx
        Lx = Lx + dh.matmul(mu) if dh is not None else Lx

        if len(h) == 0:
            maxh = zeros(1)
        else:
            maxh = max(h)

        gnorm = norm(g, Inf) if len(g) else 0.0
        lam_norm = norm(lam, Inf) if len(lam) else 0.0
        mu_norm = norm(mu, Inf) if len(mu) else 0.0
        znorm = norm(z, Inf) if len(z) else 0.0
        feascond = \
            max([gnorm, maxh]) / (1 + max([norm(x, Inf), znorm]))
        gradcond = \
            norm(Lx, Inf) / (1 + max([lam_norm, mu_norm]))
        compcond = dot(z, mu) / (1 + norm(x, Inf))
        costcond = float(absolute(f - f0) / (1 + absolute(f0)))

        hist.append({'feascond': feascond, 'gradcond': gradcond,
                     'compcond': compcond, 'costcond': costcond, 'gamma': gamma,
                     'stepsize': norm(dx), 'obj': f / opt["cost_mult"],
                     'alphap': alphap, 'alphad': alphad})

        if opt["verbose"] > 1:
            print("%3d  %12.8g %10.5g %12g %12g %12g %12g" %
                  (i, (f / opt["cost_mult"]), norm(dx), feascond, gradcond,
                   compcond, costcond))

        if feascond < opt["feastol"] and gradcond < opt["gradtol"] and \
                compcond < opt["comptol"] and costcond < opt["costtol"]:
            converged = True
            if opt["verbose"]:
                print("Converged!")
        else:  # TODO alpha and alphad are too small
            if any(isnan(x)) or (alphap < alpha_min) or \
                    (alphad < alpha_min) or (gamma < EPS) or (gamma > 1.0 / EPS):
                if opt["verbose"]:
                    print("Numerically failed.")
                eflag = -1
                break
            f0 = f
            if opt["step_control"]:
                L = f + dot(lam, g) + dot(mu, (h + z)) - gamma * sum(log(z))


# solve_opt_problem()

import torch
import torch.multiprocessing as mp


if __name__ == '__main__':
    num_processes = 5 # Specify the number of processes you want to run
    torch.cuda.init()
    mp.set_start_method('spawn')
    # Create a list of processes
    processes = []
    start_time =time.time()
    for i in range(num_processes):
        p = mp.Process(target=solve_opt_problem, args=(i,))
        processes.append(p)

    # Start the processes
    for p in processes:
        p.start()

    # Wait for all processes to finish
    for p in processes:
        p.join()

    print("--- %s seconds ---" % (time.time() - start_time))
    print("All optimization problems solved.")
