# Copyright (c) 1996-2015 PSERC. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""Builds the bus admittance matrix and branch admittance matrices.
"""

from sys import stderr
import torch
from torch import ones, conj, nonzero, any, exp, pi, cat, tensor, arange
from pypower_pych_gpu.torch_utils import sparse as csr_matrix

from pypower_pych_gpu.idx_bus import BUS_I, GS, BS
from pypower_pych_gpu.idx_brch import F_BUS, T_BUS, BR_R, BR_X, BR_B, BR_STATUS, SHIFT, TAP
device = "cuda" if torch.cuda.is_available() else "cpu"

def makeYbus(baseMVA, bus, branch):
    """Builds the bus admittance matrix and branch admittance matrices.

    Returns the full bus admittance matrix (i.e. for all buses) and the
    matrices C{Yf} and C{Yt} which, when multiplied by a complex voltage
    vector, yield the vector currents injected into each line from the
    "from" and "to" buses respectively of each line. Does appropriate
    conversions to p.u.

    @see: L{makeSbus}

    @author: Ray Zimmerman (PSERC Cornell)
    """
    ## constants
    nb = bus.shape[0]  ## number of buses
    nl = branch.shape[0]  ## number of lines

    ## check that bus numbers are equal to indices to bus (one set of bus nums)
    if any(bus[:, BUS_I] != arange(nb).to(device=bus.device)):
        stderr.write('buses must appear in order by bus number\n')

    ## for each branch, compute the elements of the branch admittance matrix where
    ##
    ##      | If |   | Yff  Yft |   | Vf |
    ##      |    | = |          | * |    |
    ##      | It |   | Ytf  Ytt |   | Vt |
    ##
    stat = branch[:, BR_STATUS]  ## ones at in-service branches
    Ys = stat / (branch[:, BR_R] + 1j * branch[:, BR_X])  ## series admittance
    Bc = stat * branch[:, BR_B]  ## line charging susceptance
    tap = ones(nl, dtype=torch.float64, device=bus.device)  ## default tap ratio = 1
    i = nonzero(branch[:, TAP])  ## indices of non-zero tap ratios
    tap[i] = branch[i, TAP].real ## assign non-zero tap ratios
    tap = tap * exp(1j * pi / 180 * branch[:, SHIFT])  ## add phase shifters

    Ytt = Ys + 1j * Bc / 2
    Yff = Ytt / (tap * conj(tap))
    Yft = - Ys / conj(tap)
    Ytf = - Ys / tap

    ## compute shunt admittance
    ## if Psh is the real power consumed by the shunt at V = 1.0 p.u.
    ## and Qsh is the reactive power injected by the shunt at V = 1.0 p.u.
    ## then Psh - j Qsh = V * conj(Ysh * V) = conj(Ysh) = Gs - j Bs,
    ## i.e. Ysh = Psh + j Qsh, so ...
    ## vector of shunt admittances
    Ysh = (bus[:, GS] + 1j * bus[:, BS]) / baseMVA

    ## build connection matrices
    f = branch[:, F_BUS].real.int()  ## list of "from" buses
    t = branch[:, T_BUS].real.int() ## list of "to" buses
    ## connection matrix for line & from buses
    Cf = torch.sparse_coo_tensor(torch.stack([arange(nl).to(device=bus.device), f], dim=0), values=ones(nl, dtype=torch.float64, device=bus.device), size=(nl, nb)).to_sparse(layout=torch.sparse_csr)
    ## connection matrix for line & to buses
    Ct = torch.sparse_coo_tensor(torch.stack([arange(nl).to(device=bus.device), t], dim=0), values=ones(nl, dtype=torch.float64, device=bus.device), size=(nl, nb)).to_sparse(layout=torch.sparse_csr)

    ## build Yf and Yt such that Yf * V is the vector of complex branch currents injected
    ## at each branch's "from" bus, and Yt is the same for the "to" bus end
    i = cat([arange(nl).to(device=bus.device), arange(nl).to(device=bus.device)])  ## double set of row indices

    Yf = torch.sparse_coo_tensor(torch.stack([i, cat((f,t))], dim=0), values=cat([Yff, Yft]), size=(nl, nb)).to_sparse(layout=torch.sparse_csr)
    Yt = torch.sparse_coo_tensor(torch.stack([i, cat((f,t))], dim=0), values=cat([Ytf, Ytt]), size=(nl, nb)).to_sparse(layout=torch.sparse_csr)
    # Yf = spdiags(Yff, 0, nl, nl) * Cf + spdiags(Yft, 0, nl, nl) * Ct
    # Yt = spdiags(Ytf, 0, nl, nl) * Cf + spdiags(Ytt, 0, nl, nl) * Ct

    ## build Ybus
    Ybus = Cf.transpose(0, 1).to(torch.complex128).mm(Yf) + Ct.transpose(0, 1).to(torch.complex128).mm(Yt) + \
           torch.sparse_coo_tensor( torch.stack([arange(nb).to(device=bus.device), arange(nb).to(device=bus.device)]), values=Ysh, size=(nb, nb)).to_sparse_csr()

    return Ybus, Yf, Yt
