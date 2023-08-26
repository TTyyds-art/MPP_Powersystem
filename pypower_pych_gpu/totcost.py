# Copyright (c) 1996-2015 PSERC. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""Computes total cost for generators at given output level.
"""
import torch
from torch import zeros, arange
from pypower_pych_gpu.torch_utils import find

from pypower_pych_gpu.polycost import polycost
from pypower_pych_gpu.idx_cost import PW_LINEAR, POLYNOMIAL, COST, NCOST, MODEL
device = "cuda" if torch.cuda.is_available() else "cpu"

def totcost(gencost, Pg):
    """Computes total cost for generators at given output level.

    Computes total cost for generators given a matrix in gencost format and
    a column vector or matrix of generation levels. The return value has the
    same dimensions as PG. Each row of C{gencost} is used to evaluate the
    cost at the points specified in the corresponding row of C{Pg}.

    @author: Ray Zimmerman (PSERC Cornell)
    @author: Carlos E. Murillo-Sanchez (PSERC Cornell & Universidad
    Autonoma de Manizales)
    """
    ng, m = gencost.shape
    totalcost = zeros(ng, dtype=torch.float64, device=Pg.device)

    if len(gencost) > 0:
        ipwl = find(gencost[:, MODEL] == PW_LINEAR)
        ipol = find(gencost[:, MODEL] == POLYNOMIAL)
        if len(ipwl) > 0:
            p = gencost[:, COST:(m-1):2]
            c = gencost[:, (COST+1):m:2]

            for i in ipwl:
                ncost = gencost[i, NCOST]
                for k in arange(ncost - 1, dtype=int).to(device):
                    p1, p2 = p[i, k], p[i, k+1]
                    c1, c2 = c[i, k], c[i, k+1]
                    m = (c2 - c1) / (p2 - p1)
                    b = c1 - m * p1
                    Pgen = Pg[i]
                    if Pgen < p2:
                        totalcost[i] = m * Pgen + b
                        break
                    totalcost[i] = m * Pgen + b

        if len(ipol) > 0:
            totalcost[ipol] = polycost(gencost[ipol, :], Pg[ipol])

    return totalcost
