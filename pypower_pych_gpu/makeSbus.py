# Copyright (c) 1996-2015 PSERC. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""Builds the vector of complex bus power injections.
"""

import torch
from torch import ones, arange
from pypower_pych_gpu.torch_utils import sparse
from pypower_pych_gpu.torch_utils import find

from pypower_pych_gpu.idx_bus import PD, QD
from pypower_pych_gpu.idx_gen import GEN_BUS, PG, QG, GEN_STATUS
device = "cuda" if torch.cuda.is_available() else "cpu"


def makeSbus(baseMVA, bus, gen):
    """Builds the vector of complex bus power injections.

    Returns the vector of complex bus power injections, that is, generation
    minus load. Power is expressed in per unit.

    @see: L{makeYbus}

    @author: Ray Zimmerman (PSERC Cornell)
    """
    ## generator info
    on = find(gen[:, GEN_STATUS] > 0)      ## which generators are on?
    gbus = gen[on, GEN_BUS].int()                  ## what buses are they at?

    ## form net complex bus power injection vector
    nb = bus.shape[0]
    ngon = on.shape[0]
    ## connection matrix, element i, j is 1 if gen on(j) at bus i is ON
    Cg = torch.sparse_coo_tensor(torch.stack([gbus, arange(ngon).to(device)],dim=0), values=ones(ngon, dtype=torch.float64, device=device), size=(nb, ngon)).to_sparse(layout=torch.sparse_csr)

    ## power injected by gens plus power injected by loads converted to p.u.
    Sbus = ( Cg.to(torch.complex128).matmul((gen[on, PG] + 1j * gen[on, QG]).reshape(-1,1)) -
             (bus[:, PD] + 1j * bus[:, QD]).reshape(-1,1) ) / baseMVA

    return Sbus.squeeze()
