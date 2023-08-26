# -*- coding: utf-8 -*-

# Copyright 1996-2015 PSERC. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# Copyright (c) 2016-2022 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.



"""Computes partial derivatives of power injection w.r.t. voltage.
"""

import torch
from torch import conj, diag, asarray, arange
from pypower_pych_gpu.torch_utils import sparse, issparse
device = "cuda" if torch.cuda.is_available() else "cpu"

def dSbus_dV(Ybus, V):
    """Computes partial derivatives of power injection w.r.t. voltage.
    """

    if issparse(Ybus):
        return dSbus_dV_sparse(Ybus, V)
    else:
        return dSbus_dV_dense(Ybus, V)


def dSbus_dV_sparse(Ybus, V):
    Ibus = Ybus.mm(V.reshape(-1, 1))
    ib = list(range(len(V)))
    diagV = torch.sparse_coo_tensor([ib, ib], values=V).to_sparse(layout=torch.sparse_csr)
    diagIbus = torch.sparse_coo_tensor( [ib, ib], values=Ibus.flatten()).to_sparse(layout=torch.sparse_csr)
    diagVnorm = torch.sparse_coo_tensor( [ib, ib], values=V / abs(V)).to_sparse(layout=torch.sparse_csr)
    dS_dVm = diagV.mm((Ybus.mm(diagVnorm)).conj()) + diagIbus.conj().mm(diagVnorm)
    dS_dVa = 1j * diagV.mm( (diagIbus.to_dense() - Ybus.mm(diagV).to_dense()).conj()).to_sparse(layout=torch.sparse_csr)
    return dS_dVm, dS_dVa


def dSbus_dV_dense(Ybus, V):
    # standard code from Pypower (slower than above)
    Ibus = Ybus * asarray(V).T

    diagV = asarray(diag(V))
    diagIbus = asarray(diag(asarray(Ibus).flatten()))
    diagVnorm = asarray(diag(V / abs(V)))

    dS_dVm = diagV * (Ybus * diagVnorm).conj() + diagIbus.conj() * diagVnorm
    dS_dVa = 1j * diagV * (diagIbus - Ybus * diagV).conj()
    return dS_dVm, dS_dVa
