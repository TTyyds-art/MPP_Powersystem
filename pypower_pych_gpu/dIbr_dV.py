# Copyright (c) 1996-2015 PSERC. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""Computes partial derivatives of branch currents w.r.t. voltage.
"""
from numpy import asmatrix
import torch
from torch import diag, asarray, arange
from pypower_pych_gpu.torch_utils import sparse, issparse
device = "cuda" if torch.cuda.is_available() else "cpu"


def dIbr_dV(branch, Yf, Yt, V):
    """Computes partial derivatives of branch currents w.r.t. voltage.

    Returns four matrices containing partial derivatives of the complex
    branch currents at "from" and "to" ends of each branch w.r.t voltage
    magnitude and voltage angle respectively (for all buses). If C{Yf} is a
    sparse matrix, the partial derivative matrices will be as well. Optionally
    returns vectors containing the currents themselves. The following
    explains the expressions used to form the matrices::

        If = Yf * V

    Partials of V, Vf & If w.r.t. voltage angles::
        dV/dVa  = j * diag(V)
        dVf/dVa = sparse(range(nl), f, j*V(f)) = j * sparse(range(nl), f, V(f))
        dIf/dVa = Yf * dV/dVa = Yf * j * diag(V)

    Partials of V, Vf & If w.r.t. voltage magnitudes::
        dV/dVm  = diag(V / abs(V))
        dVf/dVm = sparse(range(nl), f, V(f) / abs(V(f))
        dIf/dVm = Yf * dV/dVm = Yf * diag(V / abs(V))

    Derivations for "to" bus are similar.

    @author: Ray Zimmerman (PSERC Cornell)
    """
    i = arange(len(V)).to(device)

    Vnorm = V / abs(V)

    if issparse(Yf):
        diagV = sparse((V, (i, i)))
        diagVnorm = sparse((Vnorm, (i, i)))
    else:
        diagV       = asarray( diag(V) )
        diagVnorm   = asarray( diag(Vnorm) )

    dIf_dVa =  1j * Yf.mm(diagV)
    dIf_dVm = Yf.mm(diagVnorm)
    dIt_dVa = 1j * Yt.mm(diagV)
    dIt_dVm = Yt.mm(diagVnorm)

    # Compute currents.
    if issparse(Yf):
        If = Yf.mm(V.reshape(-1, 1)).squeeze()
        It = Yt.mm(V.reshape(-1, 1)).squeeze()
    else:
        If = asarray( Yf * asarray(V).T ).flatten()
        It = asarray( Yt * asarray(V).T ).flatten()

    return dIf_dVa, dIf_dVm, dIt_dVa, dIt_dVm, If, It
