# Copyright (c) 1996-2015 PSERC. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""Computes 2nd derivatives of |complex current|**2 w.r.t. V.
"""
import torch
from torch import arange
# from cupyx.scipy.sparse import csr_matrix as sparse
from pypower_pych_gpu.torch_utils import sparse
from pypower_pych_gpu.d2Ibr_dV2 import d2Ibr_dV2
device = "cuda" if torch.cuda.is_available() else "cpu"

def d2AIbr_dV2(dIbr_dVa, dIbr_dVm, Ibr, Ybr, V, lam):
    """Computes 2nd derivatives of |complex current|**2 w.r.t. V.

    Returns 4 matrices containing the partial derivatives w.r.t. voltage
    angle and magnitude of the product of a vector C{lam} with the 1st partial
    derivatives of the square of the magnitude of the branch currents.
    Takes sparse first derivative matrices of complex flow, complex flow
    vector, sparse branch admittance matrix C{Ybr}, voltage vector C{V} and
    C{nl x 1} vector of multipliers C{lam}. Output matrices are sparse.

    For more details on the derivations behind the derivative code used
    in PYPOWER information, see:

    [TN2]  R. D. Zimmerman, I{"AC Power Flows, Generalized OPF Costs and
    their Derivatives using Complex Matrix Notation"}, MATPOWER
    Technical Note 2, February 2010.
    U{http://www.pserc.cornell.edu/matpower/TN2-OPF-Derivatives.pdf}

    @see: L{dIbr_dV}.

    @author: Ray Zimmerman (PSERC Cornell)
    """
    # define
    il = arange(len(lam)).to(device)

    diaglam = sparse((lam, (il, il))).to(torch.complex128)
    diagIbr_conj = sparse((Ibr.conj(), (il, il)))

    Iaa, Iav, Iva, Ivv = d2Ibr_dV2(Ybr, V, diagIbr_conj.matmul(lam.to(torch.complex128)))

    Haa = 2 * ( Iaa + dIbr_dVa.t().matmul(diaglam.matmul(dIbr_dVa.conj())) ).to_dense().real
    Hva = 2 * ( Iva + dIbr_dVm.t().matmul(diaglam.matmul(dIbr_dVa.conj()) )).to_dense().real
    Hav = 2 * ( Iav + dIbr_dVa.t().matmul(diaglam.matmul(dIbr_dVm.conj())) ).to_dense().real
    Hvv = 2 * ( Ivv + dIbr_dVm.t().matmul(diaglam.matmul(dIbr_dVm.conj()) )).to_dense().real

    return Haa, Hav, Hva, Hvv


if __name__ == '__main__':
    # use scipy to produce the same result
    from numpy import array, arange
    from numpy import conj as c
    # from scipy.sparse import csr_matrix as sparse
    import torch
    lam = [1, 2, 3, 4, 5, 6]
    il = arange(len(lam)).to(device)
    diaglam = sparse((lam, (il, il)))
    # diaglam = torch.sparse_csr_tensor(il, il, lam, (max(il)+1, max(il)+1))
    print(diaglam)
    # print(diaglam.toarray())
    print(diaglam.to_dense())