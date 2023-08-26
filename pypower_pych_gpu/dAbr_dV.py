# Copyright (c) 1996-2015 PSERC. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""Partial derivatives of squared flow magnitudes w.r.t voltage.
"""
import torch
from pypower_pych_gpu.torch_utils import sparse as csr_matrix
from torch import arange

from pypower_pych_gpu.csr_real_imag import csr_imag, csr_real
device = "cuda" if torch.cuda.is_available() else "cpu"


def dAbr_dV(dSf_dVa, dSf_dVm, dSt_dVa, dSt_dVm, Sf, St):
    """Partial derivatives of squared flow magnitudes w.r.t voltage.

    Returns four matrices containing partial derivatives of the square of
    the branch flow magnitudes at "from" & "to" ends of each branch w.r.t
    voltage magnitude and voltage angle respectively (for all buses), given
    the flows and flow sensitivities. Flows could be complex current or
    complex or real power. Notation below is based on complex power. The
    following explains the expressions used to form the matrices:

    Let Af refer to the square of the apparent power at the "from" end of
    each branch::

        Af = abs(Sf)**2
           = Sf .* conj(Sf)
           = Pf**2 + Qf**2

    then ...

    Partial w.r.t real power::
        dAf/dPf = 2 * diag(Pf)

    Partial w.r.t reactive power::
        dAf/dQf = 2 * diag(Qf)

    Partial w.r.t Vm & Va::
        dAf/dVm = dAf/dPf * dPf/dVm + dAf/dQf * dQf/dVm
        dAf/dVa = dAf/dPf * dPf/dVa + dAf/dQf * dQf/dVa

    Derivations for "to" bus are similar.

    For more details on the derivations behind the derivative code used
    in PYPOWER information, see:

    [TN2]  R. D. Zimmerman, I{"AC Power Flows, Generalized OPF Costs and
    their Derivatives using Complex Matrix Notation"}, MATPOWER
    Technical Note 2, February 2010.
    U{http://www.pserc.cornell.edu/matpower/TN2-OPF-Derivatives.pdf}

    @return: The partial derivatives of the squared flow magnitudes w.r.t
             voltage magnitude and voltage angle given the flows and flow
             sensitivities. Flows could be complex current or complex or
             real power.
    @see: L{dIbr_dV}, L{dSbr_dV}

    @author: Ray Zimmerman (PSERC Cornell)
    """
    il = arange(len(Sf)).to(device)

    dAf_dPf = csr_matrix((2 * Sf.real, (il, il)))
    dAf_dQf = csr_matrix((2 * Sf.imag, (il, il)))
    dAt_dPt = csr_matrix((2 * St.real, (il, il)))
    dAt_dQt = csr_matrix((2 * St.imag, (il, il)))

    # Partial derivative of apparent power magnitude w.r.t voltage
    # phase angle.
    dAf_dVa = dAf_dPf.mm(dSf_dVa.to_dense().real) + dAf_dQf.mm(dSf_dVa.to_dense().imag)
    dAt_dVa = dAt_dPt.mm(dSt_dVa.to_dense().real) + dAt_dQt.mm(dSt_dVa.to_dense().imag)
    # Partial derivative of apparent power magnitude w.r.t. voltage
    # amplitude.
    dAf_dVm = dAf_dPf.mm(dSf_dVm.to_dense().real) + dAf_dQf.mm(dSf_dVm.to_dense().imag)
    dAt_dVm = dAt_dPt.mm(dSt_dVm.to_dense().real) + dAt_dQt.mm(dSt_dVm.to_dense().imag)

    return dAf_dVa, dAf_dVm, dAt_dVa, dAt_dVm
