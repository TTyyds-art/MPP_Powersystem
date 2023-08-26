import cupy as cp
from cupyx.scipy.sparse import csr_matrix

def csr_real(csr):
    result = csr.copy()
    result.data = cp.real(csr.data)
    return result

def cr_real(csr):

    csr.data = cp.real(csr.data)
    return csr

def csr_imag(csr):
    result = csr.copy()
    result.data = cp.imag(csr.data)
    return result

def cr_imag(csr):

    csr.data = cp.imag(csr.data)
    return csr
