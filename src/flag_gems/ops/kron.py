import triton.language as tl
import torch
from .mul import mul

def kron(A, B):
    maxdim = max(A.dim(), B.dim())
    pad_A = maxdim - A.dim()
    pad_B = maxdim - B.dim()
    
    a_reshape = [1] * (2 * maxdim)
    b_reshape = [1] * (2 * maxdim)
    c_reshape = [1] * maxdim
    
    for i in range(maxdim):
        if i >= pad_A:
            a_reshape[2 * i] = A.size(i - pad_A)
        if i >= pad_B:
            b_reshape[2 * i + 1] = B.size(i - pad_B)
        
        c_reshape[i] = a_reshape[2 * i] * b_reshape[2 * i + 1]
    
    A_view = A.view(a_reshape)
    B_view = B.view(b_reshape)
    
    return  mul(A_view, B_view).view(c_reshape)
  