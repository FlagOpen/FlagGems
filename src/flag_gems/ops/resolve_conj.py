import torch


def resolve_conj(A: torch.Tensor):
    print("GEMS RESOLVE_CONJ")
    return torch.complex(A.real, A.imag.neg()) if A.is_conj() else A
