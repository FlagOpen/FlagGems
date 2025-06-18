import torch

from flag_gems.ops.neg import neg_func


def resolve_neg(A: torch.Tensor):
    print("GEMS RESOLVE_NEG")
    return neg_func(A) if A.is_neg() else A
