import torch
from .abs import abs
from .add import add
from .addmm import addmm
from .bitwise_not import bitwise_not
from .bmm import bmm
from .cos import cos
from .cumsum import cumsum
from .div import div
from .dropout import native_dropout
from .exp import exp
from .gelu import gelu
from .groupnorm import group_norm
from .isinf import isinf
from .isnan import isnan
from .layernorm import layer_norm
from .mean import mean, mean_dim
from .mm import mm
from .mul import mul
from .neg import neg
from .pow_scalar import pow_scalar
from .pow_tensor_scalar import pow_tensor_scalar
from .pow_tensor_tensor import pow_tensor_tensor
from .reciprocal import reciprocal
from .relu import relu
from .rsqrt import rsqrt
from .sigmoid import sigmoid
from .silu import silu
from .sin import sin
from .softmax import softmax
from .sub import sub
from .tanh import tanh
from .triu import triu
from .var_mean import var_mean
from .vector_norm import vector_norm

from .max import max, max_dim
from .min import min, min_dim
from .amax import amax
from .argmax import argmax
from .prod import prod, prod_dim
from .sum import sum, sum_dim


aten_lib = torch.library.Library("aten", "IMPL")


def enable(lib=aten_lib):
    lib.impl("abs", abs, "PrivateUse1")
    lib.impl("add.Tensor", add, "PrivateUse1")
    lib.impl("addmm", addmm, "PrivateUse1")
    lib.impl("bitwise_not", bitwise_not, "PrivateUse1")
    lib.impl("bmm", bmm, "PrivateUse1")
    lib.impl("cos", cos, "PrivateUse1")
    lib.impl("cumsum", cumsum, "PrivateUse1")
    lib.impl("div.Tensor", div, "PrivateUse1")
    lib.impl("native_dropout", native_dropout, "AutogradPrivateUse1")
    lib.impl("exp", exp, "PrivateUse1")
    lib.impl("gelu", gelu, "PrivateUse1")
    lib.impl("native_group_norm", group_norm, "AutogradPrivateUse1")
    lib.impl("isinf", isinf, "PrivateUse1")
    lib.impl("isnan", isnan, "PrivateUse1")
    lib.impl("native_layer_norm", layer_norm, "AutogradPrivateUse1")
    lib.impl("mean", mean, "PrivateUse1")
    lib.impl("mean.dim", mean_dim, "PrivateUse1")
    lib.impl("mm", mm, "PrivateUse1")
    lib.impl("mul.Tensor", mul, "PrivateUse1")
    lib.impl("neg", neg, "PrivateUse1")
    lib.impl("pow.Scalar", pow_scalar, "PrivateUse1")
    lib.impl("pow.Tensor_Scalar", pow_tensor_scalar, "PrivateUse1")
    lib.impl("pow.Tensor_Tensor", pow_tensor_tensor, "PrivateUse1")
    lib.impl("reciprocal", reciprocal, "PrivateUse1")
    lib.impl("relu", relu, "AutogradPrivateUse1")
    lib.impl("rsqrt", rsqrt, "PrivateUse1")
    lib.impl("sigmoid", sigmoid, "AutogradPrivateUse1")
    lib.impl("silu", silu, "AutogradPrivateUse1")
    lib.impl("sin", sin, "PrivateUse1")
    lib.impl("softmax.int", softmax, "AutogradPrivateUse1")
    lib.impl("sub.Tensor", sub, "PrivateUse1")
    lib.impl("tanh", tanh, "AutogradPrivateUse1")
    lib.impl("triu", triu, "PrivateUse1")
    lib.impl("var_mean.correction", var_mean, "PrivateUse1")
    lib.impl("linalg_vector_norm", vector_norm, "PrivateUse1")

    lib.impl("max", max, "PrivateUse1")
    lib.impl("max.dim", max_dim, "PrivateUse1")
    lib.impl("min", min, "PrivateUse1")
    lib.impl("min.dim", min_dim, "PrivateUse1")
    lib.impl("amax", amax, "PrivateUse1")
    lib.impl("argmax", argmax, "PrivateUse1")
    lib.impl("prod", prod, "PrivateUse1")
    lib.impl("prod.dim_int", prod_dim, "PrivateUse1")
    lib.impl("sum", sum, "PrivateUse1")
    lib.impl("sum.dim_IntList", sum_dim, "PrivateUse1")


class use_gems:
    def __init__(self):
        self.lib = torch.library.Library("aten", "IMPL")

    def __enter__(self):
        enable(self.lib)

    def __exit__(self, exc_type, exc_val, exc_tb):
        del self.lib
