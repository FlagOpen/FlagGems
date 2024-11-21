import torch

from . import testing  # noqa: F401
from . import runtime
from .fused import *  # noqa: F403
from .ops import *  # noqa: F403
from .runtime.commom_utils import Autograd
from .runtime.register import Register

__version__ = "2.1"
device = runtime.device.device_instance.device_name
aten_lib = torch.library.Library("aten", "IMPL")


def enable(lib=aten_lib, unused=[]):
    Register(
        (
            ("abs", abs, Autograd.unable),
            ("add.Tensor", add, Autograd.unable),
            ("addmm", addmm, Autograd.unable),
            ("arange.start_step", arange_start, Autograd.unable),
            ("arange.start", arange_start, Autograd.unable),
            ("arange", arange, Autograd.unable),
            ("bitwise_and.Tensor", bitwise_and_tensor, Autograd.unable),
            ("bitwise_and.Scalar", bitwise_and_scalar, Autograd.unable),
            ("bitwise_and.Scalar_Tensor", bitwise_and_scalar_tensor, Autograd.unable),
            ("bitwise_not", bitwise_not, Autograd.unable),
            ("bitwise_or.Tensor", bitwise_or_tensor, Autograd.unable),
            ("bitwise_or.Scalar", bitwise_or_scalar, Autograd.unable),
            ("bitwise_or.Scalar_Tensor", bitwise_or_scalar_tensor, Autograd.unable),
            ("bmm", bmm, Autograd.unable),
            ("clamp", clamp, Autograd.unable),
            ("clamp.Tensor", clamp_tensor, Autograd.unable),
            ("cos", cos, Autograd.unable),
            ("pad", pad, Autograd.unable),
            ("cumsum", cumsum, Autograd.unable),
            ("div.Tensor", true_divide, Autograd.unable),
            ("div.Scalar", true_divide, Autograd.unable),
            ("div.Tensor_mode", div_mode, Autograd.unable),
            ("div.Scalar_mode", div_mode, Autograd.unable),
            ("divide.Tensor", true_divide, Autograd.unable),  # divide, an alias for div
            ("divide.Scalar", true_divide, Autograd.unable),
            ("divide.Tensor_mode", div_mode, Autograd.unable),
            ("divide.Scalar_mode", div_mode, Autograd.unable),
            (
                "true_divide.Tensor",
                true_divide,
                Autograd.unable,
            ),  # true_divide, an alias for div
            ("true_divide.Scalar", true_divide, Autograd.unable),
            ("floor_divide", floor_divide, Autograd.unable),
            ("floor_divide.Scalar", floor_divide, Autograd.unable),
            ("remainder.Tensor", remainder, Autograd.unable),
            ("native_dropout", native_dropout, Autograd.enable),
            ("erf", erf, Autograd.unable),
            ("embedding", embedding, Autograd.enable),
            ("eq.Tensor", eq, Autograd.unable),
            ("eq.Scalar", eq_scalar, Autograd.unable),
            ("exp", exp, Autograd.unable),
            ("exponential_", exponential_, Autograd.unable),
            ("ge.Tensor", ge, Autograd.unable),
            ("ge.Scalar", ge_scalar, Autograd.unable),
            ("gelu", gelu, Autograd.enable),
            ("native_group_norm", group_norm, Autograd.enable),
            ("_weight_norm_interface", weight_norm_interface, Autograd.enable),
            ("_weight_norm", weight_norm, Autograd.enable),
            ("gt.Tensor", gt, Autograd.unable),
            ("gt.Scalar", gt_scalar, Autograd.unable),
            ("isfinite", isfinite, Autograd.unable),
            ("isin.Tensor_Tensor", isin, Autograd.unable),
            ("isin.Scalar_Tensor", isin, Autograd.unable),
            ("isin.Tensor_Scalar", isin, Autograd.unable),
            ("isinf", isinf, Autograd.unable),
            ("isnan", isnan, Autograd.unable),
            ("minimum", minimum, Autograd.unable),
            ("maximum", maximum, Autograd.unable),
            ("native_layer_norm", layer_norm, Autograd.enable),
            ("le.Tensor", le, Autograd.unable),
            ("le.Scalar", le_scalar, Autograd.unable),
            ("lt.Tensor", lt, Autograd.unable),
            ("lt.Scalar", lt_scalar, Autograd.unable),
            ("rms_norm", rms_norm, Autograd.unable),
            ("rand", rand, Autograd.unable),
            ("randn", randn, Autograd.unable),
            ("rand_like", rand_like, Autograd.unable),
            ("randn_like", randn_like, Autograd.unable),
            ("zeros", zeros, Autograd.unable),
            ("ones", ones, Autograd.unable),
            ("full", full, Autograd.unable),
            ("zeros_like", zeros_like, Autograd.unable),
            ("ones_like", ones_like, Autograd.unable),
            ("full_like", full_like, Autograd.unable),
            ("resolve_neg", resolve_neg, Autograd.unable),
            ("resolve_conj", resolve_conj, Autograd.unable),
            ("normal.Tensor_float", normal_tensor_float, Autograd.unable),
            ("normal.float_Tensor", normal_float_tensor, Autograd.unable),
            ("normal.Tensor_Tensor", normal_tensor_tensor, Autograd.unable),
            ("uniform_", uniform_, Autograd.unable),
            ("mean", mean, Autograd.unable),
            ("mean.dim", mean_dim, Autograd.unable),
            ("mm", mm, Autograd.unable),
            ("mul.Tensor", mul, Autograd.unable),
            ("multinomial", multinomial, Autograd.unable),
            ("mv", mv, Autograd.unable),
            ("ne.Tensor", ne, Autograd.unable),
            ("ne.Scalar", ne_scalar, Autograd.unable),
            ("neg", neg, Autograd.unable),
            ("pow.Scalar", pow_scalar, Autograd.unable),
            ("pow.Tensor_Scalar", pow_tensor_scalar, Autograd.unable),
            ("pow.Tensor_Tensor", pow_tensor_tensor, Autograd.unable),
            ("reciprocal", reciprocal, Autograd.unable),
            ("relu", relu, Autograd.enable),
            ("rsqrt", rsqrt, Autograd.unable),
            ("sigmoid", sigmoid, Autograd.enable),
            ("silu", silu, Autograd.enable),
            ("sin", sin, Autograd.unable),
            ("softmax.int", softmax, Autograd.enable),
            ("sub.Tensor", sub, Autograd.unable),
            ("tanh", tanh, Autograd.enable),
            ("triu", triu, Autograd.unable),
            ("topk", topk, Autograd.unable),
            ("var_mean.correction", var_mean, Autograd.unable),
            ("linalg_vector_norm", vector_norm, Autograd.unable),
            ("where.self", where_self, Autograd.unable),
            ("where.ScalarSelf", where_scalar_self, Autograd.unable),
            ("where.ScalarOther", where_scalar_other, Autograd.unable),
            ("max", max, Autograd.unable),
            ("max.dim", max_dim, Autograd.unable),
            ("min", min, Autograd.unable),
            ("min.dim", min_dim, Autograd.unable),
            ("amax", amax, Autograd.unable),
            ("argmax", argmax, Autograd.unable),
            ("prod", prod, Autograd.unable),
            ("prod.dim_int", prod_dim, Autograd.unable),
            ("sum", sum, Autograd.unable),
            ("sum.dim_IntList", sum_dim, Autograd.unable),
            ("all", all, Autograd.unable),
            ("all.dim", all_dim, Autograd.unable),
            ("all.dims", all_dims, Autograd.unable),
            ("any", any, Autograd.unable),
            ("any.dim", any_dim, Autograd.unable),
            ("any.dims", any_dims, Autograd.unable),
            ("log_softmax.int", log_softmax, Autograd.enable),
            ("outer", outer, Autograd.enable),
            ("cross_entropy_loss", cross_entropy_loss, Autograd.enable),
            ("scatter.src", scatter, Autograd.unable),
            ("scatter.reduce", scatter, Autograd.unable),
            ("gather", gather, Autograd.unable),
            ("isclose", isclose, Autograd.unable),
            ("allclose", allclose, Autograd.unable),
            ("fill.Scalar", fill_scalar, Autograd.unable),
            ("fill.Tensor", fill_tensor, Autograd.unable),
            ("flip", flip, Autograd.unable),
            ("slice_scatter", slice_scatter_v2, Autograd.unable),
            ("select_scatter", select_scatter, Autograd.unable),
            ("index_select", index_select, Autograd.unable),
            ("tile", tile, Autograd.unable),
            ("masked_fill.Tensor", masked_fill, Autograd.unable),
            ("masked_fill.Scalar", masked_fill, Autograd.unable),
            ("masked_fill_.Tensor", masked_fill_, Autograd.unable),
            ("masked_fill_.Scalar", masked_fill_, Autograd.unable),
            ("_unique2", _unique2, Autograd.unable),
            ("_upsample_bicubic2d_aa", _upsample_bicubic2d_aa, Autograd.unable),
            ("upsample_nearest2d", upsample_nearest2d, Autograd.unable),
            ("nonzero", nonzero, Autograd.unable),
            ("repeat", repeat, Autograd.unable),
            ("masked_select", masked_select, Autograd.unable),
            ("stack", stack, Autograd.unable),
            ("hstack", hstack, Autograd.unable),
            ("cat", cat, Autograd.unable),
            ("repeat_interleave.self_int", repeat_interleave_self_int, Autograd.unable),
            ("vstack", vstack, Autograd.unable),
            ("repeat_interleave.Tensor", repeat_interleave_tensor, Autograd.unable),
            (
                "repeat_interleave.self_Tensor",
                repeat_interleave_self_tensor,
                Autograd.unable,
            ),
            ("randperm", randperm, Autograd.unable),
            ("diag", diag, Autograd.unable),
        ),
        unused_ops_list=unused,
        lib=lib,
    )


class use_gems:
    def __init__(self):
        self.lib = torch.library.Library("aten", "IMPL")

    def __enter__(self):
        enable(lib=self.lib)

    def __exit__(self, exc_type, exc_val, exc_tb):
        del self.lib


__all__ = [
    "enable",
    "use_gems",
]
