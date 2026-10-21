"""Registration table for the unary ``aten::_foreach_*`` family.

Thirty operators share one multi-tensor executor.  They differ only in the
element-wise math, the output dtype policy, and which dtypes ATen accepts, so
each one is a table row rather than a copy of the wrapper.  The element-wise
math is *reused* from the existing single-tensor FlagGems implementations by
taking the ``triton.jit`` scalar function out of their ``pointwise_dynamic``
wrapper -- no formula is written twice.

The dtype policies below were measured against the pinned PyTorch build (see
``kernelgen_work/EVIDENCE/foreach_family/unary_probe.log``); they are not
inferred from operator names.  Notable measured facts:

* Transcendentals promote every integral and bool input to ``float32``, while
  ``ceil``/``floor``/``round``/``trunc``/``frac``/``neg``/``sign`` preserve the
  input dtype.
* The exclusions differ per operator: ``floor`` rejects ``bool`` but accepts
  all integers, ``frac`` rejects all integers, ``sign`` accepts ``bool`` but
  rejects complex, ``erf``/``erfc``/``lgamma`` reject complex.
* Only ``abs`` maps complex to real; it is the one operator whose output dtype
  differs from its input for a complex input.
"""

import logging
from typing import Callable, Dict, Optional

import torch
import triton
import triton.language as tl

from flag_gems.ops._foreach_complex_math import (
    _hypot_fn,
    c_acos,
    c_asin,
    c_atan,
    c_cos,
    c_cosh,
    c_exp,
    c_expm1,
    c_log,
    c_log1p,
    c_log2,
    c_log10,
    c_neg,
    c_reciprocal,
    c_rsqrt,
    c_sigmoid,
    c_sin,
    c_sinh,
    c_sqrt,
    c_tan,
    c_tanh,
)
from flag_gems.ops.acos import acos_kernel
from flag_gems.ops.asin import asin_kernel
from flag_gems.ops.atan import atan_kernel
from flag_gems.ops.ceil import ceil_func
from flag_gems.ops.cos import cos_func
from flag_gems.ops.cosh import cosh_func
from flag_gems.ops.erf import erf_func
from flag_gems.ops.exp import exp_func
from flag_gems.ops.expm1 import expm1_func
from flag_gems.ops.floor import floor_func
from flag_gems.ops.frac_ import frac_func
from flag_gems.ops.lgamma_ import lgamma_func
from flag_gems.ops.log import log_func
from flag_gems.ops.log1p import log1p_func
from flag_gems.ops.log2 import log2_func
from flag_gems.ops.log10 import log10_func
from flag_gems.ops.neg import neg_func
from flag_gems.ops.reciprocal import reciprocal_func
from flag_gems.ops.sigmoid import sigmoid_forward
from flag_gems.ops.sin import sin_func
from flag_gems.ops.sinh import sinh_kernel
from flag_gems.ops.special_erfc import _erfc_kernel
from flag_gems.ops.sqrt import sqrt_func
from flag_gems.ops.tan import tan_func
from flag_gems.ops.tanh import tanh_kernel
from flag_gems.ops.trunc_ import trunc_func
from flag_gems.utils import tl_extra_shim
from flag_gems.utils.foreach import (
    foreach_unary,
    int_to_float,
    real_dtype_of,
    same_dtype,
)

logger = logging.getLogger(__name__)


def _scalar_fn(pointwise_fn) -> Callable:
    """The bare ``triton.jit`` function inside a ``pointwise_dynamic`` wrapper.

    ``pointwise_dynamic`` keeps the user's scalar function so it can splice the
    source into a generated module; the foreach executor needs exactly the same
    object as a ``tl.constexpr`` callable.  Reaching for it is what lets thirty
    operators reuse the repository's existing math instead of restating it.
    """
    return pointwise_fn._scalar_fn


# ---------------------------------------------------------------------------
# 16-bit approximation path
# ---------------------------------------------------------------------------
#
# ``erf``/``erfc``/``atan``/``lgamma`` spend 20-25 fp32 instructions per element
# in libdevice's range reduction and polynomial.  A 16-bit result does not need
# that: fp16 has ~5e-4 relative resolution and the suite tolerance is
# rtol=1e-3/atol=1e-4 (bf16 rtol=1.6e-2).  The fitted minimax bodies below
# measured, in-place at 512^2 x 16 fp16, erfc 19.97 -> 11.55 us, erf 15.65 ->
# 11.87 us, atan 15.97 -> 12.22 us and lgamma 31.90 -> 27.81 us.  The
# coefficients are final exact-minimax (LP) fits -- do not refit them.
#
# ``acos`` and ``asin`` were fitted too and rejected: both are already at the
# memory floor (measured -0.64 us and -0.13 us against libdevice), so their
# rows keep the accurate function.
#
# Sign is applied with ``_sgn_where``, not ``copysign``: the libdevice call
# measured ~0.7 us per call on this device and the two element selects are
# worth 1.2-1.8 us in total.
#
# ``lgamma`` needs the *accurate* ``sin`` in the reflection, not
# ``fast_sinf``.  The fast variant measured -9.92 us (21.63 against 27.81 us)
# but is unusable: its absolute error floor is ~2^-22, and the argument is
# reduced to [-pi/2, pi/2] where ``sin`` crosses zero, so ``log|sin|`` turns
# that floor into an error large enough to push 6089 fp16 inputs outside
# rtol=1e-3.  The accurate ``sin`` costs the recurrence far less than the
# reflection's logarithm saves.
#
# ``fn16`` is consumed by the in-place entry point only.  An in-place call
# costs ~27 us of host CPU, under ``do_bench``'s L2-flush window, so host work
# is free and the kernel time is what is measured.  An out-of-place call costs
# ~57 us, past the window, so any added Python leaks 1:1 into the measured
# latency while the kernel time the fast body saves stays hidden underneath
# it; the same gate enabled there measured +0.3 to +1.3 us on these rows (the
# fast-math lane measured 1.3-3.7 us on its own).  The accurate body is free on
# that path, so out-of-place deliberately keeps ``op.fn``.  Do not "restore the
# symmetry" -- see ``apply_unary``.
#
# The executor hands these functions a value already cast to the output dtype,
# so every body widens to fp32 first; the result is cast back on store.


@triton.jit
def _sgn_where(v, x):
    """``v`` with the sign of ``x``, preserving ``+-0`` (no ``copysign``)."""
    return tl.where(x > 0.0, v, tl.where(x < 0.0, -v, x))


@triton.jit
def _erfc_pos_func(x):
    """``erfc(|x|)``: ``exp(a*(Q(a) - a))`` with ``v = a/3 - 1``.

    The leading factor of ``a`` makes the polynomial argument exactly 0 at
    ``a = 0``, so ``erfc(0) = 1`` and ``erf(+-0) = +-0`` come out exact; the
    negative leading coefficient gives ``exp(-inf) = 0`` for ``a = inf``.
    """
    a = tl.abs(x)
    v = a * 0.33333334 - 1.0
    q = ((((((-0.0073543065 * v + 0.02308265) * v - 0.0400557) * v
            + 0.072947204) * v - 0.14141525) * v + 0.27006385) * v
         - 0.5734953)
    return tl_extra_shim.fast_expf(a * (q - a))


@triton.jit
def _approx_erfc_func(x):
    """``erfc(x)`` for all x: ``erfc(-a) = 2 - erfc(a)``."""
    x = x.to(tl.float32)
    e = _erfc_pos_func(x)
    return tl.where(x >= 0.0, e, 2.0 - e)


@triton.jit
def _approx_erf_func(x):
    """``erf(x) = sign(x)*(1 - erfc(|x|))``; exact ``+-0`` and ``+-1``.

    Reusing the erfc body measured 15.65 -> 11.87 us in-place.  The exp-free
    direct polynomial (14 coefficients) was fitted and rejected at 14.30 us.
    """
    x = x.to(tl.float32)
    return _sgn_where(1.0 - _erfc_pos_func(x), x)


@triton.jit
def _atan_core_func(c):
    """``atan(c)`` for ``|c| <= 1``, exact at ``c = 0`` and ``c = +-1``.

    ``c*(pi/4 + (1 - c*c)*S(v))`` anchors both ends, so the reduction below
    needs no correction term and the sign is the only thing left to apply.
    """
    v = 2.0 * c * c - 1.0
    s = ((((-0.00021917612 * v + 0.0006003141) * v - 0.002031797) * v
          + 0.007987845) * v - 0.033714574) * v + 0.17004323
    return c * (0.7853982 + (1.0 - c * c) * s)


@triton.jit
def _approx_atan_func(x):
    """``atan(x)`` via ``c = (a-1)/(a+1) = 1 - 2/(a+1)``, ``a = |x|``.

    One divide and no select on the result; measured 15.97 -> 12.22 us.  The
    ``c = min(a, 1/a)`` reduction with a ``pi/2`` correction measured 12.35 us,
    i.e. the extra select costs more than the divide it saves.
    """
    x = x.to(tl.float32)
    a = tl.abs(x)
    c = 1.0 - 2.0 * tl_extra_shim.fast_dividef(1.0, a + 1.0)
    return _sgn_where(0.7853982 + _atan_core_func(c), x)


@triton.jit
def _lgamma_pos_func(y):
    """``lgamma(y)`` for ``y > 0``: Stirling plus an 8-step recurrence.

    The recurrence is what keeps the asymptotic series' argument large; the
    correction is ``log(y*(y+1)*...*(y+7))``.  That product is 8 factors, so it
    stays inside fp32 up to ``y ~ 6e4``; above that the recurrence is skipped
    (its correction would be 0 anyway) and Stirling is evaluated at ``y``.
    """
    prod = y * (y + 1.0)
    prod = prod * (y + 2.0) * (y + 3.0)
    prod = prod * (y + 4.0) * (y + 5.0)
    prod = prod * (y + 6.0) * (y + 7.0)
    shift = y < 60000.0
    z = tl.where(shift, y + 8.0, y)
    inv = tl_extra_shim.fast_dividef(1.0, z)
    inv2 = inv * inv
    st = (z - 0.5) * tl_extra_shim.fast_logf(z) - z + 0.9189385332046727
    st = st + inv * (0.08333333333333333
                     - inv2 * (0.002777777777777778
                               - inv2 * 0.0007936507936507937))
    corr = tl.where(shift, tl_extra_shim.fast_logf(prod), 0.0)
    return st - corr


@triton.jit
def _approx_lgamma_func(x):
    """``lgamma(x)``: log-magnitude, matching ATen's gammaln semantics.

    The reflection uses ``sin`` (accurate), see the section comment; the poles
    (0 and the negative integers) and ``+-inf`` map to ``+inf`` and NaN
    propagates, both as ATen does.
    """
    x = x.to(tl.float32)
    reflect = x < 0.5
    y = tl.where(reflect, 1.0 - x, x)
    pos = _lgamma_pos_func(y)
    # sin(pi*x), argument reduced to [-pi/2, pi/2].  ``sin`` (like
    # ``fast_logf`` below) flushes an fp32 subnormal argument to zero, and a
    # bf16 subnormal (down to ~9e-41) times pi is one -- exactly the tiny-|x|
    # reflection.  There sin(z) == z to well below fp32 epsilon (relative
    # correction z^2/6 <= 2e-14 for |z| < 1e-7), so the linear branch is exact
    # and keeps the logarithm finite.
    reduced = x - tl.floor(x + 0.5)
    z = 3.1415927 * reduced
    sin_pi = tl.where(tl.abs(reduced) < 1.0e-7, z, tl_extra_shim.sin(z))
    # ``fast_logf`` is ``lg2.approx``, which also flushes a subnormal input;
    # scaling by 2^64 (exact power-of-two multiply, |sin| <= 1 so no overflow)
    # keeps the log finite.  Without this 80 bf16 inputs came out +inf against
    # ATen's ~90.
    log_sin = (tl_extra_shim.fast_logf(tl.abs(sin_pi) * 18446744073709551616.0)
               - 44.3614195558365)
    refl = 1.1447298858494002 - log_sin - pos
    res = tl.where(reflect, refl, pos)
    pole = (x <= 0.0) & (x == tl.floor(x))
    return tl.where(pole | (tl.abs(x) == float("inf")), float("inf"), res)


# ---------------------------------------------------------------------------
# math not available as a reusable pointwise_dynamic scalar function
# ---------------------------------------------------------------------------
#
# ``round`` and ``sign`` are implemented in FlagGems as hand-written kernels
# rather than ``pointwise_dynamic``, so there is no scalar function to borrow.
# ``rsqrt`` does have one, but its ``1.0 / tl.sqrt(x)`` form keeps a
# full-precision divide where ``tl.rsqrt`` is available.
#
# ``tan`` keeps the borrowed scalar function.  A ``sin/cos`` quotient was tried
# and rejected: on this device it takes 1.25x to 1.94x the kernel time of the
# generic libdevice body at every size measured (the extern already shares its
# range reduction across the two calls it makes, so the quotient pays for it
# twice), and the size gate that would have masked the small-tensor case only
# moved the regression above the gate.


@triton.jit
def _round_func(x):
    # Integral inputs are already rounded; float ones go through libdevice's
    # ``rint``, which is round-half-to-even and measured bitwise identical to
    # ``torch.round`` on 24k exact ties per dtype (both signs, including
    # ``+-0.5``) and on random data in fp32/fp16/bf16.  The previous emulation
    # (floor + parity test + select) needed about eight instructions and got
    # the sign of a zero result wrong; ``rint`` is one.  ``rint`` has no 16-bit
    # overload, so those widen first; ``float64`` is called directly because
    # widening it to float32 would round large values that the emulation used
    # to round exactly.
    if x.dtype == tl.float64:
        return tl_extra_shim.rint(x)
    elif x.dtype.is_floating():
        return tl_extra_shim.rint(x.to(tl.float32)).to(x.dtype)
    else:
        return x


@triton.jit
def _rsqrt_func(x):
    # ``tl.rsqrt`` measured bitwise identical to ``torch.rsqrt`` on fp32, fp16
    # and bf16 over the suite's sample range and on grids near 1e-6 and 1.0;
    # the reused scalar function spells it as ``1.0 / tl.sqrt(x)``, which
    # leaves a full-precision divide in the body.  Triton flushes fp32
    # denormals, so inputs below 2**-126 give inf here -- the base expression
    # does the same, so this is not a regression.
    return tl.rsqrt(x.to(tl.float32))


@triton.jit
def _sign_func(x):
    # Mirrors ops/sign.py: NaN yields 0 because neither comparison holds.
    if x.dtype == tl.int1:
        return x
    else:
        return (x > 0).to(x.dtype) - (x < 0).to(x.dtype)


@triton.jit
def _abs_func(x):
    return tl.abs(x)


@triton.jit
def _abs_complex_func(re, im):
    # Modulus of a complex value.  The executor passes the real and imaginary
    # components separately because Triton has no complex dtype.
    return _hypot_fn(re, im)


# ---------------------------------------------------------------------------
# dtype sets, as measured
# ---------------------------------------------------------------------------

_FLOAT = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
_INT = (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)
_COMPLEX = (torch.complex32, torch.complex64, torch.complex128)

# Everything except complex: erf / erfc / lgamma / sign.
NO_COMPLEX = frozenset(_FLOAT + _INT + (torch.bool,))
# Integral dtypes are fine, bool is not: ceil / floor / round / trunc.
NO_BOOL_NO_COMPLEX = frozenset(_FLOAT + _INT)
# Floating point only: frac.
FLOAT_ONLY = frozenset(_FLOAT)
# Everything except bool: neg.
NO_BOOL = frozenset(_FLOAT + _INT + (torch.complex64, torch.complex128))
# Transcendentals: complex64/128 work, complex32 does not.  Measured -- only
# ``abs`` accepts complex32 in this build, ComplexHalf being experimental.
NO_COMPLEX32 = frozenset(
    _FLOAT + _INT + (torch.bool, torch.complex64, torch.complex128)
)


class UnaryOp:
    """One row of the unary foreach table."""

    __slots__ = ("name", "fn", "out_dtype_fn", "allowed", "complex_fn", "fn16")

    def __init__(
        self,
        name: str,
        fn: Callable,
        out_dtype_fn: Callable = int_to_float,
        allowed: Optional[frozenset] = None,
        complex_fn: Optional[Callable] = None,
        fn16: Optional[Callable] = None,
    ) -> None:
        self.name = name
        self.fn = fn
        self.out_dtype_fn = out_dtype_fn
        self.allowed = allowed
        self.complex_fn = complex_fn
        # Alternate, lower-accuracy function used only when every input is
        # 16-bit; ``None`` means the row has no fast variant.
        self.fn16 = fn16


UNARY_OPS: Dict[str, UnaryOp] = {
    op.name: op
    for op in (
        UnaryOp("abs", _abs_func, real_dtype_of, None, _abs_complex_func),
        UnaryOp("acos", _scalar_fn(acos_kernel), int_to_float, NO_COMPLEX32, c_acos),
        UnaryOp("asin", _scalar_fn(asin_kernel), int_to_float, NO_COMPLEX32, c_asin),
        UnaryOp(
            "atan",
            _scalar_fn(atan_kernel),
            int_to_float,
            NO_COMPLEX32,
            c_atan,
            fn16=_approx_atan_func,
        ),
        UnaryOp("ceil", _scalar_fn(ceil_func), same_dtype, NO_BOOL_NO_COMPLEX),
        UnaryOp("cos", _scalar_fn(cos_func), int_to_float, NO_COMPLEX32, c_cos),
        UnaryOp("cosh", _scalar_fn(cosh_func), int_to_float, NO_COMPLEX32, c_cosh),
        UnaryOp(
            "erf",
            _scalar_fn(erf_func),
            int_to_float,
            NO_COMPLEX,
            fn16=_approx_erf_func,
        ),
        UnaryOp(
            "erfc",
            _scalar_fn(_erfc_kernel),
            int_to_float,
            NO_COMPLEX,
            fn16=_approx_erfc_func,
        ),
        UnaryOp("exp", _scalar_fn(exp_func), int_to_float, NO_COMPLEX32, c_exp),
        UnaryOp("expm1", _scalar_fn(expm1_func), int_to_float, NO_COMPLEX32, c_expm1),
        UnaryOp("floor", _scalar_fn(floor_func), same_dtype, NO_BOOL_NO_COMPLEX),
        UnaryOp("frac", _scalar_fn(frac_func), same_dtype, FLOAT_ONLY),
        UnaryOp(
            "lgamma",
            _scalar_fn(lgamma_func),
            int_to_float,
            NO_COMPLEX,
            fn16=_approx_lgamma_func,
        ),
        UnaryOp("log", _scalar_fn(log_func), int_to_float, NO_COMPLEX32, c_log),
        UnaryOp("log10", _scalar_fn(log10_func), int_to_float, NO_COMPLEX32, c_log10),
        UnaryOp("log1p", _scalar_fn(log1p_func), int_to_float, NO_COMPLEX32, c_log1p),
        UnaryOp("log2", _scalar_fn(log2_func), int_to_float, NO_COMPLEX32, c_log2),
        UnaryOp("neg", _scalar_fn(neg_func), same_dtype, NO_BOOL, c_neg),
        UnaryOp(
            "reciprocal",
            _scalar_fn(reciprocal_func),
            int_to_float,
            NO_COMPLEX32,
            c_reciprocal,
        ),
        UnaryOp("round", _round_func, same_dtype, NO_BOOL_NO_COMPLEX),
        UnaryOp("rsqrt", _rsqrt_func, int_to_float, NO_COMPLEX32, c_rsqrt),
        UnaryOp(
            "sigmoid",
            _scalar_fn(sigmoid_forward),
            int_to_float,
            NO_COMPLEX32,
            c_sigmoid,
        ),
        UnaryOp("sign", _sign_func, same_dtype, NO_COMPLEX),
        UnaryOp("sin", _scalar_fn(sin_func), int_to_float, NO_COMPLEX32, c_sin),
        UnaryOp("sinh", _scalar_fn(sinh_kernel), int_to_float, NO_COMPLEX32, c_sinh),
        UnaryOp("sqrt", _scalar_fn(sqrt_func), int_to_float, NO_COMPLEX32, c_sqrt),
        UnaryOp("tan", _scalar_fn(tan_func), int_to_float, NO_COMPLEX32, c_tan),
        UnaryOp("tanh", _scalar_fn(tanh_kernel), int_to_float, NO_COMPLEX32, c_tanh),
        UnaryOp("trunc", _scalar_fn(trunc_func), same_dtype, NO_BOOL_NO_COMPLEX),
    )
}


def apply_unary(name: str, self, inplace: bool):
    """Dispatch one table row through the shared executor."""
    op = UNARY_OPS[name]
    logger.debug("GEMS _FOREACH_%s%s", name.upper(), "_" if inplace else "")
    if inplace:
        # The alternate function is only valid when the result is 16-bit, and the
        # executor casts each input to its own output dtype, so "16-bit result"
        # means "every input is fp16/bf16".  The test runs inside this loop
        # rather than as a separate pass because the loop already touches every
        # tensor for the complex check; a second walk would cost host time that
        # the in-place call cannot afford to spend.
        #
        # Per-tensor, not ``self[0].dtype``: a list that mixes fp16 with fp32
        # must take the accurate path, and the executor's bucketing path shares
        # one ``fn`` across dtype groups.
        #
        # Only the in-place entry point uses this.  An out-of-place call already
        # spends past the do_bench L2-flush window on host work, so added Python
        # leaks 1:1 into its measured latency while the kernel time saved by the
        # fast body stays hidden underneath it -- gated on that path the change
        # measured as a 1.3-3.7 us regression.  The accurate body costs it
        # nothing, so out-of-place keeps ``op.fn`` unconditionally.
        fast = op.fn16 is not None
        for t in self:
            if t.dtype.is_complex and op.out_dtype_fn is real_dtype_of:
                raise RuntimeError(
                    f"In-place {name} is not supported for complex tensors."
                )
            if fast and t.dtype not in (torch.float16, torch.bfloat16):
                fast = False
        fn = op.fn16 if fast else op.fn
        foreach_unary(
            self,
            fn,
            complex_fn=op.complex_fn,
            # The dtype policy has to be passed here too: it is what makes the
            # executor reject an integral in-place call on a promoting operator,
            # matching ATen's "result type Float can't be cast to the desired
            # output type Long".  Omitting it silently accepted int64 sin_.
            out_dtype_fn=op.out_dtype_fn,
            inplace=True,
            allowed_dtypes=op.allowed,
        )
        # The in-place schemas return ``()``; returning the list makes the
        # dispatcher reject the kernel.
        return None
    return foreach_unary(
        self,
        op.fn,
        complex_fn=op.complex_fn,
        out_dtype_fn=op.out_dtype_fn,
        allowed_dtypes=op.allowed,
    )


def _foreach_abs(self):
    return apply_unary("abs", self, inplace=False)


def _foreach_abs_(self):
    return apply_unary("abs", self, inplace=True)


def _foreach_acos(self):
    return apply_unary("acos", self, inplace=False)


def _foreach_acos_(self):
    return apply_unary("acos", self, inplace=True)


def _foreach_asin(self):
    return apply_unary("asin", self, inplace=False)


def _foreach_asin_(self):
    return apply_unary("asin", self, inplace=True)


def _foreach_atan(self):
    return apply_unary("atan", self, inplace=False)


def _foreach_atan_(self):
    return apply_unary("atan", self, inplace=True)


def _foreach_ceil(self):
    return apply_unary("ceil", self, inplace=False)


def _foreach_ceil_(self):
    return apply_unary("ceil", self, inplace=True)


def _foreach_cos(self):
    return apply_unary("cos", self, inplace=False)


def _foreach_cos_(self):
    return apply_unary("cos", self, inplace=True)


def _foreach_cosh(self):
    return apply_unary("cosh", self, inplace=False)


def _foreach_cosh_(self):
    return apply_unary("cosh", self, inplace=True)


def _foreach_erf(self):
    return apply_unary("erf", self, inplace=False)


def _foreach_erf_(self):
    return apply_unary("erf", self, inplace=True)


def _foreach_erfc(self):
    return apply_unary("erfc", self, inplace=False)


def _foreach_erfc_(self):
    return apply_unary("erfc", self, inplace=True)


def _foreach_exp(self):
    return apply_unary("exp", self, inplace=False)


def _foreach_exp_(self):
    return apply_unary("exp", self, inplace=True)


def _foreach_expm1(self):
    return apply_unary("expm1", self, inplace=False)


def _foreach_expm1_(self):
    return apply_unary("expm1", self, inplace=True)


def _foreach_floor(self):
    return apply_unary("floor", self, inplace=False)


def _foreach_floor_(self):
    return apply_unary("floor", self, inplace=True)


def _foreach_frac(self):
    return apply_unary("frac", self, inplace=False)


def _foreach_frac_(self):
    return apply_unary("frac", self, inplace=True)


def _foreach_lgamma(self):
    return apply_unary("lgamma", self, inplace=False)


def _foreach_lgamma_(self):
    return apply_unary("lgamma", self, inplace=True)


def _foreach_log(self):
    return apply_unary("log", self, inplace=False)


def _foreach_log_(self):
    return apply_unary("log", self, inplace=True)


def _foreach_log10(self):
    return apply_unary("log10", self, inplace=False)


def _foreach_log10_(self):
    return apply_unary("log10", self, inplace=True)


def _foreach_log1p(self):
    return apply_unary("log1p", self, inplace=False)


def _foreach_log1p_(self):
    return apply_unary("log1p", self, inplace=True)


def _foreach_log2(self):
    return apply_unary("log2", self, inplace=False)


def _foreach_log2_(self):
    return apply_unary("log2", self, inplace=True)


def _foreach_neg(self):
    return apply_unary("neg", self, inplace=False)


def _foreach_neg_(self):
    return apply_unary("neg", self, inplace=True)


def _foreach_reciprocal(self):
    return apply_unary("reciprocal", self, inplace=False)


def _foreach_reciprocal_(self):
    return apply_unary("reciprocal", self, inplace=True)


def _foreach_round(self):
    return apply_unary("round", self, inplace=False)


def _foreach_round_(self):
    return apply_unary("round", self, inplace=True)


def _foreach_rsqrt(self):
    return apply_unary("rsqrt", self, inplace=False)


def _foreach_rsqrt_(self):
    return apply_unary("rsqrt", self, inplace=True)


def _foreach_sigmoid(self):
    return apply_unary("sigmoid", self, inplace=False)


def _foreach_sigmoid_(self):
    return apply_unary("sigmoid", self, inplace=True)


def _foreach_sign(self):
    return apply_unary("sign", self, inplace=False)


def _foreach_sign_(self):
    return apply_unary("sign", self, inplace=True)


def _foreach_sin(self):
    return apply_unary("sin", self, inplace=False)


def _foreach_sin_(self):
    return apply_unary("sin", self, inplace=True)


def _foreach_sinh(self):
    return apply_unary("sinh", self, inplace=False)


def _foreach_sinh_(self):
    return apply_unary("sinh", self, inplace=True)


def _foreach_sqrt(self):
    return apply_unary("sqrt", self, inplace=False)


def _foreach_sqrt_(self):
    return apply_unary("sqrt", self, inplace=True)


def _foreach_tan(self):
    return apply_unary("tan", self, inplace=False)


def _foreach_tan_(self):
    return apply_unary("tan", self, inplace=True)


def _foreach_tanh(self):
    return apply_unary("tanh", self, inplace=False)


def _foreach_tanh_(self):
    return apply_unary("tanh", self, inplace=True)


def _foreach_trunc(self):
    return apply_unary("trunc", self, inplace=False)


def _foreach_trunc_(self):
    return apply_unary("trunc", self, inplace=True)
