import torch
import torch_musa
import pytest

torch.musa.manual_seed(996)


'''
USE: pytest -s tests/test_torch_musa_unsupport.py

Op in torch_musa:
    Unsupported:
        v1.0:
            pow: pow_scalar_tensor 
            rsub
        v2.0:
            all: all_dim, all_dims
            any: any_dim, any_dims
        v3.0:
            no test case
        v4.0:

    Precision error:
        v1.0:
            layer_norm
            add: add, add_scalar_tensor fp16
            pow: pow_tensor_scalar fp16
            sub: sub, sub_scalar_tensor fp16
        v2.0:
            tanh: fp16
            clamp
            outer: fp16
            var_mean: 
            vector_norm
            group_norm: fp32
            sigmoid: fp16
        v3.0:
            no test case
        v4.0:
            gelu_and_mul
            silu_and_mul: fp16
            skip_rms_norm: equal to rms_norm
            skip_layer_norm: equal to layernorm
            apply_rotary_position_embedding

Note: 'all_dims' and 'any_dims' will cause a segment fault, and pytest cannot print the results. 
      These two ops need to be commented out.
'''


TO_CPU = True

major, minor = torch.__version__.split(".")[:2]
skip_expr = major < "2" or minor < "2"
skip_reason = "PyTorch < 2.2.0 does not support"

RESOLUTION = {
    torch.float16: 1e-3,
    torch.float32: 1.3e-6,
    torch.bfloat16: 0.016,
}

POINTWISE_SHAPES = [(1024, 1024), (16, 1024, 256), (16, 128, 64, 64), (20, 320, 15)]
REDUCTION_SHAPES = [(4096, 256 * i) for i in range(1, 10, 2)]
MNK_SHAPES = [15, 160, 1024]

FLOAT_DTYPES = [torch.float16, torch.float32]
INT_DTYPES = [torch.int16, torch.int32]

SCALARS = [0.001, -0.999, 100.001, -111.999]
DIM_LIST = [0, 1]
DIMS_LIST = [0, 1, [0, 1], [1, 0]]


def to_reference(inp, upcast=False):
    if inp is None:
        return None
    ref_inp = inp
    if TO_CPU:
        ref_inp = ref_inp.to("cpu")
    if upcast:
        ref_inp = ref_inp.to(torch.float32)
    return ref_inp


def gems_assert_close(a, b, dtype, equal_nan=False, reduce_dim=1):
    if TO_CPU:
        a = a.to("cpu")
    b = b.to(dtype)
    atol = 1e-4 * reduce_dim
    rtol = RESOLUTION[dtype]
    torch.testing.assert_close(a, b, atol=atol, rtol=rtol, equal_nan=equal_nan)


def gems_assert_equal(a, b):
    if TO_CPU:
        a = a.to("cpu")
    assert torch.equal(a, b)


############################## UNPASSED OP ##############################

################## UNSUPPORTED OP ###################
@pytest.mark.parametrize("scalar", SCALARS)
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_pow_scalar_tensor(scalar, shape, dtype):
    inp1 = scalar
    inp2 = torch.randn(shape, dtype=dtype, device="musa")
    ref_inp2 = to_reference(inp2, True)

    ref_out = torch.pow(inp1, ref_inp2)
    res_out = torch.pow(inp1, inp2)

    gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("alpha", SCALARS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_rsub(shape, alpha, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device="musa")
    inp2 = torch.randn(shape, dtype=dtype, device="musa")
    ref_inp1 = to_reference(inp1, True)
    ref_inp2 = to_reference(inp2, True)

    ref_out = torch.rsub(ref_inp1, ref_inp2, alpha=alpha)
    res_out = torch.rsub(inp1, inp2, alpha=alpha)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.skipif(skip_expr, reason=skip_reason)
@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dim", DIM_LIST)
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + [torch.bool])
@pytest.mark.parametrize("kind", ["normal", "allTrue"])
def test_accuracy_all_dim(shape, dim, keepdim, dtype, kind):
    if kind == "allTrue":
        inp = torch.ones(shape, dtype=dtype, device="musa")
    else:
        inp = torch.randint(0, 2, shape, dtype=dtype, device="musa")
    ref_inp = to_reference(inp, True)

    ref_out = torch.all(ref_inp, dim=dim, keepdim=keepdim)
    res_out = torch.all(inp, dim=dim, keepdim=keepdim)
    gems_assert_equal(res_out, ref_out)


@pytest.mark.skipif(skip_expr, reason=skip_reason)
@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dim", DIMS_LIST)
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + [torch.bool])
@pytest.mark.parametrize("kind", ["normal", "allTrue"])
def test_accuracy_all_dims(shape, dim, keepdim, dtype, kind):
    if kind == "allTrue":
        inp = torch.ones(shape, dtype=dtype, device="musa")
    else:
        inp = torch.randint(0, 2, shape, dtype=dtype, device="musa")
    ref_inp = to_reference(inp, True)

    ref_out = torch.all(ref_inp, dim=dim, keepdim=keepdim)
    res_out = torch.all(inp, dim=dim, keepdim=keepdim)
    gems_assert_equal(res_out, ref_out)


@pytest.mark.skipif(skip_expr, reason=skip_reason)
@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dim", DIM_LIST)
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + [torch.bool])
@pytest.mark.parametrize("kind", ["normal", "allFalse"])
def test_accuracy_any_dim(shape, dim, keepdim, dtype, kind):
    if kind == "allFalse":
        inp = torch.zeros(shape, dtype=dtype, device="musa")
    else:
        inp = torch.randint(0, 2, shape, dtype=dtype, device="musa")
    ref_inp = to_reference(inp, True)

    ref_out = torch.any(ref_inp, dim=dim, keepdim=keepdim)
    res_out = torch.any(inp, dim=dim, keepdim=keepdim)
    gems_assert_equal(res_out, ref_out)


@pytest.mark.skipif(skip_expr, reason=skip_reason)
@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dim", DIMS_LIST)
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + [torch.bool])
@pytest.mark.parametrize("kind", ["normal", "allFalse"])
def test_accuracy_any_dims(shape, dim, keepdim, dtype, kind):
    if kind == "allFalse":
        inp = torch.zeros(shape, dtype=dtype, device="musa")
    else:
        inp = torch.randint(0, 2, shape, dtype=dtype, device="musa")
    ref_inp = to_reference(inp, True)

    ref_out = torch.any(ref_inp, dim=dim, keepdim=keepdim)
    res_out = torch.any(inp, dim=dim, keepdim=keepdim)
    gems_assert_equal(res_out, ref_out)


################### PRECISION ERROR OP ###################
@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_layernorm(shape, dtype):
    M = shape[0]
    N = shape[1]
    layer_shape = [
        N,
    ]
    inp = torch.randn(shape, dtype=dtype, device="musa", requires_grad=True)
    weight = torch.randn(layer_shape, dtype=dtype, device="musa", requires_grad=True)
    bias = torch.randn(layer_shape, dtype=dtype, device="musa", requires_grad=True)
    eps = 1e-5

    ref_inp = to_reference(inp, True)
    ref_weight = to_reference(weight, True)
    ref_bias = to_reference(bias, True)

    ref_out = torch.layer_norm(
        ref_inp,
        list(layer_shape),
        weight=ref_weight,
        bias=ref_bias,
        eps=eps,
    )

    res_out = torch.layer_norm(
        inp, list(layer_shape), weight=weight, bias=bias, eps=eps
    )
    res_mean = torch.mean(inp, dim=1)
    res_var = torch.var(inp, dim=1, correction=0)
    res_rstd = torch.rsqrt(res_var + eps)

    ref_mean = torch.mean(ref_inp, dim=1)
    ref_var = torch.var(ref_inp, dim=1, correction=0)
    ref_rstd = torch.rsqrt(ref_var + eps)
    gems_assert_close(res_mean, ref_mean, dtype)
    gems_assert_close(res_rstd, ref_rstd, dtype)
    gems_assert_close(res_out, ref_out, dtype)

    out_grad = torch.randn_like(inp)
    ref_grad = to_reference(out_grad, True)

    (ref_in_grad, ref_weight_grad, ref_bias_grad) = torch.autograd.grad(
        ref_out, (ref_inp, ref_weight, ref_bias), ref_grad
    )
    (res_in_grad, res_weight_grad, res_bias_grad) = torch.autograd.grad(
        res_out, (inp, weight, bias), out_grad
    )
    gems_assert_close(res_in_grad, ref_in_grad, dtype, reduce_dim=N)
    gems_assert_close(res_weight_grad, ref_weight_grad, dtype, reduce_dim=M)
    gems_assert_close(res_bias_grad, ref_bias_grad, dtype, reduce_dim=M)

    
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("approximate", ["none", "tanh"])
def test_accuracy_gelu_and_mul(shape, approximate, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device="musa")
    inp2 = torch.randn(shape, dtype=dtype, device="musa")
    ref_inp1 = to_reference(inp1, True)
    ref_inp2 = to_reference(inp2, True)

    ref_out = torch.mul(
        torch.nn.functional.gelu(ref_inp1, approximate=approximate), ref_inp2
    )
    res_out = torch.mul(
    torch.nn.functional.gelu(inp1, approximate=approximate), inp2
    )

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_silu_and_mul(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device="musa")
    inp2 = torch.randn(shape, dtype=dtype, device="musa")
    ref_inp1 = to_reference(inp1, True)
    ref_inp2 = to_reference(inp2, True)

    ref_out = torch.mul(torch.nn.functional.silu(ref_inp1), ref_inp2)
    res_out = torch.mul(torch.nn.functional.silu(inp1), inp2)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("alpha", SCALARS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_add(shape, alpha, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device="musa")
    inp2 = torch.randn(shape, dtype=dtype, device="musa")
    ref_inp1 = to_reference(inp1, True)
    ref_inp2 = to_reference(inp2, True)

    ref_out = torch.add(ref_inp1, ref_inp2, alpha=alpha)
    res_out = torch.add(inp1, inp2, alpha=alpha)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("scalar", SCALARS)
@pytest.mark.parametrize("alpha", SCALARS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_add_scalar_tensor(shape, scalar, alpha, dtype):
    inp1 = scalar
    inp2 = torch.randn(shape, dtype=dtype, device="musa")
    ref_inp2 = to_reference(inp2, True)

    ref_out = torch.add(inp1, ref_inp2, alpha=alpha)
    res_out = torch.add(inp1, inp2, alpha=alpha)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("scalar", SCALARS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_pow_tensor_scalar(scalar, shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device="musa")
    inp2 = scalar
    ref_inp1 = to_reference(inp1, True)

    ref_out = torch.pow(ref_inp1, inp2)
    res_out = torch.pow(inp1, inp2)

    gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("alpha", SCALARS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_sub(shape, alpha, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device="musa")
    inp2 = torch.randn(shape, dtype=dtype, device="musa")
    ref_inp1 = to_reference(inp1, True)
    ref_inp2 = to_reference(inp2, True)

    ref_out = torch.sub(ref_inp1, ref_inp2, alpha=alpha)
    res_out = torch.sub(inp1, inp2, alpha=alpha)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_tanh(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device="musa", requires_grad=True)
    ref_inp = to_reference(inp, True)

    ref_out = torch.tanh(ref_inp)
    res_out = torch.tanh(inp)

    gems_assert_close(res_out, ref_out, dtype)

    out_grad = torch.randn_like(inp)
    ref_grad = to_reference(out_grad, True)

    (ref_in_grad,) = torch.autograd.grad(ref_out, ref_inp, ref_grad)
    (res_in_grad,) = torch.autograd.grad(res_out, inp, out_grad)
    gems_assert_close(res_in_grad, ref_in_grad, dtype)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("scalar", SCALARS)
@pytest.mark.parametrize("alpha", SCALARS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_sub_scalar_tensor(shape, scalar, alpha, dtype):
    inp1 = scalar
    inp2 = torch.randn(shape, dtype=dtype, device="musa")
    ref_inp2 = to_reference(inp2, True)

    ref_out = torch.sub(inp1, ref_inp2, alpha=alpha)
    res_out = torch.sub(inp1, inp2, alpha=alpha)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("maxi", SCALARS)
@pytest.mark.parametrize("mini", SCALARS)
@pytest.mark.parametrize("isnone", [None, "max", "min"])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_clamp(shape, maxi, mini, isnone, dtype):
    inp = torch.randn(shape, dtype=dtype, device="musa")
    if isnone == "min":
        mini = None
    elif isnone == "max":
        maxi = None
    ref_inp = to_reference(inp, True)

    ref_out = torch.clamp(ref_inp, min=mini, max=maxi)
    res_out = torch.clamp(inp, min=mini, max=maxi)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.parametrize("M", MNK_SHAPES)
@pytest.mark.parametrize("N", MNK_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_outer(M, N, dtype):
    inp1 = torch.randn(M, dtype=dtype, device="musa", requires_grad=True)
    inp2 = torch.randn(N, dtype=dtype, device="musa", requires_grad=True)
    ref_inp1 = to_reference(inp1, True)
    ref_inp2 = to_reference(inp2, True)

    ref_out = torch.outer(ref_inp1, ref_inp2)
    res_out = torch.outer(inp1, inp2)
    gems_assert_close(res_out, ref_out, dtype)

    out_grad = torch.randn_like(res_out)
    ref_grad = to_reference(out_grad, True)

    ref_in1_grad, ref_in2_grad = torch.autograd.grad(
        ref_out, (ref_inp1, ref_inp2), ref_grad
    )
    res_in1_grad, res_in2_grad = torch.autograd.grad(res_out, (inp1, inp2), out_grad)
    gems_assert_close(res_in1_grad, ref_in1_grad, dtype)
    gems_assert_close(res_in2_grad, ref_in2_grad, dtype)


@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dim", DIMS_LIST)
@pytest.mark.parametrize("correction", [0, 1])
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_varmean(shape, dim, correction, keepdim, dtype):
    inp = torch.randn(shape, dtype=dtype, device="musa")
    ref_inp = to_reference(inp, True)

    ref_var, ref_mean = torch.var_mean(
        ref_inp, dim, correction=correction, keepdim=keepdim
    )
    res_var, res_mean = torch.var_mean(
        inp, dim, correction=correction, keepdim=keepdim
    )

    gems_assert_close(res_mean, ref_mean, dtype)
    gems_assert_close(res_var, ref_var, dtype)


@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("ord", [2, float("inf"), -float("inf"), 0, 1])
@pytest.mark.parametrize("dim", DIMS_LIST)
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_vectornorm(shape, ord, dim, keepdim, dtype):
    inp = torch.randn(shape, dtype=dtype, device="musa")
    ref_inp = to_reference(inp, True)

    ref_out = torch.linalg.vector_norm(ref_inp, ord, dim, keepdim)
    res_out = torch.linalg.vector_norm(inp, ord, dim, keepdim)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.parametrize(
    "N, C, H, W, num_groups",
    [
        (16, 3, 16, 16, 1),
        (32, 32, 32, 32, 8),
        (1, 32, 32, 32, 8),
        (1, 32, 32, 32, 16),
        (1, 64, 32, 32, 16),
        (1, 64, 32, 32, 32),
        (1, 64, 32, 32, 64),
    ],
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_groupnorm(N, C, H, W, num_groups, dtype):
    HW = H * W
    inp = torch.randn(size=(N, C, H, W), dtype=dtype, device="musa", requires_grad=True)
    weight = torch.randn(size=(C,), dtype=dtype, device="musa", requires_grad=True)
    bias = torch.randn(size=(C,), dtype=dtype, device="musa", requires_grad=True)
    eps = 1e-5

    ref_inp = to_reference(inp, True)
    ref_weight = to_reference(weight, True)
    ref_bias = to_reference(bias, True)

    ref_out = torch.nn.functional.group_norm(
        ref_inp, num_groups, weight=ref_weight, bias=ref_bias, eps=eps
    )
    ref_mean = torch.mean(ref_inp.reshape([N, num_groups, -1]), dim=2)
    ref_var = torch.var(ref_inp.reshape([N, num_groups, -1]), dim=2, correction=0)
    ref_rstd = torch.rsqrt(ref_var + eps)

    # (res_out, res_mean, res_rstd) = flag_gems.group_norm(
    #     inp, weight, bias, N, C, HW, num_groups, eps
    # )
    res_out = torch.nn.functional.group_norm(
        inp, num_groups, weight=weight, bias=bias, eps=eps
    )
    res_mean = torch.mean(inp.reshape([N, num_groups, -1]), dim=2)
    res_var = torch.var(inp.reshape([N, num_groups, -1]), dim=2, correction=0)
    res_rstd = torch.rsqrt(res_var + eps)

    gems_assert_close(res_mean, ref_mean, dtype)
    gems_assert_close(res_rstd, ref_rstd, dtype)
    gems_assert_close(res_out, ref_out, dtype)

    out_grad = torch.randn_like(inp)
    ref_grad = to_reference(out_grad, True)

    (ref_in_grad, ref_weight_grad, ref_bias_grad) = torch.autograd.grad(
        ref_out, (ref_inp, ref_weight, ref_bias), ref_grad
    )
    (res_in_grad, res_weight_grad, res_bias_grad) = torch.autograd.grad(
        res_out, (inp, weight, bias), out_grad
    )
    group_size = C // num_groups
    gems_assert_close(res_in_grad, ref_in_grad, dtype, reduce_dim=group_size * HW)
    gems_assert_close(res_weight_grad, ref_weight_grad, dtype, reduce_dim=N * HW)
    gems_assert_close(res_bias_grad, ref_bias_grad, dtype, reduce_dim=N * HW)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_sigmoid(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device="musa", requires_grad=True)
    ref_inp = to_reference(inp, True)

    ref_out = torch.sigmoid(ref_inp)
    res_out = torch.sigmoid(inp)

    gems_assert_close(res_out, ref_out, dtype)

    out_grad = torch.randn_like(inp)
    ref_grad = to_reference(out_grad, True)

    (ref_in_grad,) = torch.autograd.grad(ref_out, ref_inp, ref_grad)
    (res_in_grad,) = torch.autograd.grad(res_out, inp, out_grad)
    gems_assert_close(res_in_grad, ref_in_grad, dtype)


@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_skip_layernorm(shape, dtype):
    N = shape[1]
    layer_shape = [
        N,
    ]
    inp = torch.randn(shape, dtype=dtype, device="musa")
    residual = torch.randn(shape, dtype=dtype, device="musa")
    weight = torch.randn(layer_shape, dtype=dtype, device="musa")
    bias = torch.randn(layer_shape, dtype=dtype, device="musa")
    eps = 1e-5

    ref_inp = to_reference(inp, True)
    ref_residual = to_reference(residual, True)
    ref_weight = to_reference(weight, True)
    ref_bias = to_reference(bias, True)

    ref_out = torch.layer_norm(
        ref_inp + ref_residual,
        list(layer_shape),
        weight=ref_weight,
        bias=ref_bias,
        eps=eps,
    )
    res_out = torch.layer_norm(
        inp + residual, list(layer_shape), weight=weight, bias=bias, eps=eps
    )

    gems_assert_close(res_out, ref_out, dtype)

    
@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_skip_rmsnorm(shape, dtype):
    N = shape[1]
    layer_shape = [
        N,
    ]
    inp = torch.randn(shape, dtype=dtype, device="musa")
    residual = torch.randn(shape, dtype=dtype, device="musa")
    weight = torch.randn(layer_shape, dtype=dtype, device="musa")
    eps = 1e-5

    ref_inp = to_reference(inp, True)
    ref_residual = to_reference(residual, True)
    ref_weight = to_reference(weight, True)

    def _torch_rms_norm(x, residual, weight, eps):
        x = x + residual
        variance = x.pow(2).mean(-1, keepdim=True)
        hidden_states = x * torch.rsqrt(variance + eps)
        return weight * hidden_states

    ref_out = _torch_rms_norm(
        ref_inp,
        ref_residual,
        weight=ref_weight,
        eps=eps,
    )

    res_out = torch.rms_norm(
        inp + residual, list(layer_shape), weight=weight, eps=eps
    )

    gems_assert_close(res_out, ref_out, dtype)


def get_rope_cos_sin(max_seq_len, dim, dtype, base=10000, device="musa"):
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
    t = torch.arange(max_seq_len, device=device, dtype=inv_freq.dtype)
    freqs = torch.outer(t, inv_freq)
    cos = freqs.cos().to(dtype)
    sin = freqs.sin().to(dtype)
    return cos, sin


# Copied from transformers.models.llama.modeling_llama.rotate_half
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.cohere.modeling_cohere.rotate_half
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/cohere/modeling_cohere.py
def rotate_interleave(x):
    """Rotates interleave the hidden dims of the input."""
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def torch_apply_rotary_pos_emb(
    q,
    k,
    cos,
    sin,
    position_ids,
    rotary_interleaved: bool = False,
):
    # q = q.float()
    # k = k.float()
    cos = cos[position_ids].unsqueeze(-2)  # [bs, seq_len, 1, dim/2]
    sin = sin[position_ids].unsqueeze(-2)  # [bs, seq_len, 1, dim/2]
    if rotary_interleaved:
        cos = torch.repeat_interleave(cos, 2, dim=-1)  # [bs, seq_len, 1, dim]
        sin = torch.repeat_interleave(sin, 2, dim=-1)  # [bs, seq_len, 1, dim]
        rotate_fn = rotate_interleave
    else:
        cos = torch.cat([cos, cos], dim=-1)  # [bs, seq_len, 1, dim]
        sin = torch.cat([sin, sin], dim=-1)  # [bs, seq_len, 1, dim]
        rotate_fn = rotate_half

    q_embed = (q * cos) + (rotate_fn(q) * sin)
    k_embed = (k * cos) + (rotate_fn(k) * sin)

    return q_embed, k_embed


@pytest.mark.parametrize("batch_size", [4, 8])
@pytest.mark.parametrize("max_seq_len", [512, 2048])
@pytest.mark.parametrize("q_heads,k_heads", [(8, 1), (6, 2), (1, 1), (8, 8)])
@pytest.mark.parametrize("head_dim", [64, 96, 128, 256])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("rotary_interleaved", [True, False])
def test_apply_rotary_pos_emb(
    batch_size,
    max_seq_len,
    q_heads,
    k_heads,
    head_dim,
    dtype,
    rotary_interleaved,
):
    seq_len = torch.randint(1, max_seq_len, (1,)).item()
    q = torch.randn(
        (batch_size, seq_len, q_heads, head_dim), dtype=dtype, device="musa"
    )
    k = torch.randn(
        (batch_size, seq_len, k_heads, head_dim), dtype=dtype, device="musa"
    )

    position_ids = torch.randint(0, max_seq_len, (batch_size, seq_len), device="musa")
    ref_cos, ref_sin = get_rope_cos_sin(max_seq_len, head_dim, dtype, device="cpu")
    cos, sin = get_rope_cos_sin(max_seq_len, head_dim, dtype, device="cpu")

    ref_q = to_reference(q, True)
    ref_k = to_reference(k, True)
    ref_cos = to_reference(ref_cos, True)
    ref_sin = to_reference(ref_sin, True)
    ref_position_ids = to_reference(position_ids)

    cos = cos.to('musa')
    sin = sin.to('musa')

    q_embed_ref, k_embed_ref = torch_apply_rotary_pos_emb(
        q=ref_q,
        k=ref_k,
        cos=ref_cos,
        sin=ref_sin,
        position_ids=ref_position_ids,
        rotary_interleaved=rotary_interleaved,
    )

    q_embed_out, k_embed_out = torch_apply_rotary_pos_emb(
        q=q,
        k=k,
        cos=cos,
        sin=sin,
        position_ids=position_ids,
        rotary_interleaved=rotary_interleaved,
    )

    gems_assert_close(q_embed_out, q_embed_ref, dtype)
    gems_assert_close(k_embed_out, k_embed_ref, dtype)


############################## PASSED OP ##############################


@pytest.mark.parametrize("M", MNK_SHAPES)
@pytest.mark.parametrize("N", MNK_SHAPES)
@pytest.mark.parametrize("K", MNK_SHAPES)
@pytest.mark.parametrize("alpha", SCALARS)
@pytest.mark.parametrize("beta", SCALARS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_addmm(M, N, K, alpha, beta, dtype):
    mat1 = torch.randn((M, K), dtype=dtype, device="musa")
    mat2 = torch.randn((K, N), dtype=dtype, device="musa")
    bias = torch.randn((N,), dtype=dtype, device="musa")
    ref_mat1 = to_reference(mat1, True)
    ref_mat2 = to_reference(mat2, True)
    ref_bias = to_reference(bias, True)

    ref_out = torch.addmm(ref_bias, ref_mat1, ref_mat2, alpha=alpha, beta=beta)
    res_out = torch.addmm(bias, mat1, mat2, alpha=alpha, beta=beta)

    gems_assert_close(res_out, ref_out, dtype, reduce_dim=K)

@pytest.mark.parametrize("M", MNK_SHAPES)
@pytest.mark.parametrize("N", MNK_SHAPES)
@pytest.mark.parametrize("K", MNK_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_bmm(M, N, K, dtype):
    batch = 4
    mat1 = torch.randn((batch, M, K), dtype=dtype, device="musa")
    mat2 = torch.randn((batch, K, N), dtype=dtype, device="musa")
    ref_mat1 = to_reference(mat1, True)
    ref_mat2 = to_reference(mat2, True)

    ref_out = torch.bmm(ref_mat1, ref_mat2)
    res_out = torch.bmm(mat1, mat2)

    gems_assert_close(res_out, ref_out, dtype, reduce_dim=K)


@pytest.mark.parametrize("M", MNK_SHAPES)
@pytest.mark.parametrize("N", MNK_SHAPES)
@pytest.mark.parametrize("K", MNK_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_mm(M, N, K, dtype):
    mat1 = torch.randn((M, K), dtype=dtype, device="musa")
    mat2 = torch.randn((K, N), dtype=dtype, device="musa")
    ref_mat1 = to_reference(mat1, True)
    ref_mat2 = to_reference(mat2, True)

    ref_out = torch.mm(ref_mat1, ref_mat2)
    res_out = torch.mm(mat1, mat2)

    gems_assert_close(res_out, ref_out, dtype, reduce_dim=K)


@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_cumsum(shape, dtype):
    dim = 1
    inp = torch.randn(shape, dtype=dtype, device="musa")
    ref_inp = to_reference(inp, True)

    ref_out = torch.cumsum(ref_inp, dim=dim)
    res_out = torch.cumsum(inp, dim=dim)

    gems_assert_close(res_out, ref_out, dtype, reduce_dim=shape[dim])


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_gelu(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device="musa")
    ref_inp = to_reference(inp, True)

    ref_out = torch.nn.functional.gelu(ref_inp)
    res_out = torch.nn.functional.gelu(inp)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_relu(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device="musa", requires_grad=True)
    ref_inp = to_reference(inp, True)

    ref_out = torch.nn.functional.relu(ref_inp)
    res_out = torch.nn.functional.relu(inp)

    gems_assert_close(res_out, ref_out, dtype)

    out_grad = torch.randn_like(inp)
    ref_grad = to_reference(out_grad, True)

    (ref_in_grad,) = torch.autograd.grad(ref_out, ref_inp, ref_grad)
    (res_in_grad,) = torch.autograd.grad(res_out, inp, out_grad)
    gems_assert_close(res_in_grad, ref_in_grad, dtype)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_silu(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device="musa", requires_grad=True)
    ref_inp = to_reference(inp, True)

    ref_out = torch.nn.functional.silu(ref_inp)
    res_out = torch.nn.functional.silu(inp)

    gems_assert_close(res_out, ref_out, dtype)

    out_grad = torch.randn_like(inp)
    ref_grad = to_reference(out_grad, True)

    (ref_in_grad,) = torch.autograd.grad(ref_out, ref_inp, ref_grad)
    (res_in_grad,) = torch.autograd.grad(res_out, inp, out_grad)
    gems_assert_close(res_in_grad, ref_in_grad, dtype)


@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_softmax(shape, dtype):
    dim = 1
    inp = torch.randn(shape, dtype=dtype, device="musa", requires_grad=True)
    ref_inp = to_reference(inp, True)

    ref_out = torch.nn.functional.softmax(ref_inp, dim=dim)
    res_out = torch.nn.functional.softmax(inp, dim=dim)
    gems_assert_close(res_out, ref_out, dtype)

    out_grad = torch.randn_like(inp)
    ref_grad = to_reference(out_grad, True)

    (ref_in_grad,) = torch.autograd.grad(ref_out, ref_inp, ref_grad)
    (res_in_grad,) = torch.autograd.grad(res_out, inp, out_grad)
    gems_assert_close(res_in_grad, ref_in_grad, dtype, reduce_dim=shape[dim])


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("diagonal", [-3, -1, 0, 1, 3])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_triu(shape, diagonal, dtype):
    inp = torch.randn(shape, dtype=dtype, device="musa")
    ref_inp = to_reference(inp, True)

    ref_out = torch.triu(ref_inp, diagonal)
    res_out = torch.triu(inp, diagonal)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_abs(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device="musa")
    ref_inp = to_reference(inp, True)

    ref_out = torch.abs(ref_inp)
    res_out = torch.abs(inp)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("scalar", SCALARS)
@pytest.mark.parametrize("alpha", SCALARS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_add_tensor_scalar(shape, scalar, alpha, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device="musa")
    inp2 = scalar
    ref_inp1 = to_reference(inp1, True)

    ref_out = torch.add(ref_inp1, inp2, alpha=alpha)
    res_out = torch.add(inp1, inp2, alpha=alpha)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_div(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device="musa")
    inp2 = torch.randn(shape, dtype=dtype, device="musa")
    ref_inp1 = to_reference(inp1, True)
    ref_inp2 = to_reference(inp2, True)

    ref_out = torch.div(ref_inp1, ref_inp2)
    res_out = torch.div(inp1, inp2)

    gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("scalar", SCALARS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_div_tensor_scalar(shape, scalar, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device="musa")
    inp2 = scalar
    ref_inp1 = to_reference(inp1, True)

    ref_out = torch.div(ref_inp1, inp2)
    res_out = torch.div(inp1, inp2)

    gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("scalar", SCALARS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_div_scalar_tensor(shape, scalar, dtype):
    inp1 = scalar
    inp2 = torch.randn(shape, dtype=dtype, device="musa")
    ref_inp2 = to_reference(inp2, True)

    ref_out = torch.div(inp1, ref_inp2)
    res_out = torch.div(inp1, inp2)

    gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_exp(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device="musa")
    ref_inp = to_reference(inp, True)

    ref_out = torch.exp(ref_inp)
    res_out = torch.exp(inp)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_mul(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device="musa")
    inp2 = torch.randn(shape, dtype=dtype, device="musa")
    ref_inp1 = to_reference(inp1, True)
    ref_inp2 = to_reference(inp2, True)

    ref_out = torch.mul(ref_inp1, ref_inp2)
    res_out = torch.mul(inp1, inp2)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("scalar", SCALARS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_mul_tensor_scalar(shape, scalar, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device="musa")
    inp2 = scalar
    ref_inp1 = to_reference(inp1, True)

    ref_out = torch.mul(ref_inp1, inp2)
    res_out = torch.mul(inp1, inp2)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("scalar", SCALARS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_mul_scalar_tensor(shape, scalar, dtype):
    inp1 = scalar
    inp2 = torch.randn(shape, dtype=dtype, device="musa")
    ref_inp2 = to_reference(inp2, True)

    ref_out = torch.mul(inp1, ref_inp2)
    res_out = torch.mul(inp1, inp2)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_pow(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device="musa")
    inp2 = torch.randn(shape, dtype=dtype, device="musa")
    ref_inp1 = to_reference(inp1, True)
    ref_inp2 = to_reference(inp2, True)

    ref_out = torch.pow(ref_inp1, ref_inp2)
    res_out = torch.pow(inp1, inp2)

    gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_reciprocal(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device="musa")
    ref_inp = to_reference(inp, True)

    ref_out = torch.reciprocal(ref_inp)
    res_out = torch.reciprocal(inp)

    gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_rsqrt(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device="musa")
    ref_inp = to_reference(inp, True)

    ref_out = torch.rsqrt(ref_inp)
    res_out = torch.rsqrt(inp)

    gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("scalar", SCALARS)
@pytest.mark.parametrize("alpha", SCALARS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_sub_tensor_scalar(shape, scalar, alpha, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device="musa")
    inp2 = scalar
    ref_inp1 = to_reference(inp1, True)

    ref_out = torch.sub(ref_inp1, inp2, alpha=alpha)
    res_out = torch.sub(inp1, inp2, alpha=alpha)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_mean(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device="musa")
    ref_inp = to_reference(inp, True)

    ref_out = torch.mean(ref_inp)
    res_out = torch.mean(inp)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dim", DIMS_LIST)
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_mean_dim(shape, dim, keepdim, dtype):
    inp = torch.randn(shape, dtype=dtype, device="musa")
    ref_inp = to_reference(inp, True)

    ref_out = torch.mean(ref_inp, dim, keepdim)
    res_out = torch.mean(inp, dim, keepdim)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.parametrize("M", MNK_SHAPES)
@pytest.mark.parametrize("N", MNK_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_mv(M, N, dtype):
    matrix = torch.randn((N, M), dtype=dtype, device="musa")
    vector = torch.randn((M,), dtype=dtype, device="musa")
    ref_matrix = to_reference(matrix, True)
    ref_vector = to_reference(vector, True)

    ref_out = torch.mv(ref_matrix, ref_vector)
    res_out = torch.mv(matrix, vector)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + [torch.bool])
@pytest.mark.parametrize("kind", ["normal", "allTrue"])
def test_accuracy_all(shape, dtype, kind):
    if kind == "allTrue":
        inp = torch.ones(shape, dtype=dtype, device="musa")
    else:
        inp = torch.randint(0, 2, shape, dtype=dtype, device="musa")
    ref_inp = to_reference(inp, True)

    ref_out = torch.all(ref_inp)
    res_out = torch.all(inp)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + [torch.bool])
@pytest.mark.parametrize("kind", ["normal", "allFalse"])
def test_accuracy_any(shape, dtype, kind):
    if kind == "allFalse":
        inp = torch.zeros(shape, dtype=dtype, device="musa")
    else:
        inp = torch.randint(0, 2, shape, dtype=dtype, device="musa")
    ref_inp = to_reference(inp, True)

    ref_out = torch.any(ref_inp)
    res_out = torch.any(inp)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", INT_DTYPES)
def test_accuracy_bitwiseand(shape, dtype):
    inp1 = torch.randint(
        low=-0x7FFF, high=0x7FFF, size=shape, dtype=dtype, device="musa"
    )
    inp2 = torch.randint(
        low=-0x7FFF, high=0x7FFF, size=shape, dtype=dtype, device="musa"
    )
    ref_inp1 = to_reference(inp1)
    ref_inp2 = to_reference(inp2)

    ref_out = torch.bitwise_and(ref_inp1, ref_inp2)
    res_out = torch.bitwise_and(inp1, inp2)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", INT_DTYPES)
def test_accuracy_bitwiseand_scalar(shape, dtype):
    inp1 = torch.randint(
        low=-0x7FFF, high=0x7FFF, size=shape, dtype=dtype, device="musa"
    )
    inp2 = 0x00FF
    ref_inp1 = to_reference(inp1)

    ref_out = torch.bitwise_and(ref_inp1, inp2)
    res_out = torch.bitwise_and(inp1, inp2)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", INT_DTYPES)
def test_accuracy_bitwiseand_scalar_tensor(shape, dtype):
    inp1 = 0x00FF
    inp2 = torch.randint(
        low=-0x7FFF, high=0x7FFF, size=shape, dtype=dtype, device="musa"
    )
    ref_inp2 = to_reference(inp2)

    ref_out = torch.bitwise_and(inp1, ref_inp2)
    res_out = torch.bitwise_and(inp1, inp2)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", INT_DTYPES)
def test_accuracy_bitwisenot(shape, dtype):
    inp = torch.randint(
        low=-0x7FFF, high=0x7FFF, size=shape, dtype=dtype, device="musa"
    )
    ref_inp = to_reference(inp)

    ref_out = torch.bitwise_not(ref_inp)
    res_out = torch.bitwise_not(inp)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", INT_DTYPES)
def test_accuracy_bitwiseor(shape, dtype):
    inp1 = torch.randint(
        low=-0x7FFF, high=0x7FFF, size=shape, dtype=dtype, device="musa"
    )
    inp2 = torch.randint(
        low=-0x7FFF, high=0x7FFF, size=shape, dtype=dtype, device="musa"
    )
    ref_inp1 = to_reference(inp1)
    ref_inp2 = to_reference(inp2)

    ref_out = torch.bitwise_or(ref_inp1, ref_inp2)
    res_out = torch.bitwise_or(inp1, inp2)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", INT_DTYPES)
def test_accuracy_bitwiseor_scalar(shape, dtype):
    inp1 = torch.randint(
        low=-0x7FFF, high=0x7FFF, size=shape, dtype=dtype, device="musa"
    )
    inp2 = 0x00FF
    ref_inp1 = to_reference(inp1)

    ref_out = torch.bitwise_or(ref_inp1, inp2)
    res_out = torch.bitwise_or(inp1, inp2)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", INT_DTYPES)
def test_accuracy_bitwiseor_scalar_tensor(shape, dtype):
    inp1 = 0x00FF
    inp2 = torch.randint(
        low=-0x7FFF, high=0x7FFF, size=shape, dtype=dtype, device="musa"
    )
    ref_inp2 = to_reference(inp2)

    ref_out = torch.bitwise_or(inp1, ref_inp2)
    res_out = torch.bitwise_or(inp1, inp2)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_cos(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device="musa")
    ref_inp = to_reference(inp, True)

    ref_out = torch.cos(ref_inp)
    res_out = torch.cos(inp)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_eq(shape, dtype):
    inp1 = torch.randint(0, 10, shape, dtype=dtype, device="musa")
    inp2 = torch.randint(0, 10, shape, dtype=dtype, device="musa")
    ref_inp1 = to_reference(inp1, True)
    ref_inp2 = to_reference(inp2, True)

    ref_out = torch.eq(ref_inp1, ref_inp2)
    res_out = torch.eq(inp1, inp2)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_eq_scalar(shape, dtype):
    inp1 = torch.randint(0, 10, shape, dtype=dtype, device="musa")
    inp2 = 0
    ref_inp1 = to_reference(inp1, True)

    ref_out = torch.eq(ref_inp1, inp2)
    res_out = torch.eq(inp1, inp2)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_ge(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device="musa")
    inp2 = torch.randn(shape, dtype=dtype, device="musa")
    ref_inp1 = to_reference(inp1, True)
    ref_inp2 = to_reference(inp2, True)

    ref_out = torch.ge(ref_inp1, ref_inp2)
    res_out = torch.ge(inp1, inp2)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_ge_scalar(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device="musa")
    inp2 = 0
    ref_inp1 = to_reference(inp1, True)

    ref_out = torch.ge(ref_inp1, inp2)
    res_out = torch.ge(inp1, inp2)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_gt(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device="musa")
    inp2 = torch.randn(shape, dtype=dtype, device="musa")
    ref_inp1 = to_reference(inp1, True)
    ref_inp2 = to_reference(inp2, True)

    ref_out = torch.gt(ref_inp1, ref_inp2)
    res_out = torch.gt(inp1, inp2)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_gt_scalar(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device="musa")
    ref_inp1 = to_reference(inp1, True)
    inp2 = 0

    ref_out = torch.gt(ref_inp1, inp2)
    res_out = torch.gt(inp1, inp2)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_isinf(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device="musa")
    inp = torch.masked_fill(inp, inp > 1.0, -float("inf"))
    ref_inp = to_reference(inp, True)

    ref_out = torch.isinf(ref_inp)
    res_out = torch.isinf(inp)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_isnan(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device="musa")
    inp = torch.masked_fill(inp, inp > 1.0, float("nan"))
    ref_inp = to_reference(inp, True)

    ref_out = torch.isnan(ref_inp)
    res_out = torch.isnan(inp)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_le(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device="musa")
    inp2 = torch.randn(shape, dtype=dtype, device="musa")
    ref_inp1 = to_reference(inp1, True)
    ref_inp2 = to_reference(inp2, True)

    ref_out = torch.le(ref_inp1, ref_inp2)
    res_out = torch.le(inp1, inp2)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_le_scalar(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device="musa")
    inp2 = 0
    ref_inp1 = to_reference(inp1, True)

    ref_out = torch.le(ref_inp1, inp2)
    res_out = torch.le(inp1, inp2)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_lt(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device="musa")
    inp2 = torch.randn(shape, dtype=dtype, device="musa")
    ref_inp1 = to_reference(inp1, True)
    ref_inp2 = to_reference(inp2, True)

    ref_out = torch.lt(ref_inp1, ref_inp2)
    res_out = torch.lt(inp1, inp2)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_lt_scalar(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device="musa")
    inp2 = 0
    ref_inp1 = to_reference(inp1, True)

    ref_out = torch.lt(ref_inp1, inp2)
    res_out = torch.lt(inp1, inp2)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_ne(shape, dtype):
    inp1 = torch.randint(0, 10, shape, dtype=dtype, device="musa")
    inp2 = torch.randint(0, 10, shape, dtype=dtype, device="musa")
    ref_inp1 = to_reference(inp1, True)
    ref_inp2 = to_reference(inp2, True)

    ref_out = torch.ne(ref_inp1, ref_inp2)
    res_out = torch.ne(inp1, inp2)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_ne_scalar(shape, dtype):
    inp1 = torch.randint(0, 10, shape, dtype=dtype, device="musa")
    inp2 = 0
    ref_inp1 = to_reference(inp1, True)

    ref_out = torch.ne(ref_inp1, inp2)
    res_out = torch.ne(inp1, inp2)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_neg(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device="musa")
    ref_inp = to_reference(inp, True)

    ref_out = torch.neg(ref_inp)
    res_out = torch.neg(inp)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_sin(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device="musa")
    ref_inp = to_reference(inp, True)

    ref_out = torch.sin(ref_inp)
    res_out = torch.sin(inp)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dim", DIMS_LIST)
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_amax(shape, dim, keepdim, dtype):
    inp = torch.randn(shape, dtype=dtype, device="musa")
    ref_inp = to_reference(inp, True)

    ref_out = torch.amax(ref_inp, dim=dim, keepdim=keepdim)
    res_out = torch.amax(inp, dim=dim, keepdim=keepdim)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dim", DIM_LIST)
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_argmax(shape, dim, keepdim, dtype):
    inp = torch.randn(shape, dtype=dtype, device="musa")
    ref_inp = to_reference(inp, True)

    ref_out = torch.argmax(ref_inp, dim=dim, keepdim=keepdim)
    res_out = torch.argmax(inp, dim=dim, keepdim=keepdim)
    gems_assert_equal(res_out, ref_out)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("isnone", [None, "max", "min"])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_clamp_tensor(shape, isnone, dtype):
    inp = torch.randn(shape, dtype=dtype, device="musa")
    maxi = torch.randn(shape, dtype=dtype, device="musa")
    mini = torch.randn(shape, dtype=dtype, device="musa")
    if isnone == "min":
        mini = None
    elif isnone == "max":
        maxi = None
    ref_inp = to_reference(inp, True)
    ref_maxi = to_reference(maxi, True)
    ref_mini = to_reference(mini, True)

    ref_out = torch.clamp(ref_inp, min=ref_mini, max=ref_maxi)
    res_out = torch.clamp(inp, min=mini, max=maxi)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_max(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device="musa")
    ref_inp = to_reference(inp, True)

    ref_out = torch.max(ref_inp)
    res_out = torch.max(inp)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("dim", DIM_LIST)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_max_dim(shape, dim, keepdim, dtype):
    inp = torch.randn(shape, dtype=dtype, device="musa")
    ref_inp = to_reference(inp, True)

    ref_out = torch.max(ref_inp, dim=dim, keepdim=keepdim)
    res_out = torch.max(inp, dim=dim, keepdim=keepdim)
    ref_out_value, ref_out_index = ref_out
    res_out_value, res_out_index = res_out
    gems_assert_equal(res_out_index, ref_out_index)
    gems_assert_equal(res_out_value, ref_out_value)


@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_min(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device="musa")
    ref_inp = to_reference(inp, True)

    ref_out = torch.min(ref_inp)
    res_out = torch.min(inp)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dim", DIM_LIST)
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_min_dim(shape, dim, keepdim, dtype):
    inp = torch.randn(shape, dtype=dtype, device="musa")
    ref_inp = to_reference(inp, True)

    ref_out = torch.min(ref_inp, dim=dim, keepdim=keepdim)
    res_out = torch.min(inp, dim=dim, keepdim=keepdim)
    ref_out_value, ref_out_index = ref_out
    res_out_value, res_out_index = res_out
    gems_assert_equal(res_out_index, ref_out_index)
    gems_assert_equal(res_out_value, ref_out_value)


@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_prod(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device="musa")
    ref_inp = to_reference(inp, True)

    ref_out = torch.prod(ref_inp)
    res_out = torch.prod(inp)
    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dim", DIM_LIST)
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_prod_dim(shape, dim, keepdim, dtype):
    inp = torch.randn(shape, dtype=dtype, device="musa")
    ref_inp = to_reference(inp, True)

    ref_out = torch.prod(ref_inp, dim=dim, keepdim=keepdim)
    res_out = torch.prod(inp, dim=dim, keepdim=keepdim)
    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_sum(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device="musa")
    ref_inp = to_reference(inp, True)

    ref_out = torch.sum(ref_inp)
    res_out = torch.sum(inp)

    gems_assert_close(res_out, ref_out, dtype, reduce_dim=inp.numel())


@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dim", DIMS_LIST)
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_sum_dim(shape, dim, keepdim, dtype):
    inp = torch.randn(shape, dtype=dtype, device="musa")
    ref_inp = to_reference(inp, True)

    ref_out = torch.sum(ref_inp, dim=dim, keepdim=keepdim)
    res_out = torch.sum(inp, dim=dim, keepdim=keepdim)

    if isinstance(dim, int):
        dim = [dim]
    dim = [d % inp.ndim for d in dim]
    _dim = 1
    for d in dim:
        _dim *= shape[d]
    gems_assert_close(res_out, ref_out, dtype, reduce_dim=_dim)


@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_cross_entropy_loss(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device="musa", requires_grad=True)
    dim = 1
    up_limit = shape[dim] - 1
    target_shape = list(shape)
    del target_shape[dim]
    target = torch.randint(0, up_limit, target_shape, device="musa")

    ref_inp = to_reference(inp, True)
    ref_target = to_reference(target)

    criterion = torch.nn.CrossEntropyLoss()

    ref_out = criterion(ref_inp, ref_target)
    res_out = criterion(inp, target)
    gems_assert_close(res_out, ref_out, dtype)

    out_grad = torch.randn_like(res_out)
    ref_grad = to_reference(out_grad, True)

    (ref_in_grad,) = torch.autograd.grad(ref_out, ref_inp, ref_grad)
    (res_in_grad,) = torch.autograd.grad(res_out, inp, out_grad)
    gems_assert_close(res_in_grad, ref_in_grad, dtype)


@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_log_softmax(shape, dtype):
    dim = 1
    inp = torch.randn(shape, dtype=dtype, device="musa", requires_grad=True)
    ref_inp = to_reference(inp, True)

    ref_out = torch.nn.functional.log_softmax(ref_inp, dim=dim)
    res_out = torch.nn.functional.log_softmax(inp, dim=dim)
    gems_assert_close(res_out, ref_out, dtype)

    out_grad = torch.randn_like(res_out)
    ref_grad = to_reference(out_grad, True)

    (ref_in_grad,) = torch.autograd.grad(ref_out, ref_inp, ref_grad)
    (res_in_grad,) = torch.autograd.grad(res_out, inp, out_grad)
    gems_assert_close(res_in_grad, ref_in_grad, dtype, reduce_dim=shape[dim])



@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_rmsnorm(shape, dtype):
    N = shape[1]
    layer_shape = [
        N,
    ]
    inp = torch.randn(shape, dtype=dtype, device="musa")
    weight = torch.randn(layer_shape, dtype=dtype, device="musa")
    eps = 1e-5

    ref_inp = to_reference(inp, True)
    ref_weight = to_reference(weight, True)

    def _torch_rms_norm(x, weight, eps):
        variance = x.pow(2).mean(-1, keepdim=True)
        hidden_states = x * torch.rsqrt(variance + eps)
        return weight * hidden_states

    ref_out = _torch_rms_norm(ref_inp, weight=ref_weight, eps=eps)

    res_out = torch.rms_norm(inp, list(layer_shape), weight=weight, eps=eps)

    gems_assert_close(res_out, ref_out, dtype)