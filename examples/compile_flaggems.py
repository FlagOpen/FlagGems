import torch

import flag_gems  # noqa: F401


@torch.library.register_fake("flag_gems::add_tensor")
def _(a, b):
    return a + b


def _setup_context(ctx, inputs, output):
    a, b = inputs
    saved_a, saved_b = None, None
    if ctx.needs_input_grad[0]:
        saved_b = b
    if ctx.needs_input_grad[1]:
        saved_a = a
    ctx.save_for_backward(saved_a, saved_b)


def maybe_reduce(grad, input):
    reduce_axes = []
    diff_rank = grad.ndim - input.ndim
    for i in range(grad.ndim):
        if i < diff_rank:
            reduce_axes.append(i)
        elif grad.shape[i] > 1 and input.shape[i - diff_rank] == 1:
            reduce_axes.append(i)
    result = torch.sum(grad, reduce_axes).view(input.shape)
    return result


def _backward(ctx, grad):
    a, b = ctx.saved_tensors
    grad_a, grad_b = None, None
    if ctx.needs_input_grad[0]:
        grad_a = maybe_reduce(grad, a)
    if ctx.needs_input_grad[1]:
        grad_b = maybe_reduce(grad, b)
    return grad_a, grad_b


# This code adds training support for the operator. You must provide us
# the backward formula for the operator and a `setup_context` function
# to save values to be used in the backward.
torch.library.register_autograd(
    "flag_gems::add_tensor", _backward, setup_context=_setup_context
)


def f(x, y):
    return torch.ops.flag_gems.add_tensor(x, y)


F = torch.compile(f)

x = torch.randn(2, 1, 3, device="cuda:1", requires_grad=True)
y = torch.randn(4, 1, device="cuda:1", requires_grad=True)
out = F(x, y)
ref = x + y
print(out)
print(ref)

loss = out.sum()
grad = torch.autograd.grad(loss, (x, y))
print(grad)


loss = ref.sum()
grad = torch.autograd.grad(loss, (x, y))
print(grad)
