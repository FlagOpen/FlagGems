import pytest
import torch
import torch.nn.functional as F

import flag_gems

from . import accuracy_utils as utils

# (T, N, C, target_lengths) model-relevant CTC shapes.
MIOPEN_CTC_SHAPES = [
    (32, 4, 16, [8, 6, 10, 4]),
    (64, 8, 32, [16, 12, 15, 8, 24, 20, 11, 9]),
    (128, 2, 64, [48, 32]),
]


def _make_targets(target_lengths, classes, device):
    """Build 1D concatenated int32 targets plus the padded 2D long layout."""
    pieces = []
    padded = torch.zeros(
        len(target_lengths), max(target_lengths), dtype=torch.long, device=device
    )
    for batch, length in enumerate(target_lengths):
        labels = torch.randint(1, classes, (length,), device=device, dtype=torch.long)
        pieces.append(labels)
        padded[batch, :length] = labels
    if pieces:
        concatenated = torch.cat(pieces).to(torch.int32)
    else:
        concatenated = torch.empty(0, dtype=torch.int32, device=device)
    return concatenated, padded


def _reference_ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=0):
    """Per-batch negative log-likelihoods and d(sum)/d(log_probs) in fp32."""
    work_log_probs = log_probs.float().detach().clone().requires_grad_(True)
    nll = F.ctc_loss(
        work_log_probs,
        targets,
        input_lengths,
        target_lengths,
        blank=blank,
        reduction="none",
        zero_infinity=False,
    )
    finite = torch.isfinite(nll)
    # Unreachable alignments give an infinite cost whose gradient is undefined,
    # so mask those batches out of the gradient reference instead of producing
    # non-finite values.
    total = torch.where(finite, nll, torch.zeros_like(nll)).sum()
    grads = torch.autograd.grad(total, work_log_probs)[0]
    return nll, grads, finite


@pytest.mark.miopen_ctc_loss
@pytest.mark.parametrize("shape", MIOPEN_CTC_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_miopen_ctc_loss(shape, dtype):
    t_steps, batch, classes, target_lengths = shape
    res_log_probs = torch.log_softmax(
        torch.randn(
            t_steps, batch, classes, dtype=torch.float32, device=flag_gems.device
        ),
        dim=-1,
    ).to(dtype)
    res_targets, ref_targets = _make_targets(target_lengths, classes, flag_gems.device)
    input_lengths = [t_steps] * batch

    ref_log_probs = utils.to_reference(res_log_probs, True)
    ref_targets = utils.to_reference(ref_targets, True)
    ref_costs, ref_grads, finite = _reference_ctc_loss(
        ref_log_probs, ref_targets, input_lengths, target_lengths
    )

    with flag_gems.use_gems():
        res_costs, res_grads = torch.ops.aten.miopen_ctc_loss(
            res_log_probs,
            res_targets,
            input_lengths,
            target_lengths,
            0,
            True,
            False,
        )

    assert res_costs.shape == ref_costs.shape
    assert res_grads.shape == ref_log_probs.shape

    utils.gems_assert_close(res_costs[finite], ref_costs[finite].to(dtype), dtype)
    grad_mask = finite.view(1, -1, 1).expand_as(ref_grads)
    utils.gems_assert_close(res_grads[grad_mask], ref_grads[grad_mask].to(dtype), dtype)


@pytest.mark.miopen_ctc_loss
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_miopen_ctc_loss_zero_infinity(dtype):
    """Infinite losses report zero cost and zero gradient when requested."""
    t_steps, batch, classes = 8, 3, 7
    # A target longer than the input makes the alignment unreachable -> inf loss.
    target_lengths = [4, 9, 2]
    res_log_probs = torch.log_softmax(
        torch.randn(
            t_steps, batch, classes, dtype=torch.float32, device=flag_gems.device
        ),
        dim=-1,
    ).to(dtype)
    res_targets, _ = _make_targets(target_lengths, classes, flag_gems.device)
    input_lengths = [t_steps] * batch

    with flag_gems.use_gems():
        res_costs, res_grads = torch.ops.aten.miopen_ctc_loss(
            res_log_probs,
            res_targets,
            input_lengths,
            target_lengths,
            0,
            True,
            True,
        )

    assert torch.isfinite(res_costs).all()
    assert torch.isfinite(res_grads).all()
    assert res_costs[1].item() == 0.0
    assert res_grads[:, 1, :].abs().sum().item() == 0.0


@pytest.mark.miopen_ctc_loss
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_miopen_ctc_loss_empty_target(dtype):
    """A zero-length target is valid and must agree with the reference."""
    t_steps, batch, classes = 16, 4, 12
    target_lengths = [5, 0, 3, 0]
    res_log_probs = torch.log_softmax(
        torch.randn(
            t_steps, batch, classes, dtype=torch.float32, device=flag_gems.device
        ),
        dim=-1,
    ).to(dtype)
    res_targets, ref_targets = _make_targets(target_lengths, classes, flag_gems.device)
    input_lengths = [t_steps] * batch

    ref_log_probs = utils.to_reference(res_log_probs, True)
    ref_costs, ref_grads, finite = _reference_ctc_loss(
        ref_log_probs,
        utils.to_reference(ref_targets, True),
        input_lengths,
        target_lengths,
    )

    with flag_gems.use_gems():
        res_costs, res_grads = torch.ops.aten.miopen_ctc_loss(
            res_log_probs,
            res_targets,
            input_lengths,
            target_lengths,
            0,
            True,
            False,
        )

    assert finite.all()
    utils.gems_assert_close(res_costs, ref_costs.to(dtype), dtype)
    utils.gems_assert_close(res_grads, ref_grads.to(dtype), dtype)


@pytest.mark.miopen_ctc_loss
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_miopen_ctc_loss_tensor_lengths(dtype):
    """Tensor-typed lengths take the same path as Python sequences."""
    t_steps, batch, classes = 24, 3, 9
    target_lengths = [7, 4, 9]
    res_log_probs = torch.log_softmax(
        torch.randn(
            t_steps, batch, classes, dtype=torch.float32, device=flag_gems.device
        ),
        dim=-1,
    ).to(dtype)
    res_targets, ref_targets = _make_targets(target_lengths, classes, flag_gems.device)
    input_lengths = torch.full(
        (batch,), t_steps, dtype=torch.int64, device=flag_gems.device
    )
    target_lengths_t = torch.tensor(
        target_lengths, dtype=torch.int64, device=flag_gems.device
    )

    ref_log_probs = utils.to_reference(res_log_probs, True)
    ref_costs, ref_grads, finite = _reference_ctc_loss(
        ref_log_probs,
        utils.to_reference(ref_targets, True),
        [t_steps] * batch,
        target_lengths,
    )

    with flag_gems.use_gems():
        res_costs, res_grads = flag_gems.miopen_ctc_loss(
            res_log_probs,
            res_targets,
            input_lengths,
            target_lengths_t,
            0,
            True,
            False,
        )

    assert finite.all()
    utils.gems_assert_close(res_costs, ref_costs.to(dtype), dtype)
    utils.gems_assert_close(res_grads, ref_grads.to(dtype), dtype)


@pytest.mark.miopen_ctc_loss
def test_miopen_ctc_loss_invalid_rank():
    """The operator rejects non-3D and non-integer-target inputs."""
    log_probs = torch.log_softmax(torch.randn(8, 4, device=flag_gems.device), dim=-1)
    targets = torch.empty(0, dtype=torch.int32, device=flag_gems.device)
    with flag_gems.use_gems():
        with pytest.raises(RuntimeError):
            torch.ops.aten.miopen_ctc_loss(log_probs, targets, [8], [0], 0, True, False)
