import pytest
import torch
import torch.nn.functional as F

import flag_gems

from . import base, consts


class MiopenCtcLossBenchmark(base.GenericBenchmark):
    # Shapes are (T, N, C, S) for sequence length, batch, classes, max target.
    DEFAULT_SHAPES = [
        (64, 4, 32, 16),
        (256, 16, 64, 48),
        (512, 32, 64, 48),
        (1024, 32, 128, 96),
    ]
    DEFAULT_SHAPE_DESC = "T, N, C, S"

    def init_user_config(self):
        super().init_user_config()
        # ``core_shapes.yaml`` has no ``miopen_ctc_loss`` entry, so the base
        # lookup falls back to the generic 1-D ``Benchmark`` shapes. Keep the
        # CTC shapes instead.
        self.shapes = [tuple(shape) for shape in self.DEFAULT_SHAPES]
        self.shape_desc = self.DEFAULT_SHAPE_DESC

    def set_more_shapes(self):
        return []


def miopen_ctc_loss_input_fn(shape, dtype, device):
    t_steps, batch, classes, max_target = shape
    raw = torch.randn(t_steps, batch, classes, dtype=torch.float32, device=device)
    log_probs = torch.log_softmax(raw, dim=-1).to(dtype)
    input_lengths = [t_steps] * batch
    target_lengths = [max(1, max_target - (row % 5)) for row in range(batch)]
    labels = []
    for row, length in enumerate(target_lengths):
        values = (torch.arange(length, device=device, dtype=torch.long) + row) % (
            classes - 1
        )
        labels.append(values + 1)
    targets = torch.cat(labels).to(torch.int32)

    yield (
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        0,
        True,
        False,
    )


def _miopen_ctc_loss_reference(
    log_probs,
    targets,
    input_lengths,
    target_lengths,
    blank,
    deterministic,
    zero_infinity,
):
    """Reference for the ``(costs, grad)`` pair returned by ``miopen_ctc_loss``.

    ``costs`` are the per-batch negative log-likelihoods and ``grad`` is the
    gradient of their sum with respect to ``log_probs``.
    """
    work_log_probs = log_probs.float().detach().clone().requires_grad_(True)
    padded = torch.zeros(
        (work_log_probs.shape[1], max(target_lengths)),
        dtype=torch.long,
        device=work_log_probs.device,
    )
    offset = 0
    for batch, length in enumerate(target_lengths):
        padded[batch, :length] = targets[offset : offset + length].long()
        offset += length
    costs = F.ctc_loss(
        work_log_probs,
        padded,
        input_lengths,
        target_lengths,
        blank=blank,
        reduction="none",
        zero_infinity=False,
    )
    finite = torch.isfinite(costs)
    total = torch.where(finite, costs, torch.zeros_like(costs)).sum()
    grads = torch.autograd.grad(total, work_log_probs)[0]
    if zero_infinity:
        costs = torch.where(finite, costs, torch.zeros_like(costs))
    return costs, grads


@pytest.mark.miopen_ctc_loss
def test_miopen_ctc_loss():
    bench = MiopenCtcLossBenchmark(
        op_name="miopen_ctc_loss",
        input_fn=miopen_ctc_loss_input_fn,
        torch_op=_miopen_ctc_loss_reference,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.set_gems(flag_gems.miopen_ctc_loss)
    bench.run()
