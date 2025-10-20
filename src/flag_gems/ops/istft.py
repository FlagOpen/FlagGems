import math

import torch
import triton
import triton.language as tl


@triton.jit
def _ifft_onesided_kernel(
    spec_real_ptr,
    spec_imag_ptr,
    frame_real_ptr,
    stride_freq,
    stride_frame,
    scale,
    frame_idx,
    n_fft,
    BLOCK_T: tl.constexpr,
    N_FREQS: tl.constexpr,
):
    # 在 kernel 内定义常量
    TWO_PI = 6.283185307179586  # 2 * pi

    pid = tl.program_id(0)
    time_idx = pid * BLOCK_T + tl.arange(0, BLOCK_T)
    mask = time_idx < n_fft
    acc = tl.zeros((BLOCK_T,), dtype=tl.float32)

    t_float = time_idx.to(tl.float32)
    n_fft_inv = 1.0 / n_fft

    for freq in range(N_FREQS):
        offset = freq * stride_freq + frame_idx * stride_frame
        real_val = tl.load(spec_real_ptr + offset)
        imag_val = tl.load(spec_imag_ptr + offset)
        angle = (TWO_PI * freq) * t_float * n_fft_inv
        cos_val = tl.cos(angle)
        sin_val = tl.sin(angle)
        contrib = real_val * cos_val - imag_val * sin_val
        if (freq == 0) or (freq == N_FREQS - 1):
            acc += contrib
        else:
            acc += 2.0 * contrib

    acc = acc * scale
    # 转换回原始 dtype
    output_dtype = spec_real_ptr.dtype.element_ty
    acc = acc.to(output_dtype)
    tl.store(frame_real_ptr + time_idx, acc, mask=mask)


@triton.jit
def _ifft_full_kernel(
    spec_real_ptr,
    spec_imag_ptr,
    frame_real_ptr,
    frame_imag_ptr,
    stride_freq,
    stride_frame,
    scale,
    frame_idx,
    n_fft,
    BLOCK_T: tl.constexpr,
    N_FREQS: tl.constexpr,
):
    # 在 kernel 内定义常量
    TWO_PI = 6.283185307179586  # 2 * pi

    pid = tl.program_id(0)
    time_idx = pid * BLOCK_T + tl.arange(0, BLOCK_T)
    mask = time_idx < n_fft
    acc_real = tl.zeros((BLOCK_T,), dtype=tl.float32)
    acc_imag = tl.zeros((BLOCK_T,), dtype=tl.float32)

    t_float = time_idx.to(tl.float32)
    n_fft_inv = 1.0 / n_fft

    for freq in range(N_FREQS):
        offset = freq * stride_freq + frame_idx * stride_frame
        real_val = tl.load(spec_real_ptr + offset)
        imag_val = tl.load(spec_imag_ptr + offset)
        angle = (TWO_PI * freq) * t_float * n_fft_inv
        cos_val = tl.cos(angle)
        sin_val = tl.sin(angle)
        acc_real += real_val * cos_val - imag_val * sin_val
        acc_imag += real_val * sin_val + imag_val * cos_val

    acc_real = acc_real * scale
    acc_imag = acc_imag * scale
    # 转换回原始 dtype
    output_dtype = spec_real_ptr.dtype.element_ty
    acc_real = acc_real.to(output_dtype)
    acc_imag = acc_imag.to(output_dtype)
    tl.store(frame_real_ptr + time_idx, acc_real, mask=mask)
    tl.store(frame_imag_ptr + time_idx, acc_imag, mask=mask)


@triton.jit
def _overlap_add_kernel(
    frame_real_ptr,
    frame_imag_ptr,
    output_real_ptr,
    output_imag_ptr,
    envelope_ptr,
    window_ptr,
    frame_offset,
    win_length,
    output_length,
    BLOCK_T: tl.constexpr,
    APPLY_WINDOW: tl.constexpr,
    HAS_IMAG: tl.constexpr,
):
    pid = tl.program_id(0)
    local_idx = pid * BLOCK_T + tl.arange(0, BLOCK_T)
    mask = local_idx < win_length
    output_idx = frame_offset + local_idx
    mask = mask & (output_idx < output_length)
    # 注意：Triton 没有 tl.all()，删除这个优化检查
    # if tl.all(~mask):
    #     return

    frame_real = tl.load(frame_real_ptr + local_idx, mask=mask, other=0.0)
    window_vals = tl.zeros((BLOCK_T,), dtype=frame_real.dtype) + 1.0
    if APPLY_WINDOW:
        window_vals = tl.load(window_ptr + local_idx, mask=mask, other=0.0)

    frame_real = frame_real * window_vals
    tl.atomic_add(output_real_ptr + output_idx, frame_real, mask=mask)
    tl.atomic_add(envelope_ptr + output_idx, window_vals * window_vals, mask=mask)

    if HAS_IMAG:
        frame_imag = tl.load(frame_imag_ptr + local_idx, mask=mask, other=0.0)
        frame_imag = frame_imag * window_vals
        tl.atomic_add(output_imag_ptr + output_idx, frame_imag, mask=mask)


@triton.jit
def _normalize_kernel(
    output_real_ptr,
    output_imag_ptr,
    envelope_ptr,
    length,
    BLOCK_T: tl.constexpr,
    HAS_IMAG: tl.constexpr,
):
    # 在 kernel 内定义常量
    ENVELOPE_EPS = 1e-8

    pid = tl.program_id(0)
    idx = pid * BLOCK_T + tl.arange(0, BLOCK_T)
    mask = idx < length
    env = tl.load(envelope_ptr + idx, mask=mask, other=0.0)
    inv = tl.where(env > ENVELOPE_EPS, 1.0 / env, 0.0)
    real_val = tl.load(output_real_ptr + idx, mask=mask, other=0.0)
    real_val = real_val * inv
    tl.store(output_real_ptr + idx, real_val, mask=mask)
    if HAS_IMAG:
        imag_val = tl.load(output_imag_ptr + idx, mask=mask, other=0.0)
        imag_val = imag_val * inv
        tl.store(output_imag_ptr + idx, imag_val, mask=mask)


def _make_hann_window(length, *, device, dtype):
    TWO_PI = 6.283185307179586  # 2 * pi
    n = torch.arange(length, device=device, dtype=dtype)
    return 0.5 - 0.5 * torch.cos(TWO_PI * n / length)


def istft(
    input_tensor,
    n_fft,
    hop_length=None,
    win_length=None,
    window=None,
    center=True,
    normalized=False,
    onesided=True,
    length=None,
    return_complex=False,
):
    if hop_length is None:
        hop_length = n_fft // 4
    if win_length is None:
        win_length = n_fft

    if hop_length <= 0:
        raise ValueError("hop_length must be positive")
    if win_length <= 0 or win_length > n_fft:
        raise ValueError("win_length must be in (0, n_fft]")

    if input_tensor.dim() < 2:
        raise ValueError("Input tensor must have at least 2 dimensions")

    if input_tensor.is_complex():
        spectrum = input_tensor
    else:
        if input_tensor.shape[-1] != 2:
            raise TypeError("Input tensor must be complex or have last dimension == 2")
        spectrum = torch.view_as_complex(input_tensor)

    spectrum = spectrum.contiguous()
    *batch_shape, n_freqs, n_frames = spectrum.shape
    batch_size = math.prod(batch_shape) if batch_shape else 1

    expected_freqs = n_fft // 2 + 1 if onesided else n_fft
    if n_freqs != expected_freqs:
        raise ValueError(
            f"Expected frequency dimension {expected_freqs} but got {n_freqs}"
        )

    device = spectrum.device
    real_dtype = spectrum.real.dtype
    # 支持所有 FLOAT_DTYPES (float32, float16, bfloat16)

    if window is None:
        # PyTorch 默认使用 rectangular window (全1窗口)
        window = torch.ones(win_length, device=device, dtype=real_dtype)
    else:
        if window.numel() != win_length:
            raise ValueError("window length mismatch")
        window = window.to(device=device, dtype=real_dtype)

    scale = 1.0 / math.sqrt(n_fft) if normalized else 1.0 / n_fft

    spectrum = spectrum.reshape(batch_size, n_freqs, n_frames)
    spec_real = spectrum.real.contiguous()
    spec_imag = spectrum.imag.contiguous()

    stride_freq = spec_real.stride(-2)
    stride_frame = spec_real.stride(-1)

    if center:
        full_length = hop_length * (n_frames - 1) + n_fft
        pad = n_fft // 2
    else:
        full_length = hop_length * (n_frames - 1) + win_length
        pad = 0

    needs_imag = (not onesided) or return_complex

    output_real = torch.zeros(
        (batch_size, full_length), dtype=real_dtype, device=device
    )
    output_imag = (
        torch.zeros((batch_size, full_length), dtype=real_dtype, device=device)
        if needs_imag
        else None
    )
    envelope = torch.zeros((batch_size, full_length), dtype=real_dtype, device=device)

    frame_real = torch.empty(n_fft, dtype=real_dtype, device=device)
    frame_imag = (
        torch.empty(n_fft, dtype=real_dtype, device=device) if needs_imag else None
    )

    grid_ifft = lambda meta: (triton.cdiv(n_fft, meta["BLOCK_T"]),)
    grid_overlap = lambda meta: (triton.cdiv(win_length, meta["BLOCK_T"]),)

    for b in range(batch_size):
        real_ptr = spec_real[b]
        imag_ptr = spec_imag[b]
        for frame_idx in range(n_frames):
            if onesided:
                _ifft_onesided_kernel[grid_ifft](
                    real_ptr,
                    imag_ptr,
                    frame_real,
                    stride_freq,
                    stride_frame,
                    scale,
                    frame_idx,
                    n_fft,
                    BLOCK_T=256,
                    N_FREQS=n_freqs,
                )
                if needs_imag:
                    frame_imag.zero_()
            else:
                _ifft_full_kernel[grid_ifft](
                    real_ptr,
                    imag_ptr,
                    frame_real,
                    frame_imag,
                    stride_freq,
                    stride_frame,
                    scale,
                    frame_idx,
                    n_fft,
                    BLOCK_T=256,
                    N_FREQS=n_freqs,
                )

            frame_offset = frame_idx * hop_length
            _overlap_add_kernel[grid_overlap](
                frame_real,
                frame_imag if needs_imag else frame_real,
                output_real[b],
                output_imag[b] if needs_imag else output_real[b],
                envelope[b],
                window,
                frame_offset,
                win_length,
                full_length,
                BLOCK_T=256,
                APPLY_WINDOW=True,
                HAS_IMAG=needs_imag,
            )

    grid_norm = lambda meta: (triton.cdiv(full_length, meta["BLOCK_T"]),)
    for b in range(batch_size):
        _normalize_kernel[grid_norm](
            output_real[b],
            output_imag[b] if needs_imag else output_real[b],
            envelope[b],
            full_length,
            BLOCK_T=256,
            HAS_IMAG=needs_imag,
        )

    if pad:
        output_real = output_real[..., pad:-pad]
        if needs_imag:
            output_imag = output_imag[..., pad:-pad]
        full_length = output_real.shape[-1]

    if length is not None:
        if full_length > length:
            output_real = output_real[..., :length]
            if needs_imag:
                output_imag = output_imag[..., :length]
        elif full_length < length:
            pad_size = length - full_length
            pad_shape = list(output_real.shape)
            pad_shape[-1] = pad_size
            pad_tensor = torch.zeros(pad_shape, dtype=real_dtype, device=device)
            output_real = torch.cat([output_real, pad_tensor], dim=-1)
            if needs_imag:
                output_imag = torch.cat([output_imag, pad_tensor], dim=-1)

    if batch_shape:
        output_real = output_real.reshape(*batch_shape, output_real.shape[-1])
        if needs_imag:
            output_imag = output_imag.reshape(*batch_shape, output_imag.shape[-1])
    else:
        output_real = output_real.squeeze(0)
        if needs_imag:
            output_imag = output_imag.squeeze(0)

    if return_complex:
        if needs_imag:
            return torch.complex(output_real, output_imag)
        return torch.complex(output_real, torch.zeros_like(output_real))

    if needs_imag and not onesided:
        return output_real

    return output_real
