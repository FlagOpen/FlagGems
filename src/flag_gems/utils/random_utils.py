import torch
import triton
import triton.language as tl

from ..runtime import torch_device_fn

try:
    uint_to_uniform_float = tl.uint_to_uniform_float
except AttributeError:
    # Copied from triton.language package for compatibility
    @triton.jit
    def uint_to_uniform_float(x):
        """
        Numerically stable function to convert a random uint into a random float uniformly sampled in [0, 1).
        """
        # TODO: fix frontend issues and cleanup
        # conditions can be simplified
        # scale is ((2**23 - 1) / 2**23) * 2**(N_BITS - 1)
        if tl.constexpr(x.dtype == tl.uint32) or tl.constexpr(x.dtype == tl.int32):
            # maximum value such that `MAX_INT * scale < 1.0` (with float rounding)
            x = x.to(tl.int32, bitcast=True)
            scale = 4.6566127342e-10
        else:
            tl.static_assert(
                tl.constexpr(x.dtype == tl.uint64) or tl.constexpr(x.dtype == tl.int64)
            )
            x = x.to(tl.int64, bitcast=True)
            scale = 1.0842020432385337e-19
        x = tl.where(x < 0, -x - 1, x)
        return x * scale


# This function is roughly a python wrapper of CUDAGeneratorImpl::philox_cuda_state in Pytorch.
# https://github.com/pytorch/pytorch/blob/8a4597980c2692b73f35fb3c7145eaeaf2273e77/aten/src/ATen/cuda/CUDAGeneratorImpl.cpp#L452
# It returns the current state of the default Philox RNG in seed and offset and
# updates the next offset by adding `increment`.
def philox_backend_seed_offset(increment, device=None):
    device = device or torch_device_fn.current_device()
    gen = torch_device_fn.default_generators[device]
    state_copy = gen.get_state()
    c0, c1 = state_copy.view(torch.int64)
    seed, offset = int(c0), int(c1)
    increment = (increment + 3) // 4 * 4
    c1 += increment
    # get_state returns a new tensor, so it needs set_state to update the actual generator state.
    gen.set_state(state_copy)
    return seed, offset


def per_thread_offset(N, num_blocks, num_warps, warp_threads=32):
    block_threads = num_warps * warp_threads
    max_threads = num_blocks * block_threads
    offset = (N + max_threads - 1) // max_threads
    return offset


@triton.jit
def uniform(seed, philox_offset, offset):
    seed = seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)
    i4 = offset
    c0 += i4
    _O = c0 * 0
    r0, r1, r2, r3 = tl.philox(seed, c0, c1, _O, _O)
    r0 = uint_to_uniform_float(r0)
    r1 = uint_to_uniform_float(r1)
    r2 = uint_to_uniform_float(r2)
    r3 = uint_to_uniform_float(r3)
    return r0, r1, r2, r3
