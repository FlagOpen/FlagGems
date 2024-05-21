import torch

# This function returns the current state of the default Philox RNG in seed and offset and
# updates the next offset by adding `increment`.
def philox_cuda_seed_offset(increment, device=None):
    device = device or torch.cuda.current_device()
    gen = torch.cuda.default_generators[device]
    state_copy = gen.get_state()
    c0, c1, c2, _ = state_copy.view(torch.int32)
    seed, offset = int(c0), int(c2)
    increment = (increment + 3) // 4 * 4
    c2 += increment
    # get_state returns a new tensor, so it needs set_state to update the actual generator state.
    gen.set_state(state_copy)
    return seed, offset
