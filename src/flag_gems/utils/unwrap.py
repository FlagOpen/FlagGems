def unwrap(compiled_kernel):
    if (
        not hasattr(compiled_kernel, "ret_tensors")
        or compiled_kernel.ret_tensors is None
    ):
        return None

    if len(compiled_kernel.ret_tensors) == 1:
        return compiled_kernel.ret_tensors[0]
    return (*compiled_kernel.ret_tensors,)
