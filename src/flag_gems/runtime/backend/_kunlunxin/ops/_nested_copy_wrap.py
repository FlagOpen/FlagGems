import os

_CPP_EXT = None


def _get_cpp_ext():
    global _CPP_EXT
    if _CPP_EXT is None:
        from torch.utils.cpp_extension import load

        _CPP_EXT = load(
            name="gems_nested_copy_wrap",
            sources=[os.path.join(os.path.dirname(__file__), "_nested_copy_wrap.cpp")],
            extra_cflags=["-O2"],
            verbose=False,
        )
    return _CPP_EXT


def _nested_copy_wrap(buffer, nested_sizes, nested_strides, storage_offsets):
    """Clone `buffer` and wrap it as a strided nested tensor in one C++ call."""
    return _get_cpp_ext().nested_copy_wrap(
        buffer, nested_sizes, nested_strides, storage_offsets
    )
