import os

has_c_extension = False
use_c_extension = False
aten_patch_list = []

try:
    from flag_gems import c_operators

    has_c_extension = True
except ImportError:
    c_operators = None  # avoid import error if c_operators is not available
    has_c_extension = False

if has_c_extension and os.environ.get("USE_C_EXTENSION", "0") == "1":
    try:
        from flag_gems import aten_patch

        aten_patch_list = aten_patch.get_registered_ops()
        use_c_extension = True
    except (ImportError, AttributeError):
        aten_patch_list = []
        use_c_extension = False
