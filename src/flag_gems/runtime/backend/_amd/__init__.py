from backend_utils import VendorInfoBase  # noqa: E402

vendor_info = VendorInfoBase(
    vendor_name="amd", device_name="cuda", device_query_cmd="amd-smi"
)

CUSTOMIZED_UNUSED_OPS = ()


__all__ = ["*"]
