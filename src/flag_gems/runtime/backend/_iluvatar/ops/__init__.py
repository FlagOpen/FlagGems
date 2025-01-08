from backend_utils import Autograd

from . import bmm, mm


def get_specific_ops():
    return (
        ("mm", mm.mm, Autograd.enable),
        ("bmm", bmm.bmm, Autograd.enable),
    )


def get_unused_ops():
    return ()


__all__ = ["get_specific_ops", "get_unused_ops"]
