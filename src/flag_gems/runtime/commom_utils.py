from enum import Enum

from .backend import backend_utils

Autograd = backend_utils.Autograd


class vendors(Enum):
    NVIDIA = 0
    CAMBRICON = 1
    METAX = 2
    ILUVATAR = 3
    MTHREADS = 4
    KUNLUNXIN = 5
    HYGON = 6
    AMD = 7
    AIPU = 8

    @classmethod
    def get_all_vendors(cls) -> dict:
        vendorDict = {}
        for member in cls:
            vendorDict[member.name.lower()] = member
        return vendorDict
