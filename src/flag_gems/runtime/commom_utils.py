from enum import Enum

AUTOGRAD = "Autograd"


class Autograd(Enum):
    enable = True
    unable = False

    @classmethod
    def get_all_vendors(cls):
        return [member.name for member in cls]


class vendors(Enum):
    NVIDIA = 0
    CAMBRICON = 1
    METAX = 2
    ILUVATAR = 3
    MTHREADS = 4
    KUNLUNXIN = 5
    HYGON = 6

    @classmethod
    def get_all_vendors(cls):
        return [member.name for member in cls]


vendors_map = {
    "nvidia": vendors.NVIDIA,
    "cambricon": vendors.CAMBRICON,
    "iluvatar": vendors.ILUVATAR,
    "kunlunxin": vendors.KUNLUNXIN,
    "mthreads": vendors.MTHREADS,
    "hygon": vendors.HYGON,
    "metax": vendors.METAX,
}


quick_special_cmd = {
    "nvidia": "torch.cuda",
    "cambricon": "torch.mlu",
    "mthreads": "torch.musa",
}
