import glob
import os
from typing import List, Optional, Sequence

import torch
from pip._internal.metadata import get_default_environment
from setuptools import find_packages, setup
from torch.utils.cpp_extension import (
    CUDA_HOME,
    BuildExtension,
    CppExtension,
    CUDAExtension,
)

# ----------------------------- check triton -----------------------------
# NOTE: this is used to check whether pytorch-triton or triton is installed. Since
# the name for the package to be import is the name, but the names in package manager
# are different. So we check it in this way
# 1. If the triton that is installed via pytorch-triton, then it is the version that is
# dependended by pytorch. Upgrading it may break torch. Be aware of the risk!
# 2. If the triton is installed via torch, then maybe you are aware that you are using
# triton and know about the issue mentioned above,
# 3. If you have both installed, you may have already break torch, fix it before preceeding.
# 4. If neither is installed, we will install triton.


class PackageConflictError(Exception):
    """Exception that there are conflicts in installed packages."""

    def __init__(self, message):
        super().__init__(message)

    def __str__(self):
        return self.args[0]


def detect_installed_package_from_group(
    conflicting_package_names: Sequence[str],
) -> Optional[str]:
    """Detect the installed packages in a group of mutually conflicting packages."""
    names = set(conflicting_package_names)
    if len(names) < len(conflicting_package_names):
        raise ValueError(
            f"There are duplicated package names in conflicting_package_names: {conflicting_package_names}"
        )

    environment = get_default_environment()
    installed_packages: List[str] = [
        item.canonical_name for item in environment.iter_installed_distributions()
    ]

    def _is_package_installed(package_name: str) -> bool:
        return package_name in installed_packages

    installed: List[str] = []
    for name in names:
        if _is_package_installed(name):
            installed.append(name)
    if len(installed) > 1:
        raise PackageConflictError(
            f"There are more than 1 packages ({installed}) installed in the mutually "
            f"exclusive group {conflicting_package_names}. Consider fix this before going on."
        )
    if not installed:
        return None
    return installed[0]


triton_package_name = (
    detect_installed_package_from_group(("triton", "triton-nightly", "pytorch-triton"))
    or "triton"
)


# --------------------------- flagems c extension ------------------------
library_name = "flag_gems"


def get_extensions():
    debug_mode = os.getenv("DEBUG", "0") == "1"
    use_cuda = os.getenv("USE_CUDA", "1") == "1"
    if debug_mode:
        print("Compiling in debug mode")

    use_cuda = use_cuda and torch.cuda.is_available() and CUDA_HOME is not None
    extension = CUDAExtension if use_cuda else CppExtension

    extra_link_args = []
    extra_compile_args = {
        "cxx": [
            "-O3" if not debug_mode else "-O0",
            "-fdiagnostics-color=always",
        ],
        "nvcc": [
            "-O3" if not debug_mode else "-O0",
        ],
    }
    if debug_mode:
        extra_compile_args["cxx"].append("-g")
        extra_compile_args["nvcc"].append("-g")
        extra_link_args.extend(["-O0", "-g"])

    this_dir = os.path.dirname(os.path.curdir)
    src_dir = os.path.join(this_dir, "src")
    extensions_dir = os.path.join(src_dir, library_name, "csrc")
    sources = list(glob.glob(os.path.join(extensions_dir, "*.cpp")))

    # extensions_cuda_dir = os.path.join(extensions_dir, "cuda")
    # cuda_sources = list(glob.glob(os.path.join(extensions_cuda_dir, "*.cu")))

    # if use_cuda:
    #     sources += cuda_sources

    ext_modules = [
        extension(
            f"{library_name}._C",
            sources,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
    ]

    return ext_modules


# ----------------------------- Setup -----------------------------
setup(
    name="flag_gems",
    version="2.2",
    authors=[
        {"name": "Zhixin Li", "email": "strongspoon@outlook.com"},
        {"name": "Tongxin Bai", "email": "waffle.bai@gmail.com"},
        {"name": "Yuming Huang", "email": "jokmingwong@gmail.com"},
        {"name": "Feiyu Chen", "email": "iclementine@outlook.com"},
    ],
    description="FlagGems is a function library written in Triton.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.8.0",
    license="Apache Software License",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
    ],
    install_requires=[
        f"{triton_package_name}>=2.2.0",
        "torch>=2.2.0",
        "PyYAML",
    ],
    extras_require={
        "test": [
            "pytest>=7.1.0",
            "numpy>=1.26",
            "scipy>=1.14",
        ],
        "example": [
            "transformers>=4.40.2",
        ],
    },
    url="https://github.com/FlagOpen/FlagGems",
    packages=find_packages("src"),
    package_dir={"": "src"},
    include_package_data=True,  # To include non-Python files, e.g., README
    package_data={
        "flag_gems.runtime": ["**/*.yaml"],
    },
    ext_modules=get_extensions(),
    setup_requires=["setuptools"],
    cmdclass={"build_ext": BuildExtension},
)
