# /bin/bash
SW_HOME=${SW_HOME:-/opt/sw_home}
COREX_VERSION=${COREX_VERSION:-latest}

export COREX_ARCH=${COREX_ARCH:-ivcore11}
export LLVM_SYSPATH=${SW_HOME}/sdk/ixcc/build
export PYBIND11_SYSPATH=./pybind11/pybind11-2.11.1
export CUDA_HOME=${SW_HOME}/local/corex

if [[ "${COREX_VERSION}" == "latest" ]]; then
  COREX_VERSION=`date --utc +%Y%m%d%H%M%S`
fi
export GEMS_LOCAL_VERSION_IDENTIFIER="corex.${COREX_VERSION}"

PYTHON_PATH=$(which python3)
${PYTHON_PATH} setup.py bdist_wheel -d ./build_pip 2>&1 | tee ./compile.log
exit 0
