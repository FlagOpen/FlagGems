#!/bin/bash
source $(dirname "${BASH_SOURCE[0]}")/common.sh

set -ex

if [ ! -d "$pytorch_dir" ]; then
  echo 'pytorch_dir is not a directory.'
  exit
fi

# patch pytorch ci to avoid sccache
patch=$(dirname "${BASH_SOURCE[0]}")/patch/build.sh 
cp -f $patch $pytorch_dir/.ci/pytorch/build.sh

# Copying from Pytorch CI
export BUILD_ENVIRONMENT="pytorch-linux-focal-cuda12.1-cudnn8-py3-gcc9"
export TORCH_CUDA_ARCH_LIST=8.0
export MAX_JOBS=8

cd $pytorch_dir
sh -c '.ci/pytorch/build.sh'