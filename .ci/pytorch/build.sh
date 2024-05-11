#!/bin/bash
source $(dirname "${BASH_SOURCE[0]}")/common.sh

# Copying from Pytorch CI
BUILD_ENVIRONMENT="pytorch-linux-focal-cuda12.1-cudnn8-py3-gcc9"
MAX_JOBS=8

sh -c '$pytorch_dir/.ci/pytorch/build.sh'