#!/bin/bash
source $(dirname "${BASH_SOURCE[0]}")/common.sh

set -ex

commit=$1

if [ -z "$commit" ]; then
  commit=`cat $(dirname "${BASH_SOURCE[0]}")/pinned_pytorch_commit.txt`
fi

if [[ `pwd` != $pytorch_dir ]]; then
  mkdir -p $pytorch_dir
  cd $pytorch_dir
fi

/usr/bin/git config --global --add safe.directory $pytorch_dir
/usr/bin/git init $pytorch_dir
/usr/bin/git remote add origin https://github.com/pytorch/pytorch
/usr/bin/git config --local gc.auto 0
/usr/bin/git -c protocol.version=2 fetch --prune --progress --no-recurse-submodules --quiet origin +refs/heads/*:refs/remotes/origin/* +refs/tags/*:refs/tags/*
/usr/bin/git rev-parse --verify --quiet $commit^{object}
/usr/bin/git checkout --quiet --force $commit

# Check out submodules
/usr/bin/git submodule sync --recursive
/usr/bin/git -c protocol.version=2 submodule update --init --force --recursive
/usr/bin/git submodule foreach --recursive git config --local gc.auto 0
/usr/bin/git log -1 --format='%H'
