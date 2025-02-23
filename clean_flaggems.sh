#!/bin/bash

PYTHON_PATH=$(which python3)

rm -rf python/build
${PYTHON_PATH} setup.py clean || true
rm -rf .pytest_cache
rm -rf build_pip
rm -f compile.log

yes|$PYTHON_PATH -m pip uninstall flag_gems

# Return 0 status if all finished
exit 0
