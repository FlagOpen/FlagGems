# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

from .bitwise_or import bitwise_or_scalar_

# The backend package is imported under its short name, so __name__ here is
# "_kunlunxin.ops.ior_scalar".  Derive the canonical "flag_gems.ops.<op>" name
# from it instead of hard-coding the op name: the caplog assertions in
# tests/test_ior_scalar.py (via gems_log_logger) and use_gems(record=True) both
# only see records emitted from the flag_gems logger namespace.
logger = logging.getLogger("flag_gems.ops." + __name__.rsplit(".", 1)[-1])


def ior_scalar(self, other):
    """__ior__.Scalar == bitwise_or_.Scalar; delegate to the tuned kernel."""
    logger.debug("GEMS_KUNLUNXIN IOR_SCALAR")
    return bitwise_or_scalar_(self, other)
