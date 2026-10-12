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

# Kunlunxin (XPU) override of i0_ (in-place).  Shares the tuned
# pointwise_dynamic Bessel-I0 kernel from i0.py and writes the result back into
# the input tensor via out0=A.
import logging

from .i0 import i0_func

logger = logging.getLogger(__name__)


def i0_(A):
    logger.debug("GEMS_KUNLUNXIN I0_")
    i0_func(A, out0=A)
    return A
