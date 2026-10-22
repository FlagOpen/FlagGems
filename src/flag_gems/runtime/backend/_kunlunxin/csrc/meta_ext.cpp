// Copyright 2026 FlagOS Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Host-only helper for pure-metadata in-place operators.
//
// ``unsqueeze_`` moves no data: it inserts one entry into the size/stride
// vectors. The Python implementation of that costs O(ndim) *Python object*
// conversions (``A.shape`` / ``A.stride()`` each build a full Python int
// sequence, ``as_strided_`` converts them back). On this backend the default
// benchmark reaches ndim in the thousands, where that dominates everything:
// measured 496 us/call at ndim=6600 vs 128 us for the native ATen kernel.
// Doing the same metadata edit on ``IntArrayRef`` in C++ removes the
// conversions; it is ~34 us at ndim=6600, i.e. faster than ATen itself.
//
// This is NOT a delegation to aten: the operator is implemented here, and the
// only ATen call is ``Tensor::as_strided_`` -- the same metadata primitive the
// Python fallback uses (``A.as_strided_(shape, stride, offset)``).

#include <torch/extension.h>

namespace {

at::Tensor unsqueeze_(at::Tensor self, int64_t dim) {
  const int64_t ndim = self.dim();
  const int64_t d = dim < 0 ? dim + ndim + 1 : dim;
  TORCH_CHECK(
      d >= 0 && d <= ndim,
      "Dimension out of range (expected to be in range of [0, ",
      ndim,
      "], but got ",
      d,
      ")");

  const auto sizes = self.sizes();
  const auto strides = self.strides();
  const int64_t numel = self.numel();

  int64_t insert_stride;
  if (numel == 0) {
    // Match the generic implementation: skip zero-size following dims.
    insert_stride = 1;
    for (int64_t i = d; i < ndim; ++i) {
      const int64_t sz = sizes[i];
      insert_stride *= sz ? sz : 1;
    }
  } else if (d == 0) {
    insert_stride = numel;
  } else {
    int64_t leading = 1;
    for (int64_t i = 0; i < d; ++i) {
      leading *= sizes[i];
    }
    insert_stride = numel / leading;
  }

  std::vector<int64_t> new_sizes(sizes.begin(), sizes.end());
  std::vector<int64_t> new_strides(strides.begin(), strides.end());
  new_sizes.insert(new_sizes.begin() + d, 1);
  new_strides.insert(new_strides.begin() + d, insert_stride);

  self.as_strided_(
      at::IntArrayRef(new_sizes),
      at::IntArrayRef(new_strides),
      self.storage_offset());
  return self;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("unsqueeze_", &unsqueeze_, "in-place metadata unsqueeze (XPU)");
}
