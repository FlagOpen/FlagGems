#pragma once
#include <dlfcn.h>  // dladdr
#include <array>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace flag_gems::utils {
std::filesystem::path get_path_of_this_library();
std::filesystem::path get_triton_src_path();
}  // namespace flag_gems::utils
