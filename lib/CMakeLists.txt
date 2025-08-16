find_package(Python COMPONENTS Interpreter Development REQUIRED)
if(NOT Python_INCLUDE_DIRS OR NOT Python_LIBRARIES)
    message(FATAL_ERROR "Python development files not found. Please ensure Python is installed and development headers are available.")
endif()
include_directories(${Python_INCLUDE_DIRS})

add_library(operators
            SHARED
            zeros.cpp
            utils.cpp
            sum.cpp
            max.cpp
            mm.cpp
            rms_norm.cpp
            fused_add_rms_norm.cpp
            rotary_embedding.cpp
            topk.cpp
            addmm.cpp
            nonzero.cpp
            rotary_embedding.cpp
            contiguous.cpp
            cat.cpp
            bmm.cpp
            embedding.cpp
            argmax.cpp
            fill.cpp
            softmax.cpp
            pointwise_dynamic.cpp
            exponential_.cpp
            )
target_include_directories(operators
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

target_link_libraries(operators
  PUBLIC Torch::Torch
  PRIVATE TritonJIT::triton_jit)
add_library(FlagGems::operators ALIAS operators)


if (FLAGGEMS_INSTALL)
  include(GNUInstallDirs)
  install(
    DIRECTORY "${PROJECT_SOURCE_DIR}/include/flag_gems"
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
  install(TARGETS operators
    EXPORT FlagGemsTargets
    DESTINATION ${CMAKE_INSTALL_LIBDIR})
  install(EXPORT FlagGemsTargets
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/FlagGems
    NAMESPACE FlagGems::
    FILE FlagGemsTargets.cmake)
  export(EXPORT FlagGemsTargets FILE "${PROJECT_BINARY_DIR}/FlagGemsTargets.cmake")
endif()
