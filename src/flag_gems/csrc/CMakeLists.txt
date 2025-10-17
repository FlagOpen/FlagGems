if(FLAGGEMS_USE_EXTERNAL_TRITON_JIT)
    find_package(TritonJIT REQUIRED CONFIG)
else()
    FetchContent_Declare(
        TritonJIT
        GIT_REPOSITORY https://github.com/wlxjhyf/libtriton_jit.git
        GIT_TAG main
    )
    FetchContent_MakeAvailable(TritonJIT)
endif()

set(LIB_DIR "$ORIGIN/lib")

# -------------------- c_operators --------------------
pybind11_add_module(c_operators cstub.cpp)
target_link_libraries(c_operators
    PUBLIC
        Torch::Torch_Python
    PRIVATE
        operators
        ${TritonJIT_LIBRARIES}
)
set_target_properties(c_operators PROPERTIES 
    INSTALL_RPATH "${LIB_DIR}:$ORIGIN/../../../"
    BUILD_WITH_INSTALL_RPATH TRUE
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/.."
)

# -------------------- aten_patch --------------------
pybind11_add_module(aten_patch aten_patch.cpp)
target_link_libraries(aten_patch
    PUBLIC
        Torch::Torch_Python
    PRIVATE
        operators
        ${TritonJIT_LIBRARIES}
)
set_target_properties(aten_patch PROPERTIES
    INSTALL_RPATH "${LIB_DIR}:$ORIGIN/../../../"
    BUILD_WITH_INSTALL_RPATH TRUE
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/.."
)

# -------------------- Install --------------------
if (FLAGGEMS_INSTALL)
    install(TARGETS c_operators aten_patch
            LIBRARY DESTINATION ".")
endif()