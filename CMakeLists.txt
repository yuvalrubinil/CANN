cmake_minimum_required(VERSION 3.22)
project(CANN LANGUAGES CXX CUDA)

# -----------------------------
# C++ and CUDA standards
# -----------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# -----------------------------
# Find Python
# -----------------------------
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# -----------------------------
# Fetch pybind11
# -----------------------------
include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.11.0
)
FetchContent_MakeAvailable(pybind11)

# -----------------------------
# Source files
# -----------------------------
set(SOURCES
    cann_alpha/Network.cpp
    cann_alpha/layers/Layer.cpp
    cann_alpha/layers/InputLayer.cpp
    cann_alpha/layers/OutputLayer.cpp
    cann_alpha/layers/NeuralLayer.cpp
    cann_alpha/functions/Activations.cpp
    cann_alpha/functions/Losses.cpp
    cann_alpha/layers/ConvolutionLayer.cpp
    cann_alpha/optimizers/Optimizer.cpp
    cann_alpha/optimizers/Optimizers.cpp
    bindings.cpp
)

set(CUDA_SOURCES
    cann_alpha/cata/ops/activation_ops.cu
    cann_alpha/cata/ops/convolution_ops.cu
    cann_alpha/cata/ops/loss_ops.cu
    cann_alpha/cata/ops/matrix_ops.cu
    cann_alpha/cata/ops/memory_ops.cu
    cann_alpha/cata/ops/vector_ops.cu
    cann_alpha/cata/ops/weights_init.cu
    cann_alpha/cata/tensor/tensor.cu
    cann_alpha/cata/ops/optimizers.cu
)

# -----------------------------
# Build Python module "cann"
# -----------------------------
pybind11_add_module(cann
    ${SOURCES}
    ${CUDA_SOURCES}
)

# -----------------------------
# Include directories
# -----------------------------
target_include_directories(cann PRIVATE
    ${Python3_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/cann_alpha
    ${CMAKE_CURRENT_SOURCE_DIR}/cann_alpha/cata
)

# -----------------------------
# Link Python & pybind11
# -----------------------------
target_link_libraries(cann PRIVATE
    ${Python3_LIBRARIES}
    pybind11::module
)

# -----------------------------
# CUDA build properties
# -----------------------------
set_target_properties(cann PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# -----------------------------
# Output the .pyd inside cann_alpha
# -----------------------------
set_target_properties(cann PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/cann_alpha
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/cann_alpha
)

# -----------------------------
# Windows Python extension fix
# -----------------------------
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

# -----------------------------
# CUDA architectures
# -----------------------------
set(CMAKE_CUDA_ARCHITECTURES native)

# -----------------------------
# Install targets for scikit-build wheel
# -----------------------------
install(TARGETS cann LIBRARY DESTINATION cann_alpha)
install(DIRECTORY cann_alpha/ DESTINATION cann_alpha FILES_MATCHING PATTERN "*.py")
