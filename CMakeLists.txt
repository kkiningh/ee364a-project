cmake_minimum_required(VERSION 3.10)
project(accel)

set(CMAKE_CXX_STANDARD 11)

# Set source directory
set(SOURCE_DIR "src/accel/cpp")

# Tell CMake that headers are also in SOURCE_DIR
include_directories(${SOURCE_DIR})
set(SOURCES
  "${SOURCE_DIR}/matmul.cpp")

# Add eigen
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
include_directories(${EIGEN3_INCLUDE_DIR})

# Build a pybind11 module
add_subdirectory("lib/pybind11-2.2.1")
pybind11_add_module(accel SHARED ${SOURCES} "${SOURCE_DIR}/bindings.cpp")

# Add OpenCL
find_package(OpenCL "1.2" REQUIRED)
target_link_libraries(accel PRIVATE OpenCL::OpenCL)
add_definitions(
  "-DCL_HPP_TARGET_OPENCL_VERSION=120"
  "-DCL_HPP_MINIMUM_OPENCL_VERSION=120"
)

