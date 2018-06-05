cmake_minimum_required(VERSION 3.10)
project(accel)

set(CMAKE_CXX_STANDARD 11)

# Set source directory
set(SOURCE_DIR "src/accel/cpp")

# Tell CMake that headers are also in SOURCE_DIR
include_directories(${SOURCE_DIR})
set(SOURCES
  "${SOURCE_DIR}/matmul.cpp")

# Add pybind11
add_subdirectory("lib/pybind11-2.2.1")
pybind11_add_module(accel ${SOURCES} "${SOURCE_DIR}/bindings.cpp")
