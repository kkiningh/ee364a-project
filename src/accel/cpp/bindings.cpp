#include "pybind11/pybind11.h"
#include "pybind11/eigen.h"

#include "matmul.hpp"

namespace py = pybind11;

PYBIND11_MODULE(accel, m) {
    m.def("gemm4x4", &gemm4x4);
}
