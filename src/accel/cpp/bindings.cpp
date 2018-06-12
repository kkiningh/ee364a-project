#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/eigen.h"

#include "kernels.hpp"
#include "matmul.hpp"

namespace py = pybind11;

PYBIND11_MODULE(accel, m) {
    py::class_<KernelRunner>(m, "KernelRunner")
      .def(py::init<size_t>())
      .def("run", &KernelRunner::run);
}
