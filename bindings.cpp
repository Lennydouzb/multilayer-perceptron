#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "Matrix.hpp"

namespace py = pybind11;

PYBIND11_MODULE(matrix_lib, m) {
    py::enum_<e_types>(m, "e_types")
        .value("NO_TYPE", NO_TYPE)
        .value("SCALE", SCALE)
        .value("TRANSLATION", TRANSLATION)
        .value("ROTATION", ROTATION)
        .value("EMPTY", EMPTY)
        .export_values();

    py::class_<Matrix<float>>(m, "MatrixFloat")
        .def(py::init<int>())
        .def(py::init<int, int, int, std::vector<float>, int, int>(), 
             py::arg("rows"), py::arg("cols"), py::arg("type"), py::arg("list"), py::arg("firstAxis") = 0, py::arg("secondAxis") = 0)
        .def("print", &Matrix<float>::print)
        .def("getRows", &Matrix<float>::getRows)
        .def("getCols", &Matrix<float>::getCols)
        .def("add_mat", py::overload_cast<Matrix<float>&>(&Matrix<float>::add))
		.def("getData", &Matrix<float>::getData)
        .def("add_scalar", py::overload_cast<float>(&Matrix<float>::add))
        .def("multiply_mat", py::overload_cast<Matrix<float>&>(&Matrix<float>::multiply))
        .def("transpose", &Matrix<float>::transpose)
        .def("det", &Matrix<float>::det);
}
