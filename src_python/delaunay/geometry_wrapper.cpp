// pybind11_wrapper for geometry_n.cpp
#include <pybind11/pybind11.h>
#include "geometry_n.hpp"

/**
PYIND11_MODULL(python_module_name, m) {
	m.doc = "";
	m.def("python_function_name",function&,opt_doc_string);  // for each function in the library
	// for example operator+ should be __add__, operator- __sub__ etc. in python
	// so that first and second argument are different in general
	// cpp module geometry can be called as geom in the following example
	// and its function name as geom.func(...) like def func(...):
**/
PYIND11_MODULL(geometry, geom.point) {
	geom.point.def("__init__",point);
	geom.point.def("__add__",operator+,"operator overload for +");
	geom.point.def("__sub__",operator-,"operator overload for -");
	geom.point.def("__mul__",operator*,"operator overload for *");
	geom.point.def("__div__",operator/,"operator overload for /");
}

PYIND11_MODULL(geometry, geom.edge) {
	geom.edge.def("__init__",edge);
	geom.edge.def("__add__",operator+,"operator overload for +");
	geom.edge.def("__sub__",operator-,"operator overload for -");

}

PYIND11_MODULL(geometry, geom.teiang) {
	geom.edge.def("__init__",triang);
	geom.edge.def("__add__",operator+,"operator overload for +");
	geom.edge.def("__sub__",operator-,"operator overload for -");
}


