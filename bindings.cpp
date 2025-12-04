#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "cann_alpha/Network.h"
#include "cann_alpha/cata/ops.cuh"
#include <iostream>

namespace py = pybind11;



PYBIND11_MODULE(cann, m) {
    m.doc() = "CUDA Accelerated Neural Networks module";

    
    // -------------------- Network --------------------
    py::class_<Network>(m, "Network")
        .def(py::init<py::list, std::string, float, int, std::string>(),
            py::arg("layers_config"),
            py::arg("cost_function"),
            py::arg("learning_rate"),
            py::arg("batch_size"),
            py::arg("optimizer")
            )

        .def("load_dataset", &Network::loadDataset,
            py::arg("dataset"))
        .def("dump_dataset", &Network::dumpDataset)
        .def("train", &Network::train,
            py::arg("dataset"),
            py::arg("epochs"))
        .def("get_output", &Network::getOutput)
        .def("predict", &Network::predict,
            py::arg("input_data_shape_tuple"))
        .def("loss", &Network::loss)
        .def("test", &Network::test,
            py::arg("dataset"))
        .def("__str__", &Network::to_string);
    
}
