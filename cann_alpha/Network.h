#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "cata/tensor.cuh"
#include "layers/Layer.h"
#include "layers/InputLayer.h"
#include "layers/OutputLayer.h"
#include "optimizers/Optimizer.h"

namespace py = pybind11;

Optimizer* loadOptimizer(std::string optName, std::vector<Layer*>* layers, float learningRate, float beta = 0.9f);

typedef struct DataSample {
	Tensor* data;
	int label;

	DataSample(Tensor* data, int label)
		: data(data), label(label) {
	}

	~DataSample() {
		data->free();
	}

	void free() {
		this->~DataSample();
	}
};


class Network
{

private:
	std::vector<DataSample*> dataset;
	std::vector<Layer*> layers;
	InputLayer* inputLayer = nullptr;
	OutputLayer* outputLayer = nullptr;
	Tensor* outputVector = nullptr;
	Optimizer* optimizer = nullptr;
	int batchSize = 32;

	void calcGrads();
	void subtractGrads(float batchScaler);

public:
	Network(py::list layers_config, std::string cost_function, float learning_rate, int batchSize, std::string optimizer);
	void loadDataset(py::list dataset);
	void dumpDataset();
	void feedForward(Tensor& input);
	void backPropagation(int expectedOutput, int* batchIndex);
	void train(py::list dataset, int epochs);
	int getOutput();
	int predict(py::tuple input);
	float loss();
	float test(py::list dataset);

	std::string to_string();

};
