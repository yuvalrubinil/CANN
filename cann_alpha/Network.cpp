#include "Network.h"
#include "cata/ops.cuh"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "layers/ConvolutionLayer.h"
#include <iostream>

namespace py = pybind11;



Layer* createLayer(py::dict layerConfig, std::string lossFunction="") {
	std::string type = layerConfig["type"].cast<std::string>();
	if (type == "ConvLayer") {
		std::vector<float> kernelsValues = layerConfig["kernels"].cast<std::vector<float>>();
		std::vector<int> kernelsShape = layerConfig["kernels_shape"].cast<std::vector<int>>();
		Tensor* kernels = new Tensor(kernelsValues, kernelsShape);
		char poolMode = layerConfig["pool_mode"].cast<char>();
		int poolSize = layerConfig["pool_size"].cast<int>();
		int stride = layerConfig["stride"].cast<int>();
		std::string activationFunction = layerConfig["activation_function"].cast<std::string>();
		float convLearningRate = layerConfig["conv_lr"].cast<float>();
		return new ConvolutionLayer(kernels, poolMode, poolSize, stride, activationFunction, convLearningRate);
	}
	if (type == "InputLayer") {
		std::vector<int> inputShape = layerConfig["input_shape"].cast<std::vector<int>>();
		return new InputLayer(inputShape);
	}
	int size = layerConfig["layer_size"].cast<int>();
	std::string activationFunction = layerConfig["activation_function"].cast<std::string>();
	std::string weightsInitMethod = layerConfig["weights_init_method"].cast<std::string>();
	if (type == "NeuralLayer")
		return new NeuralLayer(size, activationFunction, weightsInitMethod);
	else if (type == "OutputLayer")
		return new OutputLayer(size, activationFunction, weightsInitMethod, lossFunction);
	else
		throw std::runtime_error("Layer type unknown.");
}


Network::Network(py::list layersConfig, std::string lossFunction, float learningRate, int batchSize, std::string optimizer) {
	this->optimizer = loadOptimizer(optimizer, &this->layers, learningRate);
	size_t layersCount = layersConfig.size();
	this->inputLayer = (InputLayer*)createLayer(layersConfig[0], lossFunction);
	layers.push_back(this->inputLayer);
	for (int i = 1; i < layersCount; ++i) {
		this->layers.push_back(createLayer(layersConfig[i], lossFunction));
		this->layers[i]->connectTo(*this->layers[i - 1]);
		this->optimizer->buildOptGrads(this->layers[i]);
	}
	this->outputLayer = (OutputLayer*)this->layers[layersCount - 1];
	this->outputVector = this->outputLayer->activations;
	this->batchSize = batchSize;
}

void Network::loadDataset(py::list dataset) {
	for (size_t i = 0; i < dataset.size(); ++i) {
		py::tuple item = dataset[i].cast<py::tuple>();
		std::vector<float> data = item[0].cast<std::vector<float>>();
		std::vector<int> shape = item[1].cast<std::vector<int>>();
		int label = item[2].cast<int>();
		this->dataset.push_back(new DataSample(new Tensor(data, shape), label));
	}
}

void Network::dumpDataset() {
	for (DataSample* sample : this->dataset)
		sample->free();
	this->dataset.clear();
}

void Network::feedForward(Tensor& input) {
	this->inputLayer->plugInput(input);
	for (Layer* layer : this->layers)
		layer->calcLayer();
}

void Network::calcGrads() {
	for (int i = (int)this->layers.size() - 1; 0 <= i; --i)
		this->layers[i]->calcGrads();
}

void Network::subtractGrads(float batchScaler) {
	for (Layer* layer : this->layers) {
		layer->avgGrads(this->batchSize);
		this->optimizer->calcOptGrads(layer);
		this->optimizer->subtractGrads(layer);
		layer->resetGrads();
	}
}

void Network::backPropagation(int expectedOutput, int* inBatchIndex) {
	this->outputLayer->expectedOutput = expectedOutput;
	this->calcGrads();
	++(*inBatchIndex);
	if (*inBatchIndex == this->batchSize) {
		this->subtractGrads(this->batchSize);
		*inBatchIndex = 0;
	}
}

void Network::train(py::list dataset, int epochs) {
	std::cout << "Loading train set..." << std::endl;
	this->dumpDataset(); //freeing memory.
	this->loadDataset(dataset); //loading train set.
	float loss = 0.f;
	int i = 0;
	int tenPrecent = (int)dataset.size() / 10;
	for (int epoch = 1; epoch <= epochs; ++epoch) {
		int j = 0;
		int inBatchIndex = 0;
		std::cout << "Epoch " << epoch << "/" << epochs << ":" << std::endl;
		for (DataSample* sample : this->dataset) {
			this->feedForward(*sample->data);
			this->backPropagation(sample->label, &inBatchIndex);
			loss += this->loss();
			++i;
			if (++j % tenPrecent == 0) std::cout << (10 * j / tenPrecent) << "% Loss: " << (loss / (float)i) << std::endl;
		}
		if (inBatchIndex && (inBatchIndex < this->batchSize)) this->subtractGrads(inBatchIndex); // subtract the leftover.
	}
}

int Network::getOutput() {
	return argmaxSmallCuda(*this->outputVector);
}

int Network::predict(py::tuple input) {
	int prediction;
	std::vector<float> data = input[0].cast<std::vector<float>>();
	std::vector<int> shape = input[1].cast<std::vector<int>>();
	Tensor* inputTensor = new Tensor(data, shape);
	this->feedForward(*inputTensor);
	prediction = this->getOutput();
	inputTensor->free();
	this->inputLayer->activations = nullptr;
	return prediction;
}

float Network::loss() {
	return this->outputLayer->calcLoss();
}

float Network::test(py::list dataset) {
	std::cout << "Loading test set..." << std::endl;
	this->dumpDataset(); //dumping the train set from memory.
	this->loadDataset(dataset); //loading test set.
	int correct = 0;
	for (DataSample* sample : this->dataset) {
		this->feedForward(*sample->data);
		int prediction = this->getOutput();
		correct += (prediction == sample->label);
	}
	return (float)correct / (float)dataset.size();
}

std::string Network::to_string() {
	std::ostringstream oss;
	for (Layer* layer : this->layers)
		if (layer) oss << layer->to_string() << "\n";
	return oss.str();
}
