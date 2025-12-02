#include "NeuralLayer.h"
#include "cata/ops.cuh"
#include <iostream>

ActivationFunc* loadActivationFunc(std::string funcName);

NeuralLayer::NeuralLayer(int size, std::string activationFunction, std::string weightsInitMethod): Layer(size) {
	this->activationFunction = loadActivationFunc(activationFunction);
	this->weightsInitMethod = weightsInitMethod;
}

void NeuralLayer::buildActivations() {
	if (!this->prevLayer) {
		std::stringstream ss;
		ss << "NeuralLayer cannot build without prevLayer set.";
		throw std::runtime_error(ss.str());
	}
	std::vector<int> shape = { this->size };
	this->activations = new Tensor(shape);
}

void NeuralLayer::buildParams() {
	std::vector<int> shape = { this->size };
	std::vector<int> weightsShape = { this->size, this->prevLayer->size };
	this->biases = new Tensor(shape, 0.0f);
	this->weights = new Tensor(weightsShape);
	this->params.push_back(this->weights);
	this->params.push_back(this->biases);
}

void NeuralLayer::buildGrads() {
	std::vector<int> shape = { this->size };
	std::vector<int> weightsShape = { this->size, this->prevLayer->size };
	this->dc_dw = new Tensor(weightsShape);
	this->dc_db = new Tensor(shape);
	this->dc_dz = new Tensor(shape);
	this->grads.push_back(this->dc_dw);
	this->grads.push_back(this->dc_db);
}

void NeuralLayer::build() {
	initWeights(*this->weights, this->weightsInitMethod, this->prevLayer->size, this->size);
	this->weightsTranspose = this->weights->T(); //a view
}

void NeuralLayer::calcLayer() {
	matvecCuda(*this->weights, *this->prevLayer->activations, *this->activations); //writes to activations
	plusCuda(*this->activations, *this->biases, *this->activations); //writes to activations
	this->activationFunction->func(*this->activations, *this->activations); //writes to activations
}

void NeuralLayer::calcGrads() {
	NeuralLayer* nextNeural = static_cast<NeuralLayer*>(this->nextLayer);
	matvecCuda(*nextNeural->weightsTranspose, *nextNeural->dc_dz, *this->dc_dz); //writes dc_da to dc_dz
	this->activationFunction->chainRule(*this->activations, *this->dc_dz, *this->dc_dz); //writes to dc_dz
	outerProductPlusEqualCuda(*this->dc_dz, *this->prevLayer->activations, *this->dc_dw); //prevLayer->activations are the dz_dw, write to dc_dw
	plusCuda(*this->dc_db, *this->dc_dz, *this->dc_db); 
}

void NeuralLayer::avgGrads(float batchSize) {
	divideByScalerCuda(*this->dc_dw, *this->dc_dw, batchSize);
	divideByScalerCuda(*this->dc_db, *this->dc_db, batchSize);
}

void NeuralLayer::resetGrads() {
	fillTensorCuda(*this->dc_dw, 0.0f);
	fillTensorCuda(*this->dc_db, 0.0f);
}

void NeuralLayer::subtractGrads(float learningRate) {
	subtractByScaleCuda(*this->weights, *this->dc_dw, learningRate); //writes to weights
	subtractByScaleCuda(*this->biases, *this->dc_db, learningRate); //writes to biases
}

std::string NeuralLayer::to_string() {
	return "NeuralLayer:\n" + this->activations->to_string();
}

