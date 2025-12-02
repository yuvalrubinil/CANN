#include "OutputLayer.h"
#include "../cata/ops.cuh"
#include <iostream>

LossFunc* loadLossFunc(std::string funcName);

OutputLayer::OutputLayer(int size, std::string activationFunction, std::string weightsInitMethod, std::string lossFunction)
	: NeuralLayer(size, activationFunction, weightsInitMethod) {
	this->lossFunc = loadLossFunc(lossFunction);
	this->expectedOutput = -1;
}

void OutputLayer::build() {
	NeuralLayer::build();
	std::vector<int> shape = { this->size };
	this->expectedVector = new Tensor(shape);
}

float OutputLayer::calcLoss() {
	oneHotCuda(*this->expectedVector, expectedOutput);
	return this->lossFunc->func(*this->activations, *this->expectedVector);
}

void OutputLayer::calcGrads() {
	oneHotCuda(*this->expectedVector, expectedOutput);
	this->lossFunc->der(*this->activations, *this->expectedVector, *this->dc_dz); //write dc_da to dc_dz
	this->activationFunction->chainRule(*this->activations, *this->dc_dz, *this->dc_dz); // writes to dc_dz
	outerProductPlusEqualCuda(*this->dc_dz, *this->prevLayer->activations, *this->dc_dw); //prevLayer->activations are dz_dw, write to dc_dw
	plusCuda(*this->dc_db, *this->dc_dz, *this->dc_db);
}

std::string OutputLayer::to_string() {
	return "OutputLayer:\n" + this->activations->to_string();
}
