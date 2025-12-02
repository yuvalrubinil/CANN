#pragma once
#include "Layer.h"
#include "ActivationFunc.h"

class NeuralLayer : public Layer {
protected:
	Tensor* biases = nullptr;
	Tensor* dc_db = nullptr;
	Tensor* weights = nullptr;
	Tensor* dc_dw = nullptr;
	ActivationFunc* activationFunction;
	std::string weightsInitMethod;

public:
	Tensor* dc_dz = nullptr;
	Tensor* weightsTranspose = nullptr; // a view

	NeuralLayer(int size, std::string activationFunction, std::string weightsInitMethod);
	void buildActivations() override;
	void buildParams() override;
	void buildGrads() override;
	void build() override;
	void calcLayer() override;
	void calcGrads() override;
	void avgGrads(float batchSize) override;
	void resetGrads() override;
	void subtractGrads(float learningRate) override;
	std::string to_string() override;
};

