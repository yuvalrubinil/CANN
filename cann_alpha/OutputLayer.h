#pragma once
#include "NeuralLayer.h"
#include "LossFunc.h"

class OutputLayer: public NeuralLayer
{
private:
	Tensor* expectedVector = nullptr;
	LossFunc* lossFunc = nullptr;
public:
	int expectedOutput;
	OutputLayer(int size, std::string activationFunction, std::string weightsInitMethod, std::string lossFunction);
	void build() override;
	void calcGrads() override;
	float calcLoss();
	std::string to_string() override;
};

