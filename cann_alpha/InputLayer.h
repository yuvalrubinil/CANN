#pragma once
#include "Layer.h"

class InputLayer : public Layer {
private:
	int* inputShape;
public:
	InputLayer(std::vector<int>& inputShape);
	void plugInput(Tensor& x);
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
	int* getInputShape();
};

