#pragma once
#include <stdexcept>
#include "cata/tensor.cuh"
#include <vector>

class Layer {
private:
	void buildLayer();
public:
	Layer* prevLayer = nullptr;
	Layer* nextLayer = nullptr;
	Tensor* activations = nullptr;
	std::vector<Tensor*> params;
	std::vector<Tensor*> grads;
	std::vector<Tensor*> optGrads;
	int size;

	Layer(int size);
	~Layer();
	void setNextLayer(Layer& nextLayer);
	void setPrevLayer(Layer& prevLayer);
	void connectTo(Layer& prevLayer);
	virtual std::string to_string();
	virtual void buildActivations() = 0;
	virtual void buildParams() = 0;
	virtual void buildGrads() = 0;
	virtual void build() = 0;
	virtual void calcLayer() = 0;
	virtual void calcGrads() = 0;
	virtual void avgGrads(float batchSize) = 0;
	virtual void resetGrads() = 0;
	virtual void subtractGrads(float learningRate) = 0;
};
