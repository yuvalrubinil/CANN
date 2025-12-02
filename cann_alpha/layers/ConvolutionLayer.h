#pragma once
#include "Layer.h"
#include "../functions/ActivationFunc.h"


class ConvolutionLayer: public Layer
{
private:
	Tensor* biases = nullptr;
	Tensor* kernels = nullptr;
	Tensor* f = nullptr;  //f = func(featureMap + bias);
	char poolMode='m';
	int poolSize = 0;
	int stride=1;
	int paddingFrame=0;
	float convLearningRate = 1.f;
	ActivationFunc* activationFunction;

	Tensor* dc_dz = nullptr;
	Tensor* dc_dk = nullptr;
	Tensor* dc_db = nullptr;
	Tensor* pooledIndices = nullptr;

	int* getPrevLayerShape();
	std::vector<int> getFeatureMapShape();
	std::vector<int> getPoolingMapShape();
public:
	ConvolutionLayer(Tensor* kernels, char poolMode, int poolSize, int stride, std::string activationFunction, float convLearningRate);
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

