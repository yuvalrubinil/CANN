#include "ConvolutionLayer.h"
#include "../cata/ops.cuh"
#include "NeuralLayer.h"
#include "InputLayer.h"
#include <sstream>
#include <iostream>

ActivationFunc* loadActivationFunc(std::string funcName);




ConvolutionLayer::ConvolutionLayer(Tensor* kernels, char poolMode, int poolSize, int stride, std::string activationFunction, float convLearningRate=1.f) : Layer(1) {
	this->kernels = kernels;
	this->stride = stride;
	this->poolSize = poolSize;
	this->poolMode = poolMode;
	this->activationFunction = loadActivationFunc(activationFunction);
    this->convLearningRate = convLearningRate;
}

int* ConvolutionLayer::getPrevLayerShape() {
    if (InputLayer* prevL = dynamic_cast<InputLayer*>(this->prevLayer))
        return prevL->getInputShape();
    return this->prevLayer->activations->getShape();
}

std::vector<int> ConvolutionLayer::getFeatureMapShape() {
    int* tensorShape = this->getPrevLayerShape();
    int* kernelsShape = this->kernels->getShape();
    int imgCount = tensorShape[0];
    int imgHeight = tensorShape[1];
    int imgWidth = tensorShape[2];

    int kernelsCount = kernelsShape[0];
    int kernelWidth = kernelsShape[2];
    int paddingFrame = kernelWidth / 2;

    int fmWidth = std::ceilf((float)(imgWidth + 2 * paddingFrame - kernelWidth) / stride + 1);
    int fmHeight = std::ceilf((float)(imgHeight + 2 * paddingFrame - kernelWidth) / stride + 1);
    int fmCount = imgCount * kernelsCount;

    std::vector<int> fmShape = { fmCount, fmHeight, fmWidth };
    return fmShape;
}

std::vector<int> ConvolutionLayer::getPoolingMapShape() {
    int* tensorShape = this->getPrevLayerShape();
    int* kernelsShape = this->kernels->getShape();
    int imgCount = tensorShape[0];
    int imgHeight = tensorShape[1];
    int imgWidth = tensorShape[2];
    int kernelsCount = kernelsShape[0];
    int kernelWidth = kernelsShape[2];
    int paddingFrame = kernelWidth / 2;
    int fmWidth = std::ceilf((float)(imgWidth + 2 * paddingFrame - kernelWidth) / stride + 1);
    int fmHeight = std::ceilf((float)(imgHeight + 2 * paddingFrame - kernelWidth) / stride + 1);
    int fmCount = imgCount * kernelsCount;
    int poolingMapWidth = ceilf((float)fmWidth / (float)poolSize);
    int poolingMapHeight = ceilf((float)fmHeight / (float)poolSize);

    std::vector<int> poolingMapShape = { fmCount, poolingMapHeight, poolingMapWidth };
    return poolingMapShape;
}

void ConvolutionLayer::buildActivations() {
    if (!this->prevLayer) {
        std::stringstream ss;
        ss << "ConvolutionLayer cannot build without prevLayer set.";
        throw std::runtime_error(ss.str());
    }
    std::vector<int> poolingMapShape = getPoolingMapShape();
    this->activations = new Tensor(poolingMapShape);
    this->size = this->activations->getSize();
}

void ConvolutionLayer::buildParams() {
    std::vector<int> fmShape = getFeatureMapShape();
    this->biases = new Tensor(fmShape, 0.0f);
    this->params.push_back(this->kernels);
    this->params.push_back(this->biases);
}

void ConvolutionLayer::buildGrads() {
    int* kernelsShape = this->kernels->getShape();
    std::vector<int> fmShape = getFeatureMapShape();
    std::vector<int> poolingMapShape = getPoolingMapShape();

    this->dc_dz = new Tensor(poolingMapShape); //should be fmShape in theory but we can save as poolingShape and spread the values when doing the chain rule.
    this->dc_dk = new Tensor(kernelsShape, this->kernels->getNdim());
    this->dc_db = new Tensor(fmShape);
    this->grads.push_back(this->dc_dk);
    this->grads.push_back(this->dc_db);
}

void ConvolutionLayer::build() {
    int* kernelsShape = this->kernels->getShape();
    int kernelWidth = kernelsShape[2];
    std::vector<int> fmShape = getFeatureMapShape();
    std::vector<int> poolingMapShape = getPoolingMapShape();

    this->paddingFrame = kernelWidth / 2;
    this->f = new Tensor(fmShape);
    this->pooledIndices = new Tensor(poolingMapShape);
}

void ConvolutionLayer::calcLayer() {
    convolutionCuda(*this->prevLayer->activations, *this->kernels, *this->f, this->paddingFrame, this->stride); // f = featureMap
    plusCuda(*this->f, *this->biases, *this->f); // f = featureMap + biases 
    this->activationFunction->func(*this->f, *this->f); //f = func(featureMap + bias);
    poolingCuda(*this->f, *this->pooledIndices, *this->activations, this->poolMode, this->poolSize);
}

void ConvolutionLayer::calcGrads() {
    //lets assume next layer is neural for now.
    if (NeuralLayer* nextNeural = dynamic_cast<NeuralLayer*>(this->nextLayer)) {
        matvecCuda(*nextNeural->weightsTranspose, *nextNeural->dc_dz, *this->dc_dz); //writes dc_da to dc_dz
        this->activationFunction->chainRule(*this->activations, *this->dc_dz, *this->dc_dz); //dc_dz = dc_da * da_df * df_dz
        int kernelWidth = this->kernels->getShape()[2];
        int kernelsCount = this->kernels->getShape()[0];
        Tensor* pooledData = (this->poolMode == 'm') ? this->pooledIndices : this->activations;
        convolutionChainRulePlusEqualCuda(*this->prevLayer->activations, *pooledData, *this->dc_dz, *this->dc_db, *this->dc_dk,
            this->paddingFrame, this->stride, kernelWidth, this->poolMode, this->poolSize, kernelsCount); //adds the grads to dc_dk and dc_db
        pooledData = nullptr;
    }
    //nextLayer -> ConvolutionLayer will be developed later on...
}

void ConvolutionLayer::avgGrads(float batchSize) {
    divideByScalerCuda(*this->dc_dk, *this->dc_dk, batchSize);
    divideByScalerCuda(*this->dc_db, *this->dc_db, batchSize);
}

void ConvolutionLayer::resetGrads() {
    fillTensorCuda(*this->dc_dk, 0.0f);
    fillTensorCuda(*this->dc_db, 0.0f);
}

void ConvolutionLayer::subtractGrads(float learningRate) {
    subtractByScaleCuda(*this->kernels, *this->dc_dk, learningRate * convLearningRate);
    subtractByScaleCuda(*this->biases, *this->dc_db, learningRate);
}

std::string ConvolutionLayer::to_string() {
    std::ostringstream oss;
    oss << "ConvolutionLayer:\n" << this->activations->to_string() <<
        "kernels:\n" << this->kernels->to_string() << "biases:\n" << this->biases->to_string()
        << "dc_dk:\n" << this->dc_dk->to_string() << "dc_db:\n" << this->dc_db->to_string() << std::endl;
    return oss.str();
}


