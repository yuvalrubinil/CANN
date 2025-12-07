#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Optimizer.h"
#include "../cata/ops.cuh"

Optimizer::Optimizer(std::vector<Layer*>* layers, float learningRate, int batchSize) {
	this->layers = layers;
	this->learningRate = learningRate;
    this->batchSize = batchSize;
}

Optimizer::~Optimizer() {
	this->layers = nullptr;
}
	
void Optimizer::buildOptGrads(Layer* layer) {
    int n = (int)layer->grads.size();
    for (int i = 0; i < n; i++) {
        int* shape = layer->grads[i]->getShape();
        int ndim = layer->grads[i]->getNdim();
        layer->optGrads.push_back(new Tensor(shape, ndim, 0.0f));
    }
}

void Optimizer::avgLayerGrads(Layer* layer) {
    for (Tensor* grad : layer->grads)
        divideByScalerCuda(*grad, *grad, this->batchSize);
}

void Optimizer::resetLayerGrads(Layer* layer) {
    for (Tensor* grad : layer->grads)
        fillTensorCuda(*grad, 0.0f);
}