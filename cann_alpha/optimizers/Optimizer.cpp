#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Optimizer.h"


Optimizer::Optimizer(std::vector<Layer*>* layers, float learningRate) {
	this->layers = layers;
	this->learningRate = learningRate;
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


