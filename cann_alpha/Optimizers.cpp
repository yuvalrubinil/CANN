#include <string>
#include "Optimizer.h"
#include "../cata/ops.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


class GD : public Optimizer {
public:

    GD::GD(std::vector<Layer*>* layers, float learningRate) : Optimizer(layers, learningRate) {}

    void GD::buildOptGrads(Layer* layer) override {} //No optGrads in GD.

    void GD::calcOptGrads(Layer* layer) override {} //No optGrads in GD.

    void GD::subtractGrads(Layer* layer) override {
        int n = (int)layer->grads.size();
        for (int i = 0; i < n; i++)
            subtractByScaleCuda(*layer->params[i], *layer->grads[i], this->learningRate); // w -= lr * grad
    }
};

class Momentum: public Optimizer {
public:
    float beta = 0.9f;

    Momentum::Momentum(std::vector<Layer*>* layers, float learningRate, float beta=0.9f) : Optimizer(layers, learningRate) {
        this->beta = beta;
    }

    void Momentum::calcOptGrads(Layer* layer) override{
        int n = (int)layer->optGrads.size();
        for (int i = 0; i < n; i++)
            momentumCuda(*layer->optGrads[i] , *layer->grads[i], this->beta); //second param is the newly calculted der, using it to write the momentum to the first param.
    }

    void Momentum::subtractGrads(Layer* layer) override {
        int n = (int)layer->optGrads.size();
        for (int i = 0; i < n; i++)
            subtractByScaleCuda(*layer->params[i], *layer->optGrads[i], this->learningRate); // w -= lr * momentum
    }
};

class RMSProp : public Optimizer {
public:
    float beta = 0.9f;

    RMSProp::RMSProp(std::vector<Layer*>* layers, float learningRate, float beta=0.9f) : Optimizer(layers, learningRate) {
        this->beta = beta;
    }

    void RMSProp::calcOptGrads(Layer* layer) override {
        int n = (int)layer->optGrads.size();
        for (int i = 0; i < n; i++)
            momentumSqueredCuda(*layer->optGrads[i], *layer->grads[i], this->beta); //second param is the newly calculted der, using it to write the momentum to the first param.
    }

    void RMSProp::subtractGrads(Layer* layer) override {
        int n = (int)layer->optGrads.size();
        for (int i = 0; i < n; i++)
            rmsPropSubtractionCuda(*layer->params[i], *layer->optGrads[i], *layer->grads[i], this->learningRate);//third param is the newly calculted der, second is the squeredUpdate momentum, writing to first.
    }
};


Optimizer* loadOptimizer(std::string optName, std::vector<Layer*>* layers, float learningRate, float beta=0.9f) {
    if (optName == "gd")
        return new GD(layers, learningRate);
    else if (optName == "momentum")
        return new Momentum(layers, learningRate, beta);
    else if (optName == "rms_prop")
        return new RMSProp(layers, learningRate, beta);
    throw std::invalid_argument("Unknown optimizer: " + optName);
}