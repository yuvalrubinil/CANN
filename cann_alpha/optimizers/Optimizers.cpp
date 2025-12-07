#include <string>
#include "Optimizer.h"
#include "../cata/ops.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


class GD : public Optimizer {
public:

    GD::GD(std::vector<Layer*>* layers, float learningRate, int batchSize) : Optimizer(layers, learningRate, batchSize) {}

    void GD::buildOptGrads(Layer* layer) override {} //No optGrads in GD.

    void GD::calcOptGrads(Layer* layer) override {} //No optGrads in GD.

    void GD::subtractGrads(Layer* layer) override {
        this->avgLayerGrads(layer);
        int n = (int)layer->grads.size();
        for (int i = 0; i < n; i++)
            subtractByScaleCuda(*layer->params[i], *layer->grads[i], this->learningRate); // w -= lr * grad
    }
};

class Momentum: public Optimizer {
public:
    float beta = 0.9f;

    Momentum::Momentum(std::vector<Layer*>* layers, float learningRate, int batchSize, float beta=0.9f) : Optimizer(layers, learningRate, batchSize) {
        this->beta = beta;
    }

    void Momentum::calcOptGrads(Layer* layer) override{
        this->avgLayerGrads(layer);
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

class Adagrad : public Optimizer {
public:

    Adagrad::Adagrad(std::vector<Layer*>* layers, float learningRate, int batchSize) : Optimizer(layers, learningRate, batchSize) {}

    void Adagrad::calcOptGrads(Layer* layer) override {
        int n = (int)layer->optGrads.size();
        for (int i = 0; i < n; i++)
            accumulatedSquersCuda(*layer->optGrads[i], *layer->grads[i]); //second param is the newly calculted der, using it to write the accumulated squers to the first param.
    }

    void Adagrad::subtractGrads(Layer* layer) override {
        int n = (int)layer->optGrads.size();
        for (int i = 0; i < n; i++)
            adagradSubtractionCuda(*layer->params[i], *layer->optGrads[i], *layer->grads[i], this->learningRate);//third param is the newly calculted der, second is the accumulated squers, writing to first.
    }
};

class RMSProp : public Optimizer {
public:
    float beta = 0.9f;

    RMSProp::RMSProp(std::vector<Layer*>* layers, float learningRate, int batchSize, float beta=0.9f) : Optimizer(layers, learningRate, batchSize) {
        this->beta = beta;
    }

    void RMSProp::calcOptGrads(Layer* layer) override {
        this->avgLayerGrads(layer);
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

class Adam : public Optimizer {
public:
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    int t = 0;

    Adam::Adam(std::vector<Layer*>* layers, float learningRate, int batchSize, float beta1 = 0.9f, float beta2 = 0.999f) : Optimizer(layers, learningRate, batchSize) {
        this->beta1 = beta1;
        this->beta2 = beta2;
    }

    void Adam::buildOptGrads(Layer* layer) override{
        int n = (int)layer->grads.size();
        for (int i = 0; i < n; i++) {
            int* shape = layer->grads[i]->getShape();
            int ndim = layer->grads[i]->getNdim();
            layer->optGrads.push_back(new Tensor(shape, ndim, 0.0f)); // V grad
            layer->optGrads.push_back(new Tensor(shape, ndim, 0.0f)); // S grad
        }
    }

    void Adam::calcOptGrads(Layer* layer) override {
        this->avgLayerGrads(layer);
        int n = (int)layer->grads.size();
        for (int i = 0; i < n; i++) {
            Tensor* v = layer->optGrads[2*i];
            Tensor* s = layer->optGrads[2*i+1];
            momentumCuda(*v, *layer->grads[i], this->beta1);
            momentumSqueredCuda(*s, *layer->grads[i], this->beta2);
        }
        ++t;
    }

    void Adam::subtractGrads(Layer* layer) override {
        int n = (int)layer->params.size();
        for (int i = 0; i < n; i++) {
            Tensor* v = layer->optGrads[2*i];
            Tensor* s = layer->optGrads[2*i + 1];
            adamSubtractionCuda(*layer->params[i], *s, *v, this->learningRate, this->beta1, this->beta2, this->t);

        }
    }
};


Optimizer* loadOptimizer(std::string optName, std::vector<Layer*>* layers, float learningRate, int batchSize, float beta=0.9f) {
    if (optName == "gd")
        return new GD(layers, learningRate, batchSize);
    else if (optName == "momentum")
        return new Momentum(layers, learningRate, batchSize, beta);
    else if (optName == "adagrad")
        return new Adagrad(layers, learningRate, batchSize);
    else if (optName == "rms_prop")
        return new RMSProp(layers, learningRate, batchSize, beta);
    else if (optName == "adam")
        return new Adam(layers, learningRate, batchSize);
    throw std::invalid_argument("Unknown optimizer: " + optName);
}