#pragma once
#include <stdexcept>
#include <vector>
#include "../layers/Layer.h"

class Optimizer {
public:
    std::vector<Layer*>* layers;
    float learningRate;
    int batchSize;

    Optimizer(std::vector<Layer*>* layers, float learningRate, int batchSize);
    ~Optimizer();
    virtual void buildOptGrads(Layer* layer);
    void avgLayerGrads(Layer* layer);
    void resetLayerGrads(Layer* layer);
    virtual void calcOptGrads(Layer* layer) = 0;
    virtual void subtractGrads(Layer* layer) = 0;
};
