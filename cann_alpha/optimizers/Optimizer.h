#pragma once
#include <stdexcept>
#include <vector>
#include "../layers/Layer.h"

class Optimizer {
public:
    std::vector<Layer*>* layers;
    float learningRate;

    Optimizer(std::vector<Layer*>* layers, float learningRate);
    ~Optimizer();
    virtual void buildOptGrads(Layer* layer);
    virtual void calcOptGrads(Layer* layer) = 0;
    virtual void subtractGrads(Layer* layer) = 0;
};
