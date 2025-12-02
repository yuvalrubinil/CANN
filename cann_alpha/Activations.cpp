#include <string>
#include "ActivationFunc.h"
#include "cata/ops.cuh"


class Sigmoid : public ActivationFunc {
public:
    void func(Tensor& u, Tensor& w) override {
        sigmoidCuda(u, w);
    }
    void chainRule(Tensor& u, Tensor& dc_da, Tensor& w) override {
        sigmoidChainRuleCuda(u, dc_da, w); //writes to w
    }
};

class Relu : public ActivationFunc {
public:
    void func(Tensor& u, Tensor& w) override {
        reluCuda(u, w);
    }
    void chainRule(Tensor& u, Tensor& dc_da, Tensor& w) override {
        reluChainRuleCuda(u, dc_da, w); //writes to w
    }
};

class Softmax : public ActivationFunc {
public:
    void func(Tensor& u, Tensor& w) override {
        softmaxCuda(u, w);
    }
    void chainRule(Tensor& u, Tensor& dc_da, Tensor& w) override {
        softmaxChainRuleCuda(u, dc_da, w); //chain rule computed inside
    }
};

class Linear : public ActivationFunc {
public:
    void func(Tensor& u, Tensor& w) override {}
    void chainRule(Tensor& u, Tensor& dc_da, Tensor& w) override {
        copyCuda(dc_da, w); //writes 1*dc_da to w 
    }
};

ActivationFunc* loadActivationFunc(std::string funcName) {
    if (funcName == "sigmoid")
        return new Sigmoid();
    else if (funcName == "relu")
        return new Relu();
    else if (funcName == "softmax")
        return new Softmax();
    else if (funcName == "linear")
        return new Linear();
    throw std::invalid_argument("Unknown activation function: " + funcName);
}
