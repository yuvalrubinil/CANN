#include "LossFunc.h"
#include "cata/ops.cuh"

class MSE : public LossFunc {
public:
    float func(Tensor& y_predicted, Tensor& y_expected) override {
        return mseCuda(y_predicted, y_expected);
    }
    void der(Tensor& y_predicted, Tensor& y_expected, Tensor& w) override {
        mseDerCuda(y_predicted, y_expected, w);
    }
};

class CCE : public LossFunc {
public:
    float func(Tensor& y_predicted, Tensor& y_expected) override {
        return cceCuda(y_predicted, y_expected);
    }
    void der(Tensor& y_predicted, Tensor& y_expected, Tensor& w) override {
        cceDerCuda(y_predicted, y_expected, w);
    }
};

LossFunc* loadLossFunc(std::string funcName) {
    if (funcName == "mse")
        return new MSE();
    else if (funcName == "cce")
        return new CCE();
    throw std::invalid_argument("Unknown loss function: " + funcName);
}
