#pragma once
#include "../cata/tensor.cuh"

class LossFunc {
public:
    virtual float func(Tensor& y_predicted, Tensor& y_expected) = 0;
    virtual void der(Tensor& y_predicted, Tensor& y_expected, Tensor& w) = 0;
    virtual ~LossFunc() = default;
};
