#pragma once
#include "../cata/tensor.cuh"

class ActivationFunc {
public:
    virtual void func(Tensor& u, Tensor& w) = 0;
    virtual void chainRule(Tensor& u, Tensor& dc_da, Tensor& w) = 0;
    virtual ~ActivationFunc() = default;
};
