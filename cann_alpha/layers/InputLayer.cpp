#include "InputLayer.h"
#include <sstream>


InputLayer::InputLayer(std::vector<int>& inputShape) : Layer(1) {
    int n = (int)inputShape.size();
    this->inputShape = new int[n];
    for (int i = 0; i < n; ++i) {
        this->size *= inputShape[i];
        this->inputShape[i] = inputShape[i];
    }
}

void InputLayer::plugInput(Tensor& x) {
    if (x.getSize() != this->size) {
        std::stringstream ss;
        ss << "InputLayer is size " << this->size
            << ", cannot accept input size " << x.getSize();
        throw std::runtime_error(ss.str());
    }
    this->activations = &x; 
}

void InputLayer::buildActivations() {}
void InputLayer::buildParams() {}
void InputLayer::buildGrads() {}
void InputLayer::build() {}
void InputLayer::calcLayer() {}
void InputLayer::calcGrads() {}

std::string InputLayer::to_string() {
    //printf("%d\n", this->activations);
    if (this->activations)
        return "InputLayer:\n" + this->activations->to_string();
    return "InputLayer:\n   null -> input Tensor was freed\n";
}

int* InputLayer::getInputShape() {
    return this->inputShape;
}
