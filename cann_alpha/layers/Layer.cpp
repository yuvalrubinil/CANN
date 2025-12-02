#pragma once
#include "Layer.h" 
#include <iostream>

Layer::Layer(int size) {
	this->size = size;
}

Layer::~Layer() {
	this->prevLayer = nullptr;
	this->nextLayer = nullptr;
	if (this->activations) {
		this->activations->free();
		this->activations = nullptr;
	}
}

void Layer::setNextLayer(Layer& nextLayer) {
	this->nextLayer = &nextLayer;
}

void Layer::setPrevLayer(Layer& prevLayer) {
	this->prevLayer = &prevLayer;
}

void Layer::connectTo(Layer& prevLayer) {
	this->prevLayer = &prevLayer;
	this->prevLayer->setNextLayer(*this);
	this->buildLayer();
}

void Layer::buildLayer() {
	this->buildActivations();
	this->buildParams();
	this->buildGrads();
	this->build();
}

std::string Layer::to_string() {
	return this->activations->to_string();
}
