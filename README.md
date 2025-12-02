![CANN Logo](images/cann_logo.png)
# CANN - CUDA Accelerated Neural Networks (alpha)
A lightweight Python library for training neural networks with GPU acceleration.
Built on C++ and CUDA for maximum performance. Supports fully connected layers, convolutional layers (partial), and multiple optimizers.

# Features
1. Fully connected layers.
2. Convolutional layers (partial support).  
3. Optimizers: SGD, Minibatch, Momentum.  
4. Python bindings via PyBind11 for seamless integration with Python projects.  
5. Modular design for easy extension and creation of custom layers.  
6. Check out `layers/`, `optimizers/`, and `tensor.cuh` for the full list of supported operations

# In Development
1. Multi-layer backpropagation of convolutional layers - right now only one is supported.
2. More optimizers: RMSProp, Adam, etc.
3. Dropout mechanizem.
4. Python 3.13 and 3.14 support - right now only 3.12 is supported.

# About
I am a computer science student with a strong interest in deep learning. I created CANN for two main reasons:
1. To learn about neural networks and GPU-accelerated computation.
2. To develop reusable code that can be applied in future deep learning projects.

## Requirements
To run this project, ensure your environment meets the following requirements:
### CUDA-Enabled GPU:
- NVIDIA GPU with CUDA support.
- Latest NVIDIA driver installed.
- CUDA Toolkit v13.0.
> **Note:** This project is GPU-accelerated. It will not run effectively, if at all, without a properly configured CUDA environment.
### Python:
- Python 3.12 (64-bit).
### Operating System:
- Windows 10 or 11 (64-bit).

# Installation
You can install the package from TestPyPI using the following command:
```bash
pip install -i https://test.pypi.org/simple/ cann-alpha


# Code Sample
???

