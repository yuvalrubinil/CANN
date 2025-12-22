![CANN Logo](images/cann_logo.png)
# CANN - CUDA Accelerated Neural Networks (alpha)
A lightweight Python library for training neural networks with GPU acceleration.
Built on C++ and CUDA for maximum performance. Supports fully connected layers, convolutional layers (partial), and multiple optimizers.

# Features
1. Fully connected layers.
2. Convolutional layers (partial support).  
3. Optimizers: SGD, MinibatchGD, Momentum, Adagrad, RMSProp, Adam.  
4. Python bindings via PyBind11 for seamless integration with Python projects.  
5. Modular design for easy extension and creation of custom layers.  
6. Check out `cann_alpha/Network.h`, `cann_alpha/optimizers/Optimizers.cpp`, and `cann_alpha/layers` for the full list of supported operations

# In Development
- [x] Multi-layer backpropagation of convolutional layers.
- [x] More optimizers: RMSProp, Adam, etc.
- [ ] Dropout mechanizem.
- [ ] Python 3.13 and 3.14 support - right now only 3.12 is supported.

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
```

# Code Samples
This example demonstrates how to create, train, and test a simple network using `cann_alpha`.  
> **Note:** Replace the placeholders according to your desire, the right formats for these parameters are explianed in the next code samples.  

```python
import cann_alpha as cann

config = [
    {"type": "InputLayer", "input_shape": [1, 28, 28]},
    # Provide KERNELS and KERNELS_SHAPE in the required format
    {"type": "ConvLayer", "kernels": ..., "kernels_shape": ..., "pool_mode": 'm', "pool_size": 2, "stride": 1, "activation_function": "relu", "conv_lr": 0.1},
    {"type": "NeuralLayer", "layer_size": 512, "activation_function": "relu", "weights_init_method": "he_uniform"},
    {"type": "NeuralLayer", "layer_size": 512, "activation_function": "relu", "weights_init_method": "he_uniform"},
    {"type": "OutputLayer", "layer_size": 10, "activation_function": "softmax", "weights_init_method": "xavier"}
]

net = cann.Network(config, loss="cce", lr=0.01, batch_size=16, optimizer="gd")

# Load or prepare your dataset in the required format
train_set, test_set = ...  # Replace with your dataset

net.train(train_set, epochs=5)

accuracy = net.test(test_set)
print(f"Test Accuracy: {accuracy}")
```

This is an example of how to prepare the MNIST dataset for `cann`.  
> **Note:** This is just an example; you can use your own datasets in the same format.

```python
def build_dataset():
    import numpy as np
    import requests

    # Download the MNIST dataset
    url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
    r = requests.get(url)
    open("mnist.npz", "wb").write(r.content)

    # Load the dataset
    mnist = np.load("mnist.npz")
    train_images, train_labels = mnist["x_train"], mnist["y_train"]
    test_images, test_labels = mnist["x_test"], mnist["y_test"]

    # Helper function to convert images and labels into the required format
    def create_dataset(images, labels):
        dataset = []
        for img, label in zip(images, labels):
            flattened_img = (img / 255.0).flatten().tolist()  # Normalize and flatten
            img_shape = [1, 28, 28]  # Channels, height, width
            dataset.append((flattened_img, img_shape, int(label)))
        return dataset

    train_dataset = create_dataset(train_images, train_labels)
    test_dataset = create_dataset(test_images, test_labels)

    return train_dataset, test_dataset

# Example usage
train_set, test_set = build_dataset()
```

Define convolutional kernels
```python
# Define convolutional kernels
LAPLACIAN = [
     0,  1,  0,
     1, -4,  1,
     0,  1,  0
]

CORNER = [
     1, -1,  0,
    -1,  1,  0,
     0,  0,  0
]

KERNELS_SHAPE = [2, 3, 3]  # 2 kernels, each 3x3
KERNELS = LAPLACIAN + CORNER
```


