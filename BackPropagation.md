# BackPropagation


```
Backpropagation is the key algorithm used for training neural networks, allowing them to learn from data.
It is based on the gradient descent optimization technique and works by iteratively adjusting the weights and
biases of the network to minimize the error between the predicted and actual outputs.
In this lab, we will create a neural network to implement backpropagation for a XOR problem.
```


Imports and Libraries:
```
# All Libraries required for this lab are listed below. The libraries pre-installed on Skills Network Labs are commented. 
# If you run this notebook on a different environment, e.g., your desktop, you may need to uncomment and install certain libraries.

#!pip install numpy==1.26.4
#!pip install matplotlib==3.5.2

# Importing the required library
import numpy as np
import matplotlib.pyplot as plt
```

### Initialize Inputs
Define the input and expected output for a XOR gate problem. 

> [!NOTE]
>The XOR (exclusive OR) gate problem in neural networks refers to the challenge of training a single-layer perceptron to classify
>  data that is not linearly separable, specifically the XOR function.
> XOR Logic: XOR outputs true (1) if only one input is true, and false (0) if both are true or both are false.

```
# Defining inputs and expected output (XOR truth table)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).T  # 2x4 matrix, each column is a training example
d = np.array([0, 1, 1, 0])  # Expected output for XOR
```

### Declare the network parameters¶
Define the network parameters
  - number of input neurons
  - hidden layer neurons
  - output neurons
  - learning rate
  - number of epochs

```
# Network parameters
inputSize = 2     # Number of input neurons (x1, x2)
hiddenSize = 2    # Number of hidden neurons
outputSize = 1    # Number of output neurons
lr = 0.1          # Learning rate
epochs = 180000   # Number of training epochs
```

### Define the weights
Declare the weights for the neurons. 
The initial weights are taken as random numbers which are then optimized by the backpropagation algorithm.

```
# Initialize weights and biases randomly within the range [-1, 1]
w1 = np.random.rand(hiddenSize, inputSize) * 2 - 1  # Weights from input to hidden layer
b1 = np.random.rand(hiddenSize, 1) * 2 - 1         # Bias for hidden layer
w2 = np.random.rand(outputSize, hiddenSize) * 2 - 1  # Weights from hidden to output layer
b2 = np.random.rand(outputSize, 1) * 2 - 1         # Bias for output layer
```

### Training the Neural Network
The neural network works in 5 stages:

1. **Forward pass**
> The input X is multiplied by the weights w1 and passed through the first layer, followed by the application of the sigmoid or ReLU activation function. This gives the output for the hidden layer.
> The output of the hidden layer is then passed through the second set of weights w2 to compute the final output. Again, a sigmoid activation function is used to generate the final output a2.

2. **Error calculation**
> The error is computed as the difference between the expected output (d) and the actual output (a2).

3. **Backward pass**
> Output Layer: The derivative of the sigmoid activation function is applied to the error, producing the gradient for the output layer (da2). This is used to calculate how much the weights in the output layer need to be adjusted.
> Hidden Layer: The error is then propagated backward to the hidden layer. The gradient at the hidden layer (da1) is computed by taking the dot product of the transpose of the weights (w2.T) and the gradient from the output layer. The derivative of the activation function (sigmoid or ReLU) is used to adjust this error.

4. **Weights and bias updates**
> After computing the gradients (dz1, dz2), the weights (w1, w2) and biases (b1, b2) are updated using the learning rate (lr) and the gradients. The updates are done to minimize the error and improve the model’s predictions.

5. **Training**
> This entire process is repeated over many iterations (epochs). During each epoch, the model adjusts its weights and biases to reduce the error. Over time, the network learns to approximate the XOR function. Forward Pass:

```
# Training the network using backpropagation
error_list = []
for epoch in range(epochs):
    # Forward pass
    z1 = np.dot(w1, X) + b1  # Weighted sum for hidden layer
    a1 = 1 / (1 + np.exp(-z1))  # Sigmoid activation for hidden layer

    z2 = np.dot(w2, a1) + b2  # Weighted sum for output layer
    a2 = 1 / (1 + np.exp(-z2))  # Sigmoid activation for output layer

    # Error calculation and backpropagation
    error = d - a2  # Difference between expected and actual output
    da2 = error * (a2 * (1 - a2))  # Derivative for output layer
    dz2 = da2  # Gradient for output layer

    # Propagate error to hidden layer
    da1 = np.dot(w2.T, dz2)  # Gradient for hidden layer
    dz1 = da1 * (a1 * (1 - a1))  # Derivative for hidden layer

    # Update weights and biases
    w2 += lr * np.dot(dz2, a1.T)  # Update weights from hidden to output layer
    b2 += lr * np.sum(dz2, axis=1, keepdims=True)  # Update bias for output layer

    w1 += lr * np.dot(dz1, X.T)  # Update weights from input to hidden layer
    b1 += lr * np.sum(dz1, axis=1, keepdims=True)  # Update bias for hidden layer
    if (epoch+1)%10000 == 0:
        print("Epoch: %d, Average error: %0.05f"%(epoch, np.average(abs(error))))
        error_list.append(np.average(abs(error)))
```

<img width="281" height="314" alt="image" src="https://github.com/user-attachments/assets/8cdac172-3732-49ca-89f1-6be23a93c808" />
