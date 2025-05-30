import numpy as np
import pandas as pd
from layers.conv2d import Conv2D
from layers.maxpool import MaxPool2
from layers.dense import Dense
from utils.activations import relu, softmax

# Load test data (first 1000 samples)
df = pd.read_csv("archive/mnist_test.csv")
X_test = df.iloc[:1000, 1:].values.reshape(-1, 1, 28, 28) / 255.0
y_test = df.iloc[:1000, 0].values

# Load model weights
conv = Conv2D(num_filters=8, filter_size=3)
conv.filters = np.load("model/conv_filters.npy")

pool = MaxPool2()

dense = Dense(input_len=13*13*8, output_len=10)
dense.weights = np.load("model/dense_weights.npy")
dense.biases = np.load("model/dense_biases.npy")

# Evaluate accuracy
correct = 0
for i in range(1000):
    x = conv.forward(X_test[i])
    x = relu(x)
    x = pool.forward(x)
    x = x.flatten()
    x = dense.forward(x)
    x = softmax(x)

    if np.argmax(x) == y_test[i]:
        correct += 1

accuracy = correct / 1000 * 100
print(f"✅ Accuracy on first 1000 test samples: {accuracy:.2f}% ({correct}/1000)")
