import numpy as np
import pandas as pd
from layers.conv2d import Conv2D
from layers.maxpool import MaxPool2
from layers.dense import Dense
from utils.activations import relu, relu_deriv, softmax, softmax_cross_entropy_loss

# Load data
df = pd.read_csv("archive/mnist_train.csv")
X = df.iloc[:, 1:].values.reshape(-1, 1, 28, 28) / 255.0
y = df.iloc[:, 0].values
num_classes = 10

# One-hot encode labels
Y = np.eye(num_classes)[y]

# Model
conv = Conv2D(num_filters=8, filter_size=3)
pool = MaxPool2()
dense = Dense(input_len=13*13*8, output_len=10)

# Training loop
learning_rate = 0.001
epochs = 20
sample_size = 1000  # Change this to 1000 or more later

for epoch in range(epochs):
    loss_sum = 0
    correct = 0

    for i in range(sample_size):
        input_image = X[i]
        label = Y[i]

        # FORWARD
        conv_out = conv.forward(input_image)
        relu_out = relu(conv_out)
        pool_out = pool.forward(relu_out)
        out_flat = pool_out.flatten()
        out_dense = dense.forward(out_flat)
        out_softmax = softmax(out_dense)

        # LOSS
        loss = softmax_cross_entropy_loss(out_softmax, label)
        loss_sum += loss

        # ACCURACY
        pred = np.argmax(out_softmax)
        true = np.argmax(label)
        if pred == true:
            correct += 1

        # BACKWARD
        grad = out_softmax - label
        grad = dense.backward(grad, learning_rate)
        grad = grad.reshape(8, 13, 13)
        grad = pool.backward(grad)
        grad = relu_deriv(conv_out) * grad
        conv.backward(grad, learning_rate)

    avg_loss = loss_sum / sample_size
    accuracy = correct / sample_size * 100
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

# Create directory if not exists
import os
os.makedirs("model", exist_ok=True)

# Save model weights
np.save("model/conv_filters.npy", conv.filters)
np.save("model/dense_weights.npy", dense.weights)
np.save("model/dense_biases.npy", dense.biases)

print("✅ Model weights saved to 'model/' folder.")