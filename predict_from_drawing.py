import numpy as np
from layers.conv2d import Conv2D
from layers.maxpool import MaxPool2
from layers.dense import Dense
from utils.activations import relu, softmax
import matplotlib.pyplot as plt

# Load weights
conv = Conv2D(num_filters=8, filter_size=3)
conv.filters = np.load("model/conv_filters.npy")

pool = MaxPool2()
dense = Dense(input_len=13*13*8, output_len=10)
dense.weights = np.load("model/dense_weights.npy")
dense.biases = np.load("model/dense_biases.npy")

# Load and display input image (expects 28x28 grayscale .npy image)
image = np.load("sample_input.npy")  # shape (28, 28), pixel values [0, 1]
input_image = image[np.newaxis, :, :]  # shape (1, 28, 28)

# Run through model
out = conv.forward(input_image)
out = relu(out)
out = pool.forward(out)
out = out.flatten()
out = dense.forward(out)
out = softmax(out)

# Show result
print("Softmax:", out)
print("Predicted Digit:", np.argmax(out))
plt.imshow(image, cmap='gray')
plt.title(f"Predicted: {np.argmax(out)}")
plt.show()
