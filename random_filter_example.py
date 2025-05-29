import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from layers.conv2d import Conv2D

# Load MNIST test CSV
df = pd.read_csv("archive/mnist_test.csv")
data = df.iloc[:, 1:].values  # Ignore the labels
image = data[0].reshape(28, 28)  # First image
image = image / 255.0  # Normalize

# Prepare input shape for Conv2D (1, 28, 28)
input_image = image[np.newaxis, :, :]

# Apply convolution
conv = Conv2D(num_filters=4, filter_size=3)
output = conv.forward(input_image)  # Shape: (4, 26, 26)

# Plot
plt.figure(figsize=(12, 3))
plt.subplot(1, 5, 1)
plt.imshow(image, cmap='gray')
plt.title("Original")
plt.axis('off')

for i in range(4):
    plt.subplot(1, 5, i + 2)
    plt.imshow(output[i], cmap='gray')
    plt.title(f"Filter {i + 1}")
    plt.axis('off')

plt.tight_layout()
plt.show()
