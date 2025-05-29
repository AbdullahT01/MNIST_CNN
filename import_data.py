import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv("archive/mnist_test.csv")


# Pick an index to view
index = 2

# Extracting label and pixels
label = df.iloc[index, 0]
# .reshape(28, 28): turns the flat array into a 28×28 matrix (image)
# .astype(np.uint8): ensures the values are 8-bit integers for grayscale display
pixels = df.iloc[index, 1:].values.reshape(28, 28).astype(np.uint8)

# Plotting the image, or matrix in this case
plt.imshow(pixels, cmap='gray')
plt.title(f"Label: {label}")
plt.axis('off')
plt.show()
