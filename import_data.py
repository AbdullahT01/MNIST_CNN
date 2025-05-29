import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv("archive/mnist_test.csv")


# Pick an index to view
index = 0

# Extracting label and pixels
label = df.iloc[index, 0]
pixels = df.iloc[index, 1:].values.reshape(28, 28).astype(np.uint8)

# Plotting the image
plt.imshow(pixels, cmap='gray')
plt.title(f"Label: {label}")
plt.axis('off')
plt.show()
