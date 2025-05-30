import numpy as np

def cross_entropy_loss(predictions, label):
    return -np.log(predictions[label] + 1e-15)  # Add epsilon to prevent log(0)

def cross_entropy_derivative(predictions, label):
    grad = np.zeros_like(predictions)
    grad[label] = -1 / (predictions[label] + 1e-15)
    return grad
