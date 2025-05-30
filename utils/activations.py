import numpy as np

def relu(x):
    return np.maximum(0, x)


def softmax(x):
    e_x = np.exp(x - np.max(x))  # for numerical stability
    return e_x / np.sum(e_x)


def relu_deriv(x):
    return (x > 0).astype(float)

def softmax_cross_entropy_loss(probs, label):
    return -np.sum(label * np.log(probs + 1e-7))
