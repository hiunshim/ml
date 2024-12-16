import numpy as np


def compute_sigmoid(z):
    return 1 / (1 + np.exp(-z))
