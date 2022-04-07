import numpy as np


def calculateSquareCost(X, Y, thetas, lamb):
    m = X.shape[0]

    error = np.dot(X, thetas) - Y
    cost = (1/(2*m)) * np.sum(np.dot(error.transpose(), error))
    regularization = (lamb/(2*m)) * np.sum(np.square(thetas)[1:])
    cost = cost + regularization

    return cost
