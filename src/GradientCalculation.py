import numpy as np


def calculateGradient(X, Y, thetas, lamb):
    m = X.shape[0]
    errorVector = np.dot(X, thetas) - Y
    gradientFeaturesVector = (1/m) * np.dot(X.transpose(), errorVector)
    #regularization = (lamb/m) * np.sum(thetas[1:])
    thetasWithoutBias = thetas.copy()
    thetasWithoutBias[0, 0] = 0
    regularization = (lamb / m) * thetasWithoutBias

    gradientFeaturesVector = gradientFeaturesVector + regularization

    return gradientFeaturesVector