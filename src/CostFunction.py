import numpy as np


def calculateSquareCost(X, Y, thetas, lamb, mean, std):
    m, n = X.shape

    valuesAdjustedByNormalization = np.dot(np.multiply(X, np.tile(std, (m, 1))) + np.tile(mean, (m, 1)), thetas)
    #print("HERE")
    #np.set_printoptions(linewidth=np.inf, threshold=np.inf)
    #print(valuesAdjustedByNormalization[:100])
    #print(thetas[:100, :])
    #print("ENDHERE")
    #np.set_printoptions(linewidth=400, threshold=30)
    error = valuesAdjustedByNormalization - Y
    #print(Y)
    #print(np.dot(X, thetas))
    cost = (1/(2*m)) * np.sum(np.dot(error.transpose(), error))
    #regularization = (lamb/(2*m)) * np.sum(np.square(thetas)[1:])
    regularization = (lamb/(2*m)) * np.dot(thetas.transpose()[0, 1:], thetas[1:, 0])
    #print(np.sum(np.square(thetas)[1:]))
    #print(np.dot(thetas.transpose()[0, 1:], thetas[1:, 0]))
    cost = cost + regularization

    return cost
