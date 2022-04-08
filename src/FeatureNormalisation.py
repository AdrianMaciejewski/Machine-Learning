import numpy as np


def normaliseFeatures(X):
    m = X.shape[0]

    featureMeanAverage = np.array([X.sum(axis=0) / m])

    allSquareDefferences = np.square(X - np.tile(featureMeanAverage, (m, 1)))
    featureStandardDeviationVector = np.array([np.sqrt(allSquareDefferences.sum(axis=0) / m)])

    return np.divide(X-np.tile(featureMeanAverage, (m, 1)), np.tile(featureStandardDeviationVector, (m, 1)))