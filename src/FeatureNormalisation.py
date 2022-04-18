import numpy as np


def normaliseFeatures(X, featuresToNormalise=[],  trainingSetName=""):
    print(f"Normalising features {f'for {trainingSetName}' if len(trainingSetName) != 0 else ''}")

    if len(featuresToNormalise) != 0:
        matrixToOptimise = X[:, featuresToNormalise]
    else:
        matrixToOptimise = X[:, :]

    m = matrixToOptimise.shape[0]

    #featureMeanAverage = np.array([matrixToOptimise.sum(axis=0) / m])
    featureMeanAverage = np.array(np.mean(matrixToOptimise, axis=0))
    #print("here")
    #print(np.tile(featureMeanAverage, (m, 1)))

    #allSquareDefferences = np.square(matrixToOptimise - np.tile(featureMeanAverage, (m, 1)))
    #featureStandardDeviationVector = np.array([np.sqrt(allSquareDefferences.sum(axis=0) / m)])
    featureStandardDeviationVector = np.array([np.std(matrixToOptimise, axis=0)])
    #print("here")
    #print(np.tile(featureStandardDeviationVector, (m, 1)))

    """
    np.set_printoptions(linewidth=np.inf, threshold=np.inf)
    print(featureMeanAverage)
    print("/n")
    print(featureStandardDeviationVector)
    np.set_printoptions(linewidth=400, threshold=20)
    print("/n")
    print("/n")
    print("/n")
    print(np.tile(featureMeanAverage, (m, 1)))
    print("/n")
    print(np.tile(featureStandardDeviationVector, (m, 1)))
    print("/n")
    print("/n")
    print((X-np.tile(featureMeanAverage, (m, 1))).shape)
    print(np.tile(featureStandardDeviationVector, (m, 1)).shape)
    """

    optimisedMatrix = np.divide(matrixToOptimise-np.tile(featureMeanAverage, (m, 1)), np.tile(featureStandardDeviationVector, (m, 1)))
    if len(featuresToNormalise) != 0:
        X[:, featuresToNormalise] = optimisedMatrix
    else:
        X = optimisedMatrix

    return X, featureMeanAverage, featureStandardDeviationVector