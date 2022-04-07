import numpy as np
import matplotlib.pyplot as plt
from CostFunction import calculateSquareCost


def runGradientDescent(calculateGradient, X, Y, thetas, lamb, alpha, iterationNumber, printIterCosts=False):

    for iteration in range(0, iterationNumber):
        thetas = thetas - alpha * calculateGradient(X, Y, thetas, lamb)

        if printIterCosts:
            print(calculateSquareCost(X, Y, thetas, lamb))

    return thetas


def displayLearningCurves(calculateGradient, XTrainingSet, YTrainingSet, XCVSet, YCVSet, thetas, lamb, alpha, iterationNumber):
    fig, ax = plt.subplots()

    exampleNumbers = []
    trainingCosts = []
    crossValidationCosts = []

    for trainingExamplesNumber in range(3, 20, 2):
        XTrainingSubset = XTrainingSet[:trainingExamplesNumber+1, :]
        YTrainingSubset = YTrainingSet[:trainingExamplesNumber+1, :]
        XCVSubset = XCVSet[:trainingExamplesNumber+1, :]
        YCVSubset = YCVSet[:trainingExamplesNumber+1, :]

        optimisedThetas = runGradientDescent(calculateGradient, XTrainingSubset, YTrainingSubset, thetas, lamb, alpha, iterationNumber)

        exampleNumbers.append(trainingExamplesNumber)
        trainingCosts.append(calculateSquareCost(XTrainingSubset, YTrainingSubset, optimisedThetas, lamb))
        crossValidationCosts.append(calculateSquareCost(XCVSubset, YCVSubset, optimisedThetas, lamb))

    ax.plot(exampleNumbers, trainingCosts, label='training set')
    ax.plot(exampleNumbers, crossValidationCosts, label='cross validation set')
    ax.set_xlabel('training examples number')
    ax.set_ylabel('cost')
    ax.set_title('Learning Curves')
    ax.legend()

    plt.show()