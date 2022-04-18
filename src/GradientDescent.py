import numpy as np
from math import floor
import matplotlib.pyplot as plt
from CostFunction import calculateSquareCost


def runGradientDescent(calculateGradient, X, Y, trainingSetMean, trainingSetStd, thetas, lamb, alpha, iterationNumber, printIter=False):
    if printIter:
        print(f"Training model with alpha {alpha} and lambda {lamb} in {iterationNumber} iterations")

    prevCost = calculateSquareCost(X, Y, thetas, lamb, trainingSetMean, trainingSetStd)
    for iteration in range(0, iterationNumber):
        if printIter:
            print(f"\tTraining model, iteration: {iteration}")

        thetas = thetas - alpha * calculateGradient(X, Y, thetas, lamb)

        cost = calculateSquareCost(X, Y, thetas, lamb, trainingSetMean, trainingSetStd)
        if prevCost <= cost:
            print(prevCost)
            print(cost)
            break
        else:
            prevCost = cost

        if printIter:
            print(f"\t\tIteration {iteration} cost: {cost}")

    return thetas


def displayLearningCurves(calculateGradient, XTrainingSet, YTrainingSet, XCVSet, YCVSet, trainingSetMean, trainingSetStd, thetas, lamb, alpha, iterationNumber):
    print("Drawing learning curves:")

    fig, ax = plt.subplots()

    exampleNumbers = []
    trainingCosts = []
    crossValidationCosts = []

    m, n = XTrainingSet.shape

    for trainingExamplesNumber in range(3000, m+1, floor(m/50)):
        XTrainingSubset = XTrainingSet[:trainingExamplesNumber+1, :]
        YTrainingSubset = YTrainingSet[:trainingExamplesNumber+1, :]
        XCVSubset = XCVSet[:trainingExamplesNumber+1, :]
        YCVSubset = YCVSet[:trainingExamplesNumber+1, :]

        print(f"\tRunning gradient descent for {trainingExamplesNumber}/{m} examples")
        optimisedThetas = runGradientDescent(calculateGradient, XTrainingSubset, YTrainingSubset, trainingSetMean, trainingSetStd, thetas, lamb, alpha, iterationNumber, True if trainingExamplesNumber==2 else False)

        exampleNumbers.append(trainingExamplesNumber)
        trainingCosts.append(calculateSquareCost(XTrainingSubset, YTrainingSubset, optimisedThetas, lamb, trainingSetMean, trainingSetStd))
        crossValidationCosts.append(calculateSquareCost(XCVSet, YCVSet, optimisedThetas, lamb, trainingSetMean, trainingSetStd))

    ax.plot(exampleNumbers, trainingCosts, label='training set')
    ax.plot(exampleNumbers, crossValidationCosts, label='cross validation set')
    ax.set_xlabel('training examples number')
    ax.set_ylabel('cost')
    ax.set_title('Learning Curves')
    ax.legend()

    plt.show()