import numpy.random
from DataReader import readData
from FeatureNormalisation import normaliseFeatures
from CostFunction import calculateSquareCost
from GradientCalculation import calculateGradient
from GradientDescent import runGradientDescent, displayLearningCurves

XTrainingSet, YTrainingSet, XCrossValidationSet, YCrossValidationSet, XTestSet, YTestSet = readData(r"C:\Users\adma\Documents\ML project\data\test.txt")

XTrainingSet = normaliseFeatures(XTrainingSet)
XCrossValidationSet = normaliseFeatures(XCrossValidationSet)
XTestSet = normaliseFeatures(XTestSet)

XTrainingSet = numpy.insert(XTrainingSet, obj=0, values=1, axis=1)
XCrossValidationSet = numpy.insert(XCrossValidationSet, obj=0, values=1,  axis=1)
XTestSet = numpy.insert(XTestSet, obj=0, values=1,  axis=1)

lamb = 3
alpha = 0.01
gradientDescentIterationNumber = 100

thetas = numpy.zeros((XTrainingSet.shape[1], 1))
optimisedThetas = runGradientDescent(calculateGradient, XTrainingSet, YTrainingSet, thetas, lamb, alpha, gradientDescentIterationNumber)

displayLearningCurves(calculateGradient, XTrainingSet, YTrainingSet, XCrossValidationSet, YCrossValidationSet, thetas, lamb, alpha, gradientDescentIterationNumber)

print(f'Test set cost equals {calculateSquareCost(XTestSet, YTestSet, optimisedThetas, lamb)}')