import numpy.random
from DataReader import *
from FeatureNormalisation import normaliseFeatures
from CostFunction import calculateSquareCost
from GradientCalculation import calculateGradient
from GradientDescent import runGradientDescent, displayLearningCurves


sanitiseData(r"C:\Users\Adrian- Admin\Desktop\ML project\raw_claims.csv")

"""
XTrainingSet, YTrainingSet, XCrossValidationSet, YCrossValidationSet, XTestSet, YTestSet = readData(r"C:Users\Adrian- Admin\Desktop\ML project\src\sanitised_data2.csv")

#np.set_printoptions(linewidth=np.inf, threshold=np.inf)
#print(XTrainingSet[:20, :])
#print(YTrainingSet[:20, :])
#np.set_printoptions(linewidth=400, threshold=20)

XTrainingSet, trainingSetMean, trainingSetStd = normaliseFeatures(XTrainingSet, featuresToNormalise=[0], trainingSetName="training set")
XCrossValidationSet, crossValidationSetMean, crossValidationSetStd = normaliseFeatures(XCrossValidationSet, featuresToNormalise=[0], trainingSetName="cross validation set")
XTestSet, testSetMean, testSetStd = normaliseFeatures(XTestSet, featuresToNormalise=[0], trainingSetName="test set")

print("Adding bias term to all training sets")
XTrainingSet = numpy.insert(XTrainingSet, obj=0, values=1, axis=1)
XCrossValidationSet = numpy.insert(XCrossValidationSet, obj=0, values=1,  axis=1)
XTestSet = numpy.insert(XTestSet, obj=0, values=1,  axis=1)


#allx = normaliseFeatures(allx, featuresToNormalise=[0], trainingSetName="test set")
#allx = numpy.insert(allx, obj=0, values=1,  axis=1)

lamb = 0
alpha = 0.001
gradientDescentIterationNumber = 100



thetas = numpy.zeros((XTrainingSet.shape[1], 1))
optimisedThetas = runGradientDescent(calculateGradient, XTrainingSet, YTrainingSet, trainingSetMean, trainingSetStd, thetas, lamb, alpha, gradientDescentIterationNumber, True)



displayLearningCurves(calculateGradient, XTrainingSet, YTrainingSet, XCrossValidationSet, YCrossValidationSet, trainingSetMean, trainingSetStd, thetas, lamb, alpha, gradientDescentIterationNumber)
#print(numpy.array([[-1, 2]]).transpose())
#print(optimisedThetas)
print(f'Test set cost equals {calculateSquareCost(XTrainingSet, YTrainingSet, optimisedThetas, lamb, trainingSetMean, trainingSetStd)}')
#print(np.array([[1, 35]]))
print(optimisedThetas)
#print(np.dot(np.array([[1, 7]]),optimisedThetas))
#np.set_printoptions(linewidth=np.inf, threshold=np.inf)
#print( normaliseFeatures(allx))
#print(np.concatenate((XTrainingSet, XCrossValidationSet, XTestSet), axis=0))
#print(XTrainingSet.shape)
#print(XCrossValidationSet.shape)
#print(XTestSet.shape)
#np.set_printoptions(linewidth=400, threshold=20)
"""