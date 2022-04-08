import numpy.random
import pandas
import numpy as np
from math import *

def  readData(fileName):
    dataFrame = pandas.read_csv(fileName, header=None)

    data = dataFrame.to_numpy()
    np.random.shuffle(data)

    m = data.shape[0]
    XTrainingSet = data[:floor(0.6*m), :-1]
    YTrainingSet = np.array([data[:floor(0.6*m), -1]]).transpose()
    XCrossValidationSet = data[XTrainingSet.shape[0]:floor(0.8*m), :-1]
    YCrossValidationSet = np.array([data[YTrainingSet.shape[0]:floor(0.8*m), -1]]).transpose()
    XTestSet = data[XCrossValidationSet.shape[0]:m, :-1]
    YTestSet = np.array([data[YCrossValidationSet.shape[0]:m, -1]]).transpose()

    return [XTrainingSet, YTrainingSet, XCrossValidationSet, YCrossValidationSet, XTestSet, YTestSet]