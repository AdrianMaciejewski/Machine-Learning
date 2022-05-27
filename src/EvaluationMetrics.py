import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from math import ceil

from src.DataProcessing import *


def meanSquaredError(y, y_pred):
    error = y - y_pred
    squaredErrors = np.power(error, 2)
    totalSquaredErrors = np.sum(squaredErrors)
    meanSquaredError = totalSquaredErrors / y.shape[0]
    return meanSquaredError


def meanAbsoluteError(y, y_pred):
    error = y - y_pred
    absoluteErrors = np.abs(error)
    totalAbsoluteErrors = np.sum(absoluteErrors)
    meanAbsoluteError = totalAbsoluteErrors / y.shape[0]
    return meanAbsoluteError


def rootMeanSquaredError(y, y_pred):
    rootMeanSquaredError = np.sqrt(meanSquaredError(y, y_pred))
    return rootMeanSquaredError


def rootMeanSquaredLogError(y, y_pred):
    rootMeanSquaredLogError = np.sqrt(rootMeanSquaredError(y, y_pred))
    return rootMeanSquaredLogError


def r2Score(y, y_pred):
    rss = np.sum(np.power(y-y_pred, 2))
    tss = np.sum(np.power(y-np.sum(y)/y.shape[0], 2))
    r2Score = 1 - rss/tss
    return r2Score


def adjustedR2Score(y, y_pred, features_number):
    n = y.shape[0]
    k = features_number
    adjustedR2Score = 1 - (((1 - r2Score(y, y_pred)) * (n - 1)) / (n - k - 1))
    return adjustedR2Score


def printAllRegressionMetrics(y, y_pred, features_number):
    print(f"Mean absolute error (MAE): \t\t\t\t{meanAbsoluteError(y, y_pred)}")
    print(f"Mean squared error (MSE): \t\t\t\t{meanSquaredError(y, y_pred)}")
    print(f"Root mean squared error (RMSE): \t\t{rootMeanSquaredError(y, y_pred)}")
    print(f"Root mean squared log error (RMSLE): \t{rootMeanSquaredLogError(y, y_pred)}")
    print(f"R2 Score: \t\t\t\t\t\t\t\t{r2Score(y, y_pred)}")
    print(f"Adjusted R2 Score: \t\t\t\t\t\t{adjustedR2Score(y, y_pred, features_number)}")
    print(f"number of data samples: \t\t\t\t{y.shape[0]}")


def plotLearningCurves(algorithm_instance, x_training, y_training, x_cv, y_cv, numerical_features_indexes, starting_sample_number, group_split_number, is_logged):
    m, n = x_training.shape
    fig, ax = plt.subplots()

    sample_numbers = []
    training_costs = []
    cv_costs = []

    for trainingExamplesNumber in range(starting_sample_number, m + 1, ceil((m - starting_sample_number) / group_split_number)):
        x_training_subset = x_training[:trainingExamplesNumber, :].copy()
        x_training_subset_normalised = x_training_subset
        x_training_subset_normalised[:, numerical_features_indexes], x_training_mean, x_training_std = normalize(x_training_subset[:, numerical_features_indexes])
        x_cv_normalised = x_cv.copy()
        x_cv_normalised[:, numerical_features_indexes] = shift_and_scale_matrix(x_cv[:, numerical_features_indexes], x_training_mean, x_training_std)

        y_training_subset = y_training[:trainingExamplesNumber, :].copy()
        y_training_subset_normalised, y_training_shift, y_training_scale = normalize(y_training_subset)
        y_cv_normalised = shift_and_scale_matrix(y_cv, y_training_shift, y_training_scale)

        print(f"\tRunning gradient descent for {trainingExamplesNumber}/{m} examples")
        #algorithm_instance.X = x_training_subset_normalised
        #algorithm_instance.Y = y_training_subset_normalised
        #algorithm_instance.thetas = np.zeros((algorithm_instance.X.shape[1], 1))
        algorithm_instance.fit(x_training_subset_normalised, y_training_subset_normalised[:, 0])

        y_training_prediction_normalised = np.array([algorithm_instance.predict(x_training_subset_normalised)]).transpose()
        y_training_prediction = reverse_shift_and_scale(y_training_prediction_normalised, y_training_shift, y_training_scale)
        y_cv_prediction_normalised = np.array([algorithm_instance.predict(x_cv_normalised)]).transpose()
        y_cv_prediction = reverse_shift_and_scale(y_cv_prediction_normalised, y_training_shift, y_training_scale)

        y_cv_original = y_cv.copy()
        if is_logged:
            y_training_prediction = np.exp(y_training_prediction)
            y_cv_prediction = np.exp(y_cv_prediction)
            y_training_subset = np.exp(y_training_subset)
            y_cv_original = np.exp(y_cv)

        sample_numbers.append(trainingExamplesNumber)
        training_costs.append(meanSquaredError(y_training_subset, y_training_prediction))
        cv_costs.append(meanSquaredError(y_cv_original, y_cv_prediction))

        print("\tTraining set:")
        printAllRegressionMetrics(y_training_subset, y_training_prediction, x_training_subset_normalised.shape[1])
        print("\tTest set:")
        printAllRegressionMetrics(y_cv_original, y_cv_prediction, x_training_subset_normalised.shape[1])

    ax.plot(sample_numbers, training_costs, label='training set')
    ax.plot(sample_numbers, cv_costs, label='cross validation set')
    ax.set_xlabel('training examples number')
    ax.set_ylabel('cost')
    ax.set_title('Learning Curves')
    ax.legend()

    plt.show()


def plot_predictions_and_values(data, hue):
    sns.scatterplot(data=data, x="value", y="predicted value", hue=hue)
    plt.show()
