import numpy as np
import matplotlib.pyplot as plt
from src.EvaluationMetrics import meanSquaredError


class RidgeRegression:
    def __init__(self, X, Y, alpha, iteration_number, lamb=0):
        self.iterationNumber = iteration_number
        self.X = X
        self.Y = Y
        self.alpha = alpha
        self.lamb = lamb
        self.thetas = np.zeros((X.shape[1], 1))
        self.iterationLoss = None

    def fit(self, X, Y):
        self.X = X
        self.Y = np.array([Y]).transpose()
        self.thetas = np.zeros((self.X.shape[1], 1))
        self.iterationLoss = np.empty(self.iterationNumber)
        for i in range(0, self.iterationNumber):
            gradient = self.calculate_gradient(self.X, self.Y, self.thetas, self.lamb)
            self.thetas = self.thetas - self.alpha * gradient

            predicted_values = np.dot(self.X, self.thetas)
            self.iterationLoss[i] = meanSquaredError(self.Y, predicted_values)
        return self.thetas

    def calculate_gradient(self, X, Y, thetas, lamb):
        """
            Gradient descent based on 1/2 Mean Squared Error with L2 regularization

        :param X: features matrix
        :param Y: target vector
        :param thetas: weights of features
        :param lamb: L2 regularization parameter weight
        :return: regularized gradient
        """
        m = X.shape[0]

        error_vector = np.dot(X, thetas) - Y
        gradient_vector = (1 / m) * np.dot(X.transpose(), error_vector)

        weights_without_bias = thetas.copy()
        weights_without_bias[0, 0] = 0
        regularization = (lamb ) * weights_without_bias

        return gradient_vector + regularization

    def predict(self, X):
        return np.dot(X, self.thetas)[:, 0]

    def plotIterationLoss(self):
        fig, ax = plt.subplots()
        ax.plot(list(range(0, self.iterationNumber)), self.iterationLoss)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Cost')
        ax.set_title('Gradient descent - cost by iteration')
        fig.canvas.manager.set_window_title('Gradient descent - cost by iteration')
        plt.show()