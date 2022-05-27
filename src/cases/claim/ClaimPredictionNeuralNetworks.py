import copy

import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor

from src.DataProcessing import *
from src.EvaluationMetrics import printAllRegressionMetrics
from src.cases.claim.ClaimPredictionBase import ClaimPredictionBase


class ClaimPredictionNeuralNetwork(ClaimPredictionBase):
    def __init__(self, df, target_feature, numerical_features, categorical_features):
        super().__init__(df, target_feature, numerical_features, categorical_features)

    def fit_k_fold(self, activation, learning_rate_init, max_iter, hidden_layer_sizes, alpha, plot_loss_curve=False):
        regression = MLPRegressor(activation=activation, learning_rate_init=learning_rate_init, max_iter=max_iter, hidden_layer_sizes=hidden_layer_sizes, alpha=alpha, random_state=1)

        samples_number = self.df.shape[0]
        all_train_predictions = np.zeros((10, samples_number))
        all_test_predictions = np.zeros((10, samples_number))
        k_fold_number = len(self.k_fold_splits)

        for i in range(0, k_fold_number):
            print(f"Calculating k-fold {i+1}")
            x_train = self.prepared_dfs[i]["x_train"].to_numpy(dtype=float)
            y_train = self.prepared_dfs[i]["y_train"].to_numpy(dtype=float)
            x_test = self.prepared_dfs[i]["x_test"].to_numpy(dtype=float)
            y_test = self.prepared_dfs[i]["y_test"].to_numpy(dtype=float)
            train_indices = self.k_fold_splits[i]["train"]
            test_indices = self.k_fold_splits[i]["test"]
            regression.fit(x_train, y_train[:, 0])

            all_train_predictions[i, train_indices] = regression.predict(x_train)
            all_test_predictions[i, test_indices] = regression.predict(x_test)

        self.train_predictions = np.zeros(samples_number)
        self.test_predictions = np.zeros(samples_number)
        self.train_predictions = np.sum(all_train_predictions, axis=0) / (k_fold_number - 1)
        self.test_predictions = np.sum(all_test_predictions, axis=0)
        self.train_predictions = self._reverse_transformation_on_prediction(np.array([self.train_predictions]).transpose(), self.y_shift_vector, self.y_scale_vector)
        self.test_predictions = self._reverse_transformation_on_prediction(np.array([self.test_predictions]).transpose(), self.y_shift_vector, self.y_scale_vector)

        y = np.array([self.df[self.target_feature].to_numpy(dtype=float)]).transpose()
        print(y)
        print(self.train_predictions)
        print(y.shape)
        print(self.train_predictions.shape)
        print("\tTraining set:")
        printAllRegressionMetrics(y, self.train_predictions, self.prepared_dfs[0]["x_train"].shape[1])
        print("\tTest set:")
        printAllRegressionMetrics(y, self.test_predictions, self.prepared_dfs[0]["x_train"].shape[1])

        if plot_loss_curve:
            fig, ax = plt.subplots()
            ax.plot(list(range(0, len(regression.loss_curve_))), regression.loss_curve_, label="NN learning curve")
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Cost')
            ax.set_title('Gradient descent - cost by iteration')
            fig.canvas.manager.set_window_title('Gradient descent - cost by iteration')
            plt.show()

        return self.train_predictions, self.test_predictions

    def fit_hold_out(self, activation, learning_rate_init, max_iter, hidden_layer_sizes, alpha, plot_loss_curve=False):
        regression = MLPRegressor(activation=activation, learning_rate_init=learning_rate_init, max_iter=max_iter, hidden_layer_sizes=hidden_layer_sizes, alpha=alpha, random_state=1)

        x_train = self.prepared_dfs["x_train"].to_numpy(dtype=float)
        y_train = self.prepared_dfs["y_train"].to_numpy(dtype=float)
        x_cv = self.prepared_dfs["x_cv"].to_numpy(dtype=float)
        y_cv = self.prepared_dfs["y_cv"].to_numpy(dtype=float)
        x_test = self.prepared_dfs["x_test"].to_numpy(dtype=float)
        y_test = self.prepared_dfs["y_test"].to_numpy(dtype=float)
        regression.fit(x_train, y_train[:, 0])

        self.train_predictions = np.zeros(x_train.shape[0])
        self.cv_predictions = np.zeros(x_cv.shape[0])
        self.test_predictions = np.zeros(x_test.shape[0])
        self.train_predictions = regression.predict(x_train)
        self.cv_predictions = regression.predict(x_cv)
        self.test_predictions = regression.predict(x_test)
        self.train_predictions = self._reverse_transformation_on_prediction(np.array([self.train_predictions]).transpose(), self.y_shift_vector, self.y_scale_vector)
        self.cv_predictions = self._reverse_transformation_on_prediction(np.array([self.cv_predictions]).transpose(), self.y_shift_vector, self.y_scale_vector)
        self.test_predictions = self._reverse_transformation_on_prediction(np.array([self.test_predictions]).transpose(), self.y_shift_vector, self.y_scale_vector)

        print(self.y_df.iloc[self.hold_out_splits["cv"], :].to_numpy(dtype=float))
        print(self.cv_predictions)

        print("\tTraining set:")
        printAllRegressionMetrics(self.y_df.iloc[self.hold_out_splits["train"], :].to_numpy(dtype=float), self.train_predictions, x_train.shape[1])
        print("\tCV set:")
        printAllRegressionMetrics(self.y_df.iloc[self.hold_out_splits["cv"], :].to_numpy(dtype=float), self.cv_predictions, x_cv.shape[1])
        print("\tTest set:")
        printAllRegressionMetrics(self.y_df.iloc[self.hold_out_splits["test"], :].to_numpy(dtype=float), self.test_predictions, x_test.shape[1])

        if plot_loss_curve:
            fig, ax = plt.subplots()
            ax.plot(list(range(0, len(regression.loss_curve_))), regression.loss_curve_, label="NN learning curve")
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Cost')
            ax.set_title('Gradient descent - cost by iteration')
            fig.canvas.manager.set_window_title('Gradient descent - cost by iteration')
            plt.show()

        return self.train_predictions, self.cv_predictions, self.test_predictions

    def _reverse_transformation_on_prediction(self, array, mean, std):
        return np.exp(reverse_shift_and_scale(array, mean, std))

    def plot_learning_curves(self, starting_sample_number, group_split_number, data_preparation_kws, fit_kws, k_fold):
        self._plot_learning_curves(starting_sample_number, group_split_number, copy.deepcopy(self), data_preparation_kws, fit_kws, k_fold)