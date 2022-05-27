import copy

from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor

from src.EvaluationMetrics import *


class ClaimPredictionBase:
    def __init__(self, df, target_feature, numerical_features, categorical_features):
        self.df = df.copy()
        self.x_df = df.drop(columns=target_feature)
        self.y_df = df[target_feature].to_frame()
        self.target_feature = target_feature
        self.numerical_features = numerical_features.copy()
        self.categorical_features = categorical_features.copy()
        self.k_fold_splits = None
        self.hold_out_splits = None
        self.prepared_dfs = None
        self.train_predictions = None
        self.cv_predictions = None
        self.test_predictions = None
        self.x_shift_vector = 0
        self.x_scale_vector = 1
        self.y_shift_vector = 0
        self.y_scale_vector = 1

    def split_with_k_folds(self, n_splits):
        self.k_fold_splits = []
        kf = KFold(n_splits=n_splits)
        for train_indices, test_indices in kf.split(self.df):
            self.k_fold_splits.append({"train": train_indices, "test": test_indices})

    def split_with_holdout(self):
        m = self.df.shape[0]
        self.hold_out_splits = {}
        self.hold_out_splits["train"] = list(range(0, int(floor(m*0.6))))
        self.hold_out_splits["test"] = list(range(ceil(m*0.6), int(floor(m*0.8))))
        self.hold_out_splits["cv"] = list(range(ceil(m*0.8), m))

        self.hold_out_splits["test"] = self._get_indices_with_categorical_values_present_in_training_and_test_set(self.hold_out_splits["train"], self.hold_out_splits["test"])
        self.hold_out_splits["cv"] = self._get_indices_with_categorical_values_present_in_training_and_test_set(self.hold_out_splits["train"], self.hold_out_splits["cv"])

    def _get_indices_with_categorical_values_present_in_training_and_test_set(self, train_indices, test_indices):
        training_x_df = self.df.drop(columns=self.target_feature).iloc[train_indices, :].copy()
        test_x_df = self.df.drop(columns=self.target_feature).iloc[test_indices, :].copy()
        test_y_df = self.df[self.target_feature].to_frame().copy()

        column_names = list(self.df)
        categorical_features_indexes = [column_names.index(column) for column in self.categorical_features]

        unique_values_by_column = {}
        for column in self.categorical_features:
            unique_values_by_column[column] = np.unique(training_x_df[column])
        valid_test_indexes = []
        for index in test_indices:
            is_valid_index = True
            for column in self.categorical_features:
                if self.df[column].iloc[index] not in unique_values_by_column[column]:
                    is_valid_index = False
                    break
            if is_valid_index:
                valid_test_indexes.append(index)
        return valid_test_indexes

    def _log_transform(self, x_df, y_df):
        x_array = x_df.to_numpy()
        y_array = y_df.to_numpy()

        x_df = x_df.copy()
        y_df = y_df.copy()

        column_names = list(self.x_df.columns)
        numerical_features_indexes = [column_names.index(column) for column in self.numerical_features]

        x_df[self.numerical_features] = logFeatures(x_array[:, numerical_features_indexes].astype(float), zeroBufor=0.1)
        y_df[self.target_feature] = logFeatures(y_array.astype(float), zeroBufor=0.1)
        return x_df, y_df

    def _one_hot_encode_k_fold(self, x_df_train, x_df_test):
        x_df_train = oneHotEncode(x_df_train, self.categorical_features)
        x_df_test = oneHotEncodeWithGivenColumns(x_df_test, self.categorical_features, list(x_df_train.columns))
        return x_df_train, x_df_test

    def _one_hot_encode_hold_out(self, x_df_train, x_df_cv, x_df_test):
        x_df_train = oneHotEncode(x_df_train, self.categorical_features)
        x_df_cv = oneHotEncodeWithGivenColumns(x_df_cv, self.categorical_features, list(x_df_train.columns))
        x_df_test = oneHotEncodeWithGivenColumns(x_df_test, self.categorical_features, list(x_df_train.columns))
        return x_df_train, x_df_cv, x_df_test

    def _smooth_mean_target_encode_k_fold(self, x_train_df, y_train_df, x_test_df, m):
        x_train_df = x_train_df.copy()
        x_train_df[self.target_feature] = y_train_df[self.target_feature].to_frame().to_numpy()
        x_test_df = x_test_df.copy()
        total_train_mean = y_train_df.astype(float).mean()
        for column in self.categorical_features:
            x_train_df[column], transformation_dict = mean_target_smooth_encoding(x_train_df,  column, self.target_feature, m)
            x_test_df[column] = map_category_values(x_test_df[column], transformation_dict, total_train_mean)

        x_train_df = x_train_df.drop(columns=[self.target_feature]).astype(float)
        x_test_df = x_test_df.astype(float)
        return x_train_df, x_test_df

    def _smooth_mean_target_encode_hold_out(self, x_train_df, y_train_df, x_cv_df, x_test_df, m):
        x_train_df = x_train_df.copy()
        x_train_df[self.target_feature] = y_train_df[self.target_feature].to_frame().to_numpy()
        x_test_df = x_test_df.copy()
        x_cv_df = x_cv_df.copy()
        total_train_mean = y_train_df.astype(float).mean()
        for column in self.categorical_features:
            x_train_df[column], transformation_dict = mean_target_smooth_encoding(x_train_df,  column, self.target_feature, m)
            x_test_df[column] = map_category_values(x_test_df[column], transformation_dict, total_train_mean)
            x_cv_df[column] = map_category_values(x_cv_df[column], transformation_dict, total_train_mean)

        x_train_df = x_train_df.drop(columns=[self.target_feature]).astype(float)
        x_test_df = x_test_df.astype(float)
        x_cv_df = x_cv_df.astype(float)
        return x_train_df, x_cv_df, x_test_df

    def _normalise_features_k_fold(self, x_train_df, y_train_df, x_test_df, y_test_df, include_categorical_features=False):
        features = self.numerical_features.copy()
        if include_categorical_features:
            features.extend(self.categorical_features)

        x_train_df = x_train_df.copy()
        x_test_df = x_test_df.copy()
        x_train_df[features], self.x_shift_vector, self.x_scale_vector = normalize(x_train_df[features].to_numpy(dtype=float))
        x_test_df[features] = shift_and_scale_matrix(x_test_df[features].to_numpy(dtype=float), self.x_shift_vector, self.x_scale_vector)

        y_train_df = y_train_df.copy()
        y_test_df = y_test_df.copy()
        y_train_df[self.target_feature], self.y_shift_vector, self.y_scale_vector = normalize(y_train_df.to_numpy(dtype=float))
        y_test_df[self.target_feature] = shift_and_scale_matrix(y_test_df.to_numpy(dtype=float), self.y_shift_vector, self.y_scale_vector)

        return x_train_df, x_test_df, y_train_df, y_test_df

    def _normalise_features_hold_out(self, x_train_df, y_train_df, x_cv_df, y_cv_df, x_test_df, y_test_df, include_categorical_features=False):
        features = self.numerical_features.copy()
        if include_categorical_features:
            features.extend(self.categorical_features)

        x_train_df = x_train_df.copy()
        x_cv_df = x_cv_df.copy()
        x_test_df = x_test_df.copy()
        x_train_df[features], self.x_shift_vector, self.x_scale_vector = normalize(x_train_df[features].to_numpy(dtype=float))
        x_cv_df[features] = shift_and_scale_matrix(x_cv_df[features].to_numpy(dtype=float), self.x_shift_vector, self.x_scale_vector)
        x_test_df[features] = shift_and_scale_matrix(x_test_df[features].to_numpy(dtype=float), self.x_shift_vector, self.x_scale_vector)

        y_train_df = y_train_df.copy()
        y_cv_df = y_cv_df.copy()
        y_test_df = y_test_df.copy()
        y_train_df[self.target_feature], self.y_shift_vector, self.y_scale_vector = normalize(y_train_df.to_numpy(dtype=float))
        y_cv_df[self.target_feature] = shift_and_scale_matrix(y_cv_df.to_numpy(dtype=float), self.y_shift_vector, self.y_scale_vector)
        y_test_df[self.target_feature] = shift_and_scale_matrix(y_test_df.to_numpy(dtype=float), self.y_shift_vector, self.y_scale_vector)

        return x_train_df, x_cv_df, x_test_df, y_train_df, y_cv_df, y_test_df

    def prepare_data_k_fold(self, log_data, encoding, normalize_features):
        if not (encoding == "target" or encoding == "one-hot"):
            raise Exception("use existing encoder")

        prepared_dfs = []

        for splits in self.k_fold_splits:
            x_train_data = self.x_df.iloc[splits["train"], :].copy()
            y_train_data = self.y_df.iloc[splits["train"], :].copy()
            x_test_data = self.x_df.iloc[splits["test"], :].copy()
            y_test_data = self.y_df.iloc[splits["test"], :].copy()

            if encoding == "target":
                x_train_data, x_test_data = self._smooth_mean_target_encode_k_fold(x_train_data, y_train_data, x_test_data, 0.1)
            if log_data:
                x_train_data, y_train_data = self._log_transform(x_train_data, y_train_data)
                x_test_data, y_test_data = self._log_transform(x_test_data, y_test_data)
            if encoding == "one-hot":
                x_train_data, x_test_data = self._one_hot_encode_k_fold(x_train_data, x_test_data)
            if normalize_features:
                x_train_data, x_test_data, y_train_data, y_test_data = self._normalise_features_k_fold(x_train_data, y_train_data, x_test_data, y_test_data, True if encoding == "target" else False)

            prepared_dfs.append({"x_train": x_train_data, "y_train": y_train_data, "x_test": x_test_data, "y_test": y_test_data})

        self.prepared_dfs = prepared_dfs
        return prepared_dfs

    def prepare_data_hold_out(self, log_data, encoding, normalize_features):
        if not (encoding == "target" or encoding == "one-hot"):
            raise Exception("use existing encoder")

        x_train_data = self.x_df.iloc[self.hold_out_splits["train"], :].copy()
        y_train_data = self.y_df.iloc[self.hold_out_splits["train"], :].copy()
        x_test_data = self.x_df.iloc[self.hold_out_splits["test"], :].copy()
        y_test_data = self.y_df.iloc[self.hold_out_splits["test"], :].copy()
        x_cv_data = self.x_df.iloc[self.hold_out_splits["cv"], :].copy()
        y_cv_data = self.y_df.iloc[self.hold_out_splits["cv"], :].copy()

        if encoding == "target":
            x_train_data, x_cv_data, x_test_data = self._smooth_mean_target_encode_hold_out(x_train_data, y_train_data, x_cv_data, x_test_data, 0.01)
        if log_data:
            x_train_data, y_train_data = self._log_transform(x_train_data, y_train_data)
            x_cv_data, y_cv_data = self._log_transform(x_cv_data, y_cv_data)
            x_test_data, y_test_data = self._log_transform(x_test_data, y_test_data)
        if encoding == "one-hot":
            x_train_data, x_cv_data, x_test_data = self._one_hot_encode_hold_out(x_train_data, x_cv_data, x_test_data)
        if normalize_features:
            x_train_data, x_cv_data, x_test_data, y_train_data, y_cv_data, y_test_data = self._normalise_features_hold_out(x_train_data, y_train_data, x_cv_data, y_cv_data, x_test_data, y_test_data, True if encoding == "target" else False)

        prepared_dfs = {"x_train": x_train_data, "y_train": y_train_data, "x_cv": x_cv_data, "y_cv": y_cv_data, "x_test": x_test_data, "y_test": y_test_data}

        self.prepared_dfs = prepared_dfs
        return prepared_dfs

    def plot_predictions_and_values_hold_out(self):
        train_length = self.train_predictions.shape[0]
        test_length = self.test_predictions.shape[0]
        array = np.empty((train_length + test_length, 2), dtype=float)
        hue = np.empty((train_length + test_length), dtype='U30')

        array[0:train_length, 0] = self.y_df.iloc[self.hold_out_splits["train"], :].to_numpy()[:, 0]
        array[train_length:train_length+test_length, 0] = self.y_df.iloc[self.hold_out_splits["test"], :].to_numpy()[:, 0]
        array[0:train_length, 1] = self.train_predictions[:, 0]
        array[train_length:train_length+test_length, 1] = self.test_predictions[:, 0]
        hue[0:train_length] = "train set"
        hue[train_length:train_length+test_length] = "test set"

        data = pd.DataFrame(data=array, columns=["value", "predicted value"])
        plot_predictions_and_values(data, hue)

    def plot_predictions_and_values_k_fold(self):
        train_length = self.train_predictions.shape[0]
        test_length = self.test_predictions.shape[0]
        array = np.empty((train_length + test_length, 2), dtype=float)
        hue = np.empty((train_length + test_length), dtype='U30')

        array[0:train_length, 0] = self.y_df.to_numpy()[:, 0]
        array[train_length:train_length+test_length, 0] = self.y_df.to_numpy()[:, 0]
        array[0:train_length, 1] = self.train_predictions[:, 0]
        array[train_length:train_length+test_length, 1] = self.test_predictions[:, 0]
        hue[0:train_length] = "train set"
        hue[train_length:train_length+test_length] = "test set"

        data = pd.DataFrame(data=array, columns=["value", "predicted value"])
        plot_predictions_and_values(data, hue)

    def save_predictions(self, system_directory_path, cross_validation=False):
        savePredictions(self.df, self.target_feature, self.categorical_features, self.train_predictions, fr"{system_directory_path}\claim_training_set_predictions.csv", orderBy=-1)
        savePredictions(self.df, self.target_feature, self.categorical_features, self.test_predictions, fr"{system_directory_path}\claim_test_set_predictions.csv", orderBy=-1)
        if cross_validation:
            savePredictions(self.df, self.target_feature, self.categorical_features, self.cv_predictions, fr"{system_directory_path}\claim_cv_set_predictions.csv", orderBy=-1)

    def _plot_learning_curves(self, starting_sample_number, group_split_number, algorithm_instance, data_preparation_kws, fit_kws, is_k_fold):
        if algorithm_instance is None:
            algorithm_instance = {}
        print("Generating learning curve...")
        m, n = self.x_df.shape
        fig, ax = plt.subplots()

        sample_numbers = []
        training_costs = []
        cv_costs = []

        for trainingExamplesNumber in range(starting_sample_number, m + 1, ceil((m - starting_sample_number) / group_split_number)):
            print(f"\tTraining {trainingExamplesNumber}/{m} examples")

            df = self.df.iloc[:trainingExamplesNumber, :].copy()
            x_training_subset = self.x_df.iloc[:trainingExamplesNumber, :].copy()
            y_training_subset = self.y_df.iloc[:trainingExamplesNumber, :].copy()

            algorithm_instance.df = df
            algorithm_instance.x_df = x_training_subset
            algorithm_instance.y_df = y_training_subset
            if is_k_fold:
                algorithm_instance.split_with_k_folds(10)
                algorithm_instance.prepare_data_k_fold(**data_preparation_kws)
                train_predictions, cv_predictions = algorithm_instance.fit_k_fold(**fit_kws)

                sample_numbers.append(trainingExamplesNumber)
                training_costs.append(meanSquaredError(algorithm_instance.y_df.to_numpy(dtype=float), train_predictions))
                cv_costs.append(meanSquaredError(algorithm_instance.y_df.to_numpy(dtype=float), cv_predictions))
            else:
                algorithm_instance.split_with_holdout()
                algorithm_instance.prepare_data_hold_out(**data_preparation_kws)
                train_predictions, cv_predictions, test_predictions = algorithm_instance.fit_hold_out(**fit_kws)

                sample_numbers.append(trainingExamplesNumber)
                training_costs.append(meanSquaredError(algorithm_instance.y_df.iloc[algorithm_instance.hold_out_splits["train"], :].to_numpy(dtype=float), train_predictions))
                cv_costs.append(meanSquaredError(algorithm_instance.y_df.iloc[algorithm_instance.hold_out_splits["cv"], :].to_numpy(dtype=float), cv_predictions))


        ax.plot(sample_numbers, training_costs, label='training set')
        ax.plot(sample_numbers, cv_costs, label='cv set')
        ax.set_xlabel('training examples number')
        ax.set_ylabel('cost')
        ax.set_title('Learning Curves')
        ax.legend()

        plt.show()
