import os
import sys

from src.DataAnalysis import DataAnalysis
from src.cases.claim.ClaimDataProcessing import *
from src.cases.claim.ClaimPredictionGradientBoosting import ClaimPredictionGradientBoosting
from src.cases.claim.ClaimPredictionNeuralNetworks import ClaimPredictionNeuralNetwork
from src.cases.claim.ClaimPredictionRandomForest import ClaimPredictionRandomForest
from src.cases.claim.ClaimPredictionRidgeRegression import ClaimPredictionRidgeRegression


class PredictionPipeline:
    def __init__(self):
        self.system_abs_path = os.path.dirname(os.path.abspath(__file__)) + "\.."
        self.df = self._load_data()
        self.target_feature = "TotalClaimCost"
        self.numerical_features = ["PreviousClaimCount", "PreviousClaimCost", "Age"]
        self.categorical_features = ["CoverCauseDisplayName_2", "CoverCauseDisplayName_3", "ClaimantGender",
                                "ClaimantIsSmoker",
                                "MainDiagnoseCode", "BenefitCode", "IncidentCountryCode"]

    def _load_data(self):
        print(self.system_abs_path)
        df = pd.read_csv(fr"{self.system_abs_path}\data\nis\claims_shuffeled.csv",
                         dtype=str, encoding='cp1252')
        return df

    def shuffle_data(self, shuffle_number):
        df = shuffleDataFrame(self.df, shuffle_number)
        df.to_csv(fr"{self.system_abs_path}\nis\claims_shuffeled_new.csv", index_label=False)
        self.df = df
        return self.df

    def prepare_data(self):
        print(f"Data set pre-preparing shape {self.df.shape}")
        data_processing = ClaimDataProcessing(self.df.copy())
        data_processing.drop_redundant_columns()
        data_processing.fill_null_previous_claim_cost()
        data_processing.drop_nulls()
        data_processing.drop_claims_other_than_closed_and_column()
        data_processing.drop_zero_or_less_cost_claims()
        data_processing.drop_logarithmic_outliers("TotalClaimCost", 3)
        data_processing.split_cover_causes()
        data_processing.drop_CoverCauseDisplayName_1_travel_records_and_column()
        data_processing.calculate_age()
        data_processing.convert_diagnose_codes()

        # some experimental filtering
        #data_processing.choose_rows_with_n_most_numerous_values_of_parameter(10, "MainDiagnoseCode")
        #data_processing.drop_records_by_max_value(self.target_feature, 1000)

        # often used to get rid of too small groups of data
        #data_processing.drop_records_by_category_min_quantity(self.categorical_features, 20)

        data_processing.save_outliers(system_directory_path=fr"{self.system_abs_path}\data\claim_prediction")
        print(f"Data set post-preparing shape {data_processing.df.shape}")

        self.df = data_processing.df
        return self.df

    def exploratory_data_analysis(self):
        analysis = DataAnalysis(self.df.copy(), self.target_feature, self.numerical_features, self.categorical_features)

        analysis.plot_numerical_features_distribution()
        analysis.correlationMatrix()
        analysis.plot_numerical_features_pair_plots()
        
        analysis.plot_categorical_features_distribution()

        analysis.plot_cost_by_category("ClaimantIsSmoker", 60)
        analysis.plot_cost_by_category("ClaimantGender", 60)
        analysis.plot_cost_by_category("BenefitCode", 60)
        analysis.plot_cost_by_category("IncidentCountryCode", 60)
        analysis.plot_cost_by_category("CoverCauseDisplayName_2", 60)
        analysis.plot_cost_by_category("CoverCauseDisplayName_3", 60)
        analysis.plot_cost_by_category("MainDiagnoseCode", 60)

        analysis.plot_diagnoses_with_hue("IncidentCountryCode", 10, True)
        analysis.plot_diagnoses_with_hue("BenefitCode", 10, True)
        analysis.plot_diagnoses_with_hue("CoverCauseDisplayName_3", 10, True)

        # logarithmic distribution
        log_df = self.df.copy()
        log_df[self.numerical_features] = logFeaturesDataFrame(log_df[self.numerical_features], self.numerical_features, zeroBufor=0.1)
        log_analysis = DataAnalysis(log_df, self.target_feature, self.numerical_features, self.categorical_features)
        log_analysis.plot_numerical_features_distribution()

        # mean smooth encoding
        encoded_df = self.df.copy()
        for column in self.categorical_features:
            encoded_df[column] = mean_target_smooth_encoding(encoded_df, column, self.target_feature, m=0.1)[0]
        encoded_df = encoded_df.astype(float)
        new_numerical_features = self.numerical_features.copy()
        new_numerical_features.extend(self.categorical_features)
        encoded_analysis = DataAnalysis(encoded_df, self.target_feature, new_numerical_features, self.categorical_features)
        encoded_analysis.plot_pair_plot(6, 6, new_numerical_features, float)

        # normalised data for PCA
        encoded_normalized_df = self.df.copy()
        for column in self.categorical_features:
            encoded_normalized_df[column] = \
            mean_target_smooth_encoding(encoded_normalized_df, column, self.target_feature, m=0.1)[0]
        encoded_normalized_df = encoded_normalized_df.astype(float)
        encoded_normalized_df = pd.DataFrame(data=zScore(logFeaturesDataFrame(encoded_normalized_df, encoded_normalized_df.columns, zeroBufor=0.1))[0], columns=encoded_normalized_df.columns)
        # PCA with numerical
        encoded_normalized_analysis = DataAnalysis(encoded_normalized_df, self.target_feature, self.numerical_features, self.categorical_features)
        encoded_normalized_analysis.plot_2D_PCA_numerical_features()
        encoded_normalized_analysis.correlationMatrix()
        # PCA with all data
        encoded_normalized_analysis = DataAnalysis(encoded_normalized_df, self.target_feature, new_numerical_features, [])
        encoded_normalized_analysis.plot_2D_PCA_numerical_features()
        encoded_normalized_analysis.correlationMatrix()

    def train_ridge_regression(self, k_fold=True, learning_curve=False):
        claim_prediction_lasso_regression = ClaimPredictionRidgeRegression(self.df, self.target_feature, self.numerical_features, self.categorical_features)
        claim_prediction_lasso_regression.add_cost_per_previous_claim()
        claim_prediction_lasso_regression.add_bias()

        data_preparation_kws = {"log_data": True, "encoding": "one-hot", "normalize_features": True}

        ridge_parameters = {}
        ridge_parameters["alpha"] = 0.03
        ridge_parameters["iteration_number"] = 1000
        ridge_parameters["lamb"] = 0.000

        if k_fold:
            claim_prediction_lasso_regression.split_with_k_folds(10)
            claim_prediction_lasso_regression.prepare_data_k_fold(**data_preparation_kws)
            claim_prediction_lasso_regression.fit_k_fold(**ridge_parameters)
            claim_prediction_lasso_regression.plot_predictions_and_values_k_fold()
        else:
            claim_prediction_lasso_regression.split_with_holdout()
            claim_prediction_lasso_regression.prepare_data_hold_out(**data_preparation_kws)
            claim_prediction_lasso_regression.fit_hold_out(**ridge_parameters)
            claim_prediction_lasso_regression.plot_predictions_and_values_hold_out()

        if learning_curve:
            claim_prediction_lasso_regression.plot_learning_curves(1000, 30, data_preparation_kws, ridge_parameters, k_fold)

        claim_prediction_lasso_regression.save_predictions(system_directory_path=fr"{self.system_abs_path}\data\claim_prediction\predictions\ridge")

    def train_neural_network(self, k_fold=True, learning_curve=False):
        claim_prediction_neural_network = ClaimPredictionNeuralNetwork(self.df, self.target_feature, self.numerical_features, self.categorical_features)

        data_preparation_kws = {"log_data": True, "encoding": "one-hot", "normalize_features": True}

        neural_network_parameters = {}
        neural_network_parameters["activation"] = "logistic"
        neural_network_parameters["learning_rate_init"] = 0.01
        neural_network_parameters["max_iter"] = 1000
        neural_network_parameters["hidden_layer_sizes"] = (100, 100)
        neural_network_parameters["alpha"] = 0.006

        if k_fold:
            claim_prediction_neural_network.split_with_k_folds(10)
            claim_prediction_neural_network.prepare_data_k_fold(**data_preparation_kws)
            claim_prediction_neural_network.fit_k_fold(**neural_network_parameters)
            claim_prediction_neural_network.plot_predictions_and_values_k_fold()
        else:
            claim_prediction_neural_network.split_with_holdout()
            claim_prediction_neural_network.prepare_data_hold_out(**data_preparation_kws)
            claim_prediction_neural_network.fit_hold_out(**neural_network_parameters)
            claim_prediction_neural_network.plot_predictions_and_values_hold_out()

        if learning_curve:
            claim_prediction_neural_network.plot_learning_curves(1000, 30, data_preparation_kws, neural_network_parameters, k_fold)

        claim_prediction_neural_network.save_predictions(system_directory_path=fr"{self.system_abs_path}\data\claim_prediction\predictions\neural_network")

    def train_random_forest(self, k_fold=True, learning_curve=False):
        claim_prediction_random_forest = ClaimPredictionRandomForest(self.df, self.target_feature, self.numerical_features, self.categorical_features)

        data_preparation_kws = {"log_data": False, "encoding": "target", "normalize_features": False}

        random_forest_parameters = {}
        random_forest_parameters["n_estimators"] = 1000
        random_forest_parameters["random_state"] = 0
        random_forest_parameters["max_features"] = "log2"
        random_forest_parameters["max_depth"] = None
        random_forest_parameters["min_samples_split"] = 50
        random_forest_parameters["min_samples_leaf"] = 10

        if k_fold:
            claim_prediction_random_forest.split_with_k_folds(10)
            claim_prediction_random_forest.prepare_data_k_fold(**data_preparation_kws)
            claim_prediction_random_forest.fit_k_fold(**random_forest_parameters)
            claim_prediction_random_forest.plot_predictions_and_values_k_fold()
        else:
            claim_prediction_random_forest.split_with_holdout()
            claim_prediction_random_forest.prepare_data_hold_out(**data_preparation_kws)
            claim_prediction_random_forest.fit_hold_out(**random_forest_parameters)
            claim_prediction_random_forest.plot_predictions_and_values_hold_out()

        if learning_curve:
            claim_prediction_random_forest.plot_learning_curves(1000, 30, data_preparation_kws, random_forest_parameters, k_fold)

        claim_prediction_random_forest.save_predictions(system_directory_path=fr"{self.system_abs_path}\data\claim_prediction\predictions\random_forest")

    def train_gradient_boosting(self, k_fold=True, learning_curve=False):
        claim_prediction_gradient_boosting = ClaimPredictionGradientBoosting(self.df, self.target_feature, self.numerical_features, self.categorical_features)

        data_preparation_kws = {"log_data": False, "encoding": "target", "normalize_features": False}

        gradient_boosting_parameters = {}
        gradient_boosting_parameters["iterations"] = 100
        gradient_boosting_parameters["learning_rate"] = 0.1
        gradient_boosting_parameters["depth"] = 4
        gradient_boosting_parameters["l2_leaf_reg"] = 0


        if k_fold:
            claim_prediction_gradient_boosting.split_with_k_folds(10)
            claim_prediction_gradient_boosting.prepare_data_k_fold(**data_preparation_kws)
            claim_prediction_gradient_boosting.fit_k_fold(**gradient_boosting_parameters)
            claim_prediction_gradient_boosting.plot_predictions_and_values_k_fold()
        else:
            claim_prediction_gradient_boosting.split_with_holdout()
            claim_prediction_gradient_boosting.prepare_data_hold_out(**data_preparation_kws)
            claim_prediction_gradient_boosting.fit_hold_out(**gradient_boosting_parameters)
            claim_prediction_gradient_boosting.plot_predictions_and_values_hold_out()

        if learning_curve:
            claim_prediction_gradient_boosting.plot_learning_curves(1000, 30, data_preparation_kws, gradient_boosting_parameters, k_fold)

        claim_prediction_gradient_boosting.save_predictions(system_directory_path=fr"{self.system_abs_path}\data\claim_prediction\predictions\gradient_boosting")
