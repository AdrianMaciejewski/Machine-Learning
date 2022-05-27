import numpy as np

from src.DataProcessing import *


class ClaimDataProcessing:
    def __init__(self, df):
        self.df = df
        self.outliers = None

    def drop_redundant_columns(self):
        data = self.df.copy()
        data = data.drop(columns=["id", "MainDiagnoseName", "AlternativeDiagnoseCode", "AlternativeDiagnoseName", "Name"])
        self.df = data
        return self.df

    def fill_null_previous_claim_cost(self):
        data = self.df.copy()
        data["PreviousClaimCost"] = data["PreviousClaimCost"].fillna(0)
        self.df = data
        return self.df

    def drop_nulls(self):
        data = self.df.copy()
        data = data.dropna()
        print(f"Dropped { self.df.shape[0] - data.shape[0]} records with null values:")
        print(f"{np.sum(self.df.isna())}")
        self.df = data
        return self.df

    def drop_claims_other_than_closed_and_column(self):
        data = self.df.copy()
        data = data.drop(index=data.loc[data["ClaimStatus"] != "Closed"].index)
        data = data.drop(columns=["ClaimStatus"])
        print(f"Dropped { self.df.shape[0] - data.shape[0]} records with denied claim")
        self.df = data
        return self.df

    def drop_CoverCauseDisplayName_1_travel_records_and_column(self):
        data = self.df.copy()
        data = data.drop(index=data.loc[data["CoverCauseDisplayName_1"] == "Travel"].index)
        data = data.drop(columns=["CoverCauseDisplayName_1"])
        print(f"Dropped { self.df.shape[0] - data.shape[0]} records with Travel value in CoverCauseDisplayName_1")
        self.df = data
        return self.df

    def drop_zero_or_less_cost_claims(self):
        data = self.df.copy()
        data = data.drop(index=data.loc[data["TotalClaimCost"] <= "0"].index)
        print(f"Dropped { self.df.shape[0] - data.shape[0]} records with claim total cost less than or equal 0")
        self.df = data
        return self.df

    def split_cover_causes(self):
        data = self.df.copy()
        tree_depth = 3
        cover_causes = data["CoverCauseDisplayName"]
        split_cover_causes = np.empty((cover_causes.shape[0], tree_depth), dtype='U30')
        for i in range(0, cover_causes.shape[0]):
            splitString = np.array(cover_causes.iloc[i].split('\\'))
            split_cover_causes[i, :splitString.shape[0]] = splitString
            split_cover_causes[i, splitString.shape[0]:] = '-'

        for i in range(0, split_cover_causes.shape[1]):
            data[f"CoverCauseDisplayName_{i + 1}"] = split_cover_causes[:, i]
        data = data.drop(columns=["CoverCauseDisplayName"])
        self.df = data
        return self.df

    def calculate_age(self):
        data = self.df.copy()
        incident_dates = pd.to_datetime(data["IncidentDate"])
        date_of_birth = pd.to_datetime(data["ClaimantDateOfBirth"])
        age = np.around((incident_dates - date_of_birth) / np.timedelta64(1, 'Y'), 1)
        data[f"Age"] = age
        data = data.drop(columns=["IncidentDate"])
        data = data.drop(columns=["ClaimantDateOfBirth"])
        self.df = data
        return self.df

    def drop_rows_below_cost(self, cost):
        self.df = self.df.drop(self.df.loc[self.df["TotalClaimCost"].astype(float) < cost].index)

    def drop_logarithmic_outliers_for_separate_category_values(self, column, zero_buffer=0.1, std_threshold=3):
        all_outliers_number = 0
        unique_values = np.unique(self.df[column].to_numpy(dtype='U30'))
        for value in unique_values:
            data = self.df["TotalClaimCost"].loc[self.df[column] == value]
            outliers_indexes = self.get_logarithmic_outlier_indexes(data, np.log(zero_buffer), 14, zero_buffer, std_threshold)
            self.df = self.df.drop(self.df.index[outliers_indexes])
            all_outliers_number += len(outliers_indexes)
            print(all_outliers_number, value)
        print(f"Dropped {all_outliers_number} outliers by separate category values for {column}")

    def drop_logarithmic_outliers(self, column, zero_buffer=0.1, std_threshold=3):
        outliers_indexes = []
        outliers_indexes.extend(i for i in self.get_logarithmic_outlier_indexes(self.df[column], np.log(zero_buffer), 14, zero_buffer, std_threshold) if i not in outliers_indexes)
        print(f"Dropped {len(outliers_indexes)} outliers for {column}")
        self.outliers = self.df.iloc[outliers_indexes, :]
        self.df = self.df.drop(self.df.index[outliers_indexes])

    def get_logarithmic_outlier_indexes(self, data, x_log_min, x_log_max, zero_buffer, std_threshold):
        feature_data = data.to_numpy(dtype=float)
        feature_data = logFeatures(feature_data, zeroBufor=zero_buffer)
        value_in_rage_indexes = np.array(np.where((feature_data >= x_log_min) & (feature_data <= x_log_max))[0])
        mean = np.mean(feature_data[value_in_rage_indexes])
        std = np.std(feature_data[value_in_rage_indexes])
        z_scores = (feature_data - mean) / std
        outliers = []
        for i in range(0, data.shape[0]):
            if abs(z_scores[i]) > std_threshold:
                outliers.append(i)
        return outliers

    def save_outliers(self, system_directory_path):
        if self.outliers is not None:
            self.outliers.to_csv(fr"{system_directory_path}\claim_logarythmic_outliers.csv", index=False)
        else:
            print("Can not save outliers - self.Outliers is None")

    def choose_rows_with_n_most_numerous_values_of_parameter(self, n_rows, column):
        self.df = choose_rows_with_n_most_numerous_values_of_parameter(self.df, n_rows, column)

    def create_other_category_by_category_coverage(self, features_to_check, min_total_coverage):
        for category in features_to_check:
            self.df[category] = create_other_category_by_category_coverage(self.df[category].to_numpy(dtype='U30'), min_category_coverage=min_total_coverage)

    def create_other_category_by_category_min_quantity(self, features_to_check, min_quantity):
        for category in features_to_check:
            self.df[category] = create_other_category_by_category_min_quantity(self.df[category].to_numpy(dtype='U30'), min_category_quantity=min_quantity)

    def drop_records_by_category_coverage(self, features_to_check, min_total_coverage):
        for category in features_to_check:
            self.df = drop_records_by_category_coverage(self.df, category, min_category_coverage=min_total_coverage)

    def drop_records_by_category_min_quantity(self, features_to_check, min_quantity):
        starting_categories_number = 0
        starting_records_number = self.df.shape[0]
        all_dropped_categories_number = 0

        for category in features_to_check:
            starting_categories_number += np.unique(self.df[category].to_numpy(dtype='U30')).shape[0]
            self.df, dropped_categories_number = drop_records_by_category_min_quantity(self.df, category, min_category_quantity=min_quantity)
            all_dropped_categories_number += dropped_categories_number
        all_dropped_records_number = starting_records_number - self.df.shape[0]

        print(f"Dropped {all_dropped_categories_number} out of {starting_categories_number} unique categorical values - remaining {starting_categories_number - all_dropped_categories_number}")
        print(f"Dropped {all_dropped_records_number} out of {starting_records_number} records - remaining {starting_records_number - all_dropped_records_number}")

    def drop_records_by_max_value(self, feature, max_value):
        starting_records_number = self.df.shape[0]
        self.df = drop_records_by_max_value(self.df, feature, max_value=max_value)
        end_records_number = self.df.shape[0]
        print(f"Dropped {starting_records_number - end_records_number} records with cost higher than {max_value} out of {starting_records_number} records - remaining {end_records_number}")

    def convert_diagnose_codes(self):
        diagnose_codes = self.df["MainDiagnoseCode"]
        converted_diagnose_codes = np.empty((diagnose_codes.shape[0]), dtype=float)
        for i in range(0, diagnose_codes.shape[0]):
            code = diagnose_codes.iloc[i]
            if ord(code[0]) in range(48, 58):
                converted_diagnose_codes[i] = int(floor(float(code)))
            else:
                converted_diagnose_codes[i] = int(floor(float(code[1:])))
        self.df["MainDiagnoseCode"] = converted_diagnose_codes
        return converted_diagnose_codes.astype('U30')