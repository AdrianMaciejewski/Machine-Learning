import math

import pandas as pd
import pandas as pn
import numpy as np
import category_encoders as ce
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder
from sklearn.feature_extraction import FeatureHasher
from math import *


def hashEncoding(df, column, outputNumber):
    hasher = FeatureHasher(n_features=outputNumber, input_type='string')
    return hasher.transform(df[column]).toarray()

def label_encoding(df, columns):
    le = LabelEncoder()
    dfCopy = df[columns].copy()
    for column in columns:
        le.fit(dfCopy[column].drop_duplicates())
        dfCopy[column] = le.transform(dfCopy[column])
    return dfCopy

def oridinalEncoding(df, columns, mapping):
    encoder = ce.OrdinalEncoder(cols=[columns], return_df=True, mapping=[mapping])
    df_train_transformed = encoder.fit_transform(df)
    return df_train_transformed


def oneHotEncoding(df, columns):
    # Create object for one-hot encoding
    encoder = ce.OneHotEncoder(cols=columns, handle_unknown='return_nan', return_df=True, use_cat_names=True)
    # Fit and transform Data
    data_encoded = encoder.fit_transform(df)
    return data_encoded#.to_numpy(dtype=float)

def oneHotEncode(df, columns_to_encode):
    """
    Encodes given columns using one hot encoding technique. Additionally creates column "<column>_other" for
    each column in order to have column for values that appeared in test dataset but not in training
    :param df:
    :param columns_to_encode:
    :return:
    """
    array = df.to_numpy(dtype='U30')
    original_columns = list(df.columns)

    one_hot_columns = []
    for column in columns_to_encode:
        unique_values = list(np.unique(df[column].to_numpy()))
        single_one_hot_column = []
        for value in unique_values:
            single_one_hot_column.append(f"{column}_{value}")
        single_one_hot_column.append(f"{column}_other")
        one_hot_columns.extend(single_one_hot_column)
    print(f"One hot columns number {len(one_hot_columns)}")

    not_encoded_features = df.drop(columns=columns_to_encode).to_numpy(dtype='U30')
    new_array_length = not_encoded_features.shape[1] + len(one_hot_columns)
    encoded_array = np.zeros((array.shape[0], new_array_length))
    for i in range(0, array.shape[0]):
        encoded_array[i, :not_encoded_features.shape[1]] = not_encoded_features[i, :]
        for column in columns_to_encode:
            original_column_index = original_columns.index(column)
            column_value = array[i, original_column_index]
            one_hot_column_index = one_hot_columns.index(f"{column}_{column_value}")
            encoded_array[i, one_hot_column_index + not_encoded_features.shape[1]] = 1

    not_encoded_columns = list(df.drop(columns=columns_to_encode).columns)
    all_columns = not_encoded_columns.copy()
    all_columns.extend(one_hot_columns)

    encoded_df = pd.DataFrame(data=encoded_array, columns=all_columns)
    return encoded_df


def oneHotEncodeWithGivenColumns(df, columns_to_encode, existing_columns):
    """
    This function requires presence of "<column>_other" for all columns, in already existing columns.
    It is necessary because column values during training and testing may differ.
    Already existing columns must be provided ordered with respect to dataframe.
    :param df:
    :param columns_to_encode:
    :return:
    """
    array = df.to_numpy(dtype='U30')
    original_columns = list(df.columns)

    not_encoded_features = df.drop(columns=columns_to_encode).to_numpy(dtype='U30')
    new_array_length = len(existing_columns)
    encoded_array = np.zeros((array.shape[0], new_array_length))
    for i in range(0, array.shape[0]):
        encoded_array[i, :not_encoded_features.shape[1]] = not_encoded_features[i, :]
        for column in columns_to_encode:
            original_column_index = original_columns.index(column)
            column_value = array[i, original_column_index]
            one_hot_column = f"{column}_{column_value}"
            not_present_values = 0
            if one_hot_column not in existing_columns:
                one_hot_column = f"{column}_other"
                not_present_values += 1
            existing_columns_index = existing_columns.index(one_hot_column)
            encoded_array[i, existing_columns_index] = 1

    encoded_df = pd.DataFrame(data=encoded_array, columns=existing_columns)
    return encoded_df


def binaryEncoding(df, columns):
    #Create object for binary encoding
    encoder = ce.BinaryEncoder(cols=columns, return_df=True)
    # Fit and transform Data
    data_encoded = encoder.fit_transform(df)
    return data_encoded


def meanTargetEncoding(df, columns, targetColumn):
    #Create object for target encoding
    encoder = ce.TargetEncoder(cols=columns)
    # Fit and transform Data
    data_encoded = encoder.fit_transform(df[columns], df[targetColumn])
    return data_encoded

def shuffleDataFrame(df, iterations):
    copy = df.copy()
    for i in range(0, iterations):
        copy = copy.sample(frac=1)
    return copy


def splitDataInto3Sets(x_df, y_df):
    m = x_df.shape[0]
    x_array = x_df.to_numpy()
    y_array = y_df.to_numpy()

    training_size = floor(0.6*m)
    training_start_index = 0
    training_end_index = training_start_index + training_size
    training_indexes = np.array(range(training_start_index, training_end_index))
    x_training_array = x_array[training_start_index:training_end_index, :]
    y_training_array = y_array[training_start_index:training_end_index]
    x_training_df = pd.DataFrame(data=x_training_array, columns=x_df.columns)
    y_training_df = pd.DataFrame(data=y_training_array, columns=y_df.columns)

    cv_size = floor(0.2*m)
    cv_start_index = training_end_index
    cv_end_index = cv_start_index + cv_size
    cv_indexes = np.array(range(cv_start_index, cv_end_index))
    x_cv_array = x_array[cv_start_index:cv_end_index, :]
    y_cv_array = y_array[cv_start_index:cv_end_index]
    x_cv_df = pd.DataFrame(data=x_cv_array, columns=x_df.columns)
    y_cv_df = pd.DataFrame(data=y_cv_array, columns=y_df.columns)

    test_size = floor(0.2*m)
    test_start_index = cv_end_index
    test_end_index = test_start_index + test_size
    test_indexes = np.array(range(test_start_index, test_end_index))
    x_test_array = x_array[test_start_index:test_end_index, :]
    y_test_array = y_array[test_start_index:test_end_index]
    x_test_df = pd.DataFrame(data=x_test_array, columns=x_df.columns)
    y_test_df = pd.DataFrame(data=y_test_array, columns=y_df.columns)

    return [x_training_df, y_training_df, training_indexes, x_cv_df, y_cv_df, cv_indexes, x_test_df, y_test_df, test_indexes]

def splitDataInto2Sets(x_df, y_df):
    m = x_df.shape[0]
    x_array = x_df.to_numpy()
    y_array = y_df.to_numpy()

    training_size = floor(0.7*m)
    training_start_index = 0
    training_end_index = training_start_index + training_size
    training_indexes = np.array(range(training_start_index, training_end_index))
    x_training_array = x_array[training_start_index:training_end_index, :]
    y_training_array = y_array[training_start_index:training_end_index, :]
    x_training_df = pd.DataFrame(data=x_training_array, columns=x_array.columns)
    y_training_df = pd.DataFrame(data=y_training_array, columns=y_array.columns)

    test_size = floor(0.3*m)
    test_start_index = training_end_index
    test_end_index = test_start_index + test_size
    test_indexes = np.array(range(test_start_index, test_end_index))
    x_test_array = x_array[test_start_index:test_end_index, :]
    y_test_array = y_array[test_start_index:test_end_index, :]
    x_test_df = pd.DataFrame(data=x_test_array, columns=x_array.columns)
    y_test_df = pd.DataFrame(data=y_test_array, columns=y_array.columns)

    return [x_training_df, y_training_df, training_indexes, x_test_df, y_test_df, test_indexes]

def placeColumnOnEnd(data, column):
    y = data[column].to_numpy(dtype=float)
    data = data.drop(columns=[column])
    data[column] = y
    return data


def infinitePrinting():
    np.set_printoptions(linewidth=np.inf, threshold=np.inf)
    pd.set_option('expand_frame_repr', None)
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', None)


def calcFeaturesQuantity(npArray):
    if len(npArray.shape) > 1:
        raise Exception("uniqueFeaturesQuantity only accepts one dimensional array")

    uniqueFeatures = np.unique(npArray)
    quantityByFeature = {}
    for feature in uniqueFeatures:
        quantityByFeature[feature] = np.sum(npArray == feature)
    quantityByFeature = {k: v for k, v in sorted(quantityByFeature.items(), key=lambda item: item[1], reverse=True)}
    return quantityByFeature


def savePredictions(originalDF, target_feature, categorical_features, predictions, path, orderBy):
    originalDF = placeColumnOnEnd(originalDF, target_feature)
    columns = list(originalDF.columns)
    columns.append("Prediction")
    columns.append("Absolute difference")
    columns.append("Percentage difference")
    originalDatasetArray = originalDF.to_numpy(dtype='U30')
    combinedData = np.empty((originalDatasetArray.shape[0], len(columns)), dtype="U30")
    combinedData[:, :len(columns) - 3] = originalDatasetArray
    combinedData[:, len(columns) - 3] = np.around(predictions, 2)[:, 0]
    combinedData[:, len(columns) - 2] = np.around(np.abs(combinedData[:, -3].astype(float) - combinedData[:, -4].astype(float)), 2)
    combinedData[:, len(columns) - 1] = np.around(combinedData[:, - 2].astype(float) / combinedData[:, -4].astype(float) * 100, 2)
    combinedData = combinedData[combinedData[:, orderBy].astype(float).argsort()]

    categorical_features_indexes = [columns.index(column) for column in categorical_features]
    combinedDataWithSuperCategory = groupBySuperCategories(combinedData, categorical_features_indexes)
    combinedDataWithSuperCategorySorted = sortByColumnUniqueValuesQuantity(combinedDataWithSuperCategory, combinedDataWithSuperCategory.shape[1] - 1, orderBy - 1 if orderBy < 0 else orderBy + 1)
    columns.append("SuperCategory")

    combinedDataDF = pd.DataFrame(data=combinedDataWithSuperCategorySorted, columns=columns)
    combinedDataDF.to_csv(path, index=False)


def mean_target_smooth_encoding(df, column, targetColumn, m):
    dfCopy = df.copy()
    dfCopy[targetColumn] = dfCopy[targetColumn].astype(float)

    totalMean = dfCopy[targetColumn].mean()
    grouped_data = dfCopy.groupby(column)
    agg = grouped_data[targetColumn].agg(["count", "mean"])
    counts = agg["count"]
    categoryMeans = agg["mean"]

    smooth = (counts * categoryMeans + m * totalMean) / (counts + m)
    transformation_dict = smooth.to_dict()

    return df[column].map(smooth).to_numpy(dtype=float), transformation_dict


def map_category_values(series, transformation_dict, total_mean):
    new_series = series.copy()
    for i in range(0, new_series.shape[0]):
        try:
            new_series.iloc[i] = transformation_dict[new_series.iloc[i]]
        except KeyError:
            new_series.iloc[i] = total_mean

    return new_series.to_numpy(dtype=float)


def addPolynomialFeatures(df, degree):
    quad = PolynomialFeatures(degree=degree, interaction_only=True)
    x_quad = quad.fit_transform(df)
    return pd.DataFrame(data=x_quad)


def drop_records_by_category_min_quantity(df, column, min_category_quantity):
    count = df.shape[0]
    array = df[column].to_numpy(dtype='U30')

    unique_values_before_filtering = np.unique(df[column].to_numpy(dtype='U30')).shape[0]
    quantity_by_value = calcFeaturesQuantity(array)
    valid_indexes = []
    for i in range(0, count):
        if quantity_by_value[array[i]] >= min_category_quantity:
            valid_indexes.append(i)
    valid_data_array = df.to_numpy()[valid_indexes, :]
    df = pd.DataFrame(data=valid_data_array, columns=df.columns)
    unique_values_after_filtering = np.unique(df[column].to_numpy()).shape[0]
    dropped_categories_number = unique_values_before_filtering - unique_values_after_filtering
    print(f"{column}: {unique_values_before_filtering} - {dropped_categories_number} = {unique_values_after_filtering} (all categorical values) - (dropped categorical values with quantity less than {min_category_quantity}) = (remaining categorical values) ({count-len(valid_indexes)} records)")
    return df, dropped_categories_number


def drop_records_by_category_coverage(df, column, min_category_coverage):
    count = df.shape[0]
    array = df[column].to_numpy(dtype='U30')

    unique_values_before_filtering = np.unique(df[column].to_numpy()).shape[0]
    quantity_by_value = calcFeaturesQuantity(df[column].to_numpy(dtype='U30'))
    valid_indexes = []
    for i in range(0, count):
        if quantity_by_value[array[i]] / count >= min_category_coverage:
            valid_indexes.append(i)
    valid_data_array = df.to_numpy()[valid_indexes, :]
    df = pd.DataFrame(data=valid_data_array, columns=df.columns)
    unique_values_after_filtering = np.unique(df[column].to_numpy()).shape[0]
    dropped_categories_number = unique_values_before_filtering - unique_values_after_filtering
    print(f"{dropped_categories_number} unique values with coverage less than {min_category_coverage} dropped for {column} - {count - len(valid_indexes)} records")
    return df


def drop_records_by_max_value(df, column, max_value):
    count = df.shape[0]
    array = df[column].to_numpy(dtype=float)

    valid_indexes = []
    for i in range(0, count):
        if array[i] <= max_value:
            valid_indexes.append(i)
    valid_data_array = df.to_numpy()[valid_indexes, :]
    df = pd.DataFrame(data=valid_data_array, columns=df.columns)
    print(f"Records with values bigger than {max_value} dropped for {column} - {count-len(valid_indexes)} records")
    return df


def zScore(npArray):
    m = npArray.shape[0]

    featureMeanAverage = np.array([np.mean(npArray, axis=0)])
    featureStandardDeviation = np.array([np.std(npArray, axis=0)])

    for i in range(0, featureStandardDeviation.shape[1]):
        if featureStandardDeviation[0, i] == 0:
            featureStandardDeviation[0, i] = 1

    z_score_array = np.divide(npArray-np.tile(featureMeanAverage, (m, 1)), np.tile(featureStandardDeviation, (m, 1)))
    return z_score_array, featureMeanAverage, featureStandardDeviation


def shift_and_scale_matrix(npArray, feature_shift, feature_scale):
    m = npArray.shape[0]
    normalisedMatrix = np.divide(npArray-np.tile(feature_shift, (m, 1)), np.tile(feature_scale, (m, 1)))
    return normalisedMatrix


def normalize(npArray):
    m, n = npArray.shape

    features_min = np.array([np.amin(npArray, axis=0)])
    features_max = np.array([np.amax(npArray, axis=0)])
    difference = features_max - features_min
    for i in range(0, difference.shape[1]):
        if difference[0, i] == 0:
            difference[0, i] = 1

    normalised_array = np.divide(npArray-np.tile(features_min, (m, 1)), np.tile(difference, (m, 1)))
    return normalised_array, features_min, difference


def reverse_shift_and_scale(np_array, shift, scale):
    m = np_array.shape[0]
    return np_array * np.tile(scale, (m, 1)) + np.tile(shift, (m, 1))


def logFeaturesDataFrame(df, features, zeroBufor):
    return logFeatures(df[features].to_numpy(dtype=float), zeroBufor)


def logFeatures(npArray, zeroBufor):
    xData = npArray.astype(float)
    zeroIndexes = np.where(xData == 0)[0]
    xData[zeroIndexes] += zeroBufor
    return np.log(xData)


def featuresAboveCorrelationThreshold(df, columnIndex, threshold):
    corrColumn = np.absolute(df.corr().round(3).iloc[:, columnIndex])
    columnIndexes = list(np.where(corrColumn >= threshold)[0])
    columnIndexes.remove(columnIndex)
    data = df.iloc[:, columnIndexes]
    if math.isnan(corrColumn.iloc[0]):
        data.insert(0, column="bias", value=1)
    return data


def createOtherCategoryByTotalCoverage(dfColumn, totalCoverageThreshold):
    count = dfColumn.shape[0]
    newColumn = dfColumn.copy()

    quantityByFeature = calcFeaturesQuantity(dfColumn.to_numpy())
    uniqueFeatures = list(quantityByFeature.keys())
    quantities = list(quantityByFeature.values())
    totalCovedSamples = 0
    for i in range(0, len(quantityByFeature.items())):
        if totalCovedSamples / count < totalCoverageThreshold:
            totalCovedSamples += quantities[i]
        else:
            newColumn.loc[dfColumn == uniqueFeatures[i]] = "Other"
    return newColumn


def create_other_category_by_category_coverage(column, min_category_coverage):
    count = column.shape[0]

    quantity_by_value = calcFeaturesQuantity(column)
    invalid_value_quantities = []
    for quantity in quantity_by_value.values():
        if quantity / count < min_category_coverage:
            invalid_value_quantities.append(quantity)
    invalid_indexes = []
    for i in range(0, count):
        if quantity_by_value[column[i]] / count < min_category_coverage:
            invalid_indexes.append(i)
    column[invalid_indexes] = "other_transformed"
    print(f"{len(invalid_value_quantities)} unique values with coverage less than {min_category_coverage} transformed to 'other_transformed' - {sum(invalid_value_quantities)} records")
    return column


def create_other_category_by_category_min_quantity(column, min_category_quantity):
    count = column.shape[0]

    unique_values_before_filtering = np.unique(column).shape[0]
    quantity_by_value = calcFeaturesQuantity(column)
    valid_indexes = []
    for i in range(0, count):
        if quantity_by_value[column[i]] >= min_category_quantity:
            valid_indexes.append(i)
    column[valid_indexes] = "other_transformed"
    unique_values_after_filtering = np.unique(column).shape[0]
    dropped_categories_number = unique_values_before_filtering - unique_values_after_filtering
    print(f"{dropped_categories_number} unique values with quantity less than {min_category_quantity} transformed to 'other_transformed' - {count - len(valid_indexes)} records")
    return column


def choose_rows_with_n_most_numerous_values_of_parameter(df, n_rows, column):
    column_array = df[column].to_numpy(dtype='U30')
    quantity_by_category_value = calcFeaturesQuantity(column_array)
    category_values = list(quantity_by_category_value.keys())
    valid_indexes = []
    for i in range(0, n_rows):
        valid_indexes.extend(np.where(column_array == category_values[i])[0])
    array = df.to_numpy(dtype='U30')[valid_indexes, :]
    filtered_df = pd.DataFrame(data=array, columns=df.columns)
    return filtered_df


def constructSuperCategoryCode(superCategoryAsArray):
    return "".join(f"{subCategory}___" if np.where(superCategoryAsArray == subCategory)[-1][-1] != superCategoryAsArray.shape[0] - 1 else f"{subCategory}" for subCategory in superCategoryAsArray)


def getUniqueSuperCategoriesAsSubCategoryArrays(npArray, columnNameIndexes):
    print(f"Searching for unique super categories of {columnNameIndexes}...")
    uniqueSuperCategories = np.empty([npArray.shape[0], len(columnNameIndexes) + 1], dtype=np.dtype("U100"))

    size = 0
    for superCategory in npArray[:, columnNameIndexes]:
        superCategoryCode = constructSuperCategoryCode(superCategory)
        if np.where(uniqueSuperCategories[:, -1] == superCategoryCode)[0].shape[0] == 0:
            uniqueSuperCategories[size, :-1] = superCategory
            uniqueSuperCategories[size, -1] = superCategoryCode
            size += 1

    uniqueSuperCategories = uniqueSuperCategories[0: size, :-1]

    print(f"Found { uniqueSuperCategories.shape[0]} unique super categories")
    return uniqueSuperCategories


def groupBySuperCategories(npArray, superCategoryColumnIndexes):
    print("Finding number of examples with super categories...")

    superCategories = getUniqueSuperCategoriesAsSubCategoryArrays(npArray, superCategoryColumnIndexes)
    processedValues = np.empty([npArray.shape[0], npArray.shape[1]+1]).astype(np.dtype("U100"))
    size = 0

    index = 0
    for superCategory in superCategories:
        examplesWithSuperCategory = getExamplesWithSuperCategory(npArray, superCategoryColumnIndexes, superCategory)
        superCategoryCode = constructSuperCategoryCode(superCategory)
        additionalSize = examplesWithSuperCategory.shape[0]
        processedValues[size:size+additionalSize, :-1] = examplesWithSuperCategory
        processedValues[size:size+additionalSize, -1] = superCategoryCode
        size += additionalSize

        if index % 1000 == 0:
            print(f"Finding number of examples with super categories in range {index+1}-{index+1000}")
        index += 1

    return processedValues


def sortByColumnUniqueValuesQuantity(npArray, primaryColumnIndex, secondaryColumnIndex):
    print("Ordering super categories by quantity in descending order...")
    npArrayLength = npArray.shape[0]

    primaryColumnValues = npArray[:, primaryColumnIndex]
    uniquePrimaryColumnValues = np.unique(primaryColumnValues)
    uniquePrimaryColumnValuesPosition = np.zeros([uniquePrimaryColumnValues.shape[0], 3], np.int) #startIndex(inclusive), endIndex(exclusive), length

    size = 0
    for index in range(0, npArrayLength):
        if index != 0 and (primaryColumnValues[index - 1] != primaryColumnValues[index] or index + 1 == npArrayLength):
            uniquePrimaryColumnValuesPosition[size, 1] = index
            uniquePrimaryColumnValuesPosition[size, 2] = uniquePrimaryColumnValuesPosition[size, 1] - uniquePrimaryColumnValuesPosition[size, 0]
            if size + 1 != uniquePrimaryColumnValues.shape[0]:
                uniquePrimaryColumnValuesPosition[size + 1, 0] = index
                size += 1
    uniquePrimaryColumnValuesPosition[size, 1] = npArrayLength
    uniquePrimaryColumnValuesPosition[size, 2] = uniquePrimaryColumnValuesPosition[size, 1] - uniquePrimaryColumnValuesPosition[size, 0]

    sortedArray = np.empty_like(npArray)
    sortedIndexes = np.flip(uniquePrimaryColumnValuesPosition[:, 2].astype(np.int)[:].argsort())
    size = 0
    for index in sortedIndexes:
        categorySamples = npArray[uniquePrimaryColumnValuesPosition[index, 0]:uniquePrimaryColumnValuesPosition[index, 1], :]
        secondarySortedIndexes = np.flip(categorySamples[:, secondaryColumnIndex].astype(np.float)[:].argsort())
        sortedArray[size:size + categorySamples.shape[0], :] = categorySamples[secondarySortedIndexes, :]
        size += categorySamples.shape[0]

    return sortedArray


def getExamplesWithSuperCategory(npArray, superCategoryColumnsIndexes, superCategory):
    rowsWithSuperCategoryBoolList = npArray[:, superCategoryColumnsIndexes[0]] == superCategory[0]
    for index in range(1, len(superCategoryColumnsIndexes)):
        rowsWithSuperCategoryBoolList = rowsWithSuperCategoryBoolList & (npArray[:, superCategoryColumnsIndexes[index]] == superCategory[index])
    return npArray[rowsWithSuperCategoryBoolList, :]

