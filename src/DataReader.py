import math

import numpy.random
import pandas as pn
import numpy as np
from math import *

def  readData(fileName):
    print(f"Loading data from {fileName}")
    dataFrame = pn.read_csv(fileName)
    print("Data loaded")

    print("Converting to numpy")
    data = dataFrame.to_numpy()
    print("Shuffling matrix rows")
    np.random.shuffle(data)

    print("Splitting data into sets")
    m = data.shape[0]
    XTrainingSet = data[:floor(0.6*m), 1:-1]
    YTrainingSet = np.array([data[:floor(0.6*m), -1]]).transpose()
    XCrossValidationSet = data[XTrainingSet.shape[0]:floor(0.8*m), 1:-1]
    YCrossValidationSet = np.array([data[YTrainingSet.shape[0]:floor(0.8*m), -1]]).transpose()
    XTestSet = data[(XCrossValidationSet.shape[0] + XTrainingSet.shape[0]):m, 1:-1] #XTestSet = data[XCrossValidationSet.shape[0]:m, 0:-1]
    YTestSet = np.array([data[(YCrossValidationSet.shape[0] + YTrainingSet.shape[0]):m, -1]]).transpose() #YTestSet = np.array([data[YCrossValidationSet.shape[0]:m, -1]]).transpose()

    return [XTrainingSet, YTrainingSet, XCrossValidationSet, YCrossValidationSet, XTestSet, YTestSet]

def sanitiseData(fileName):
    np.set_printoptions(linewidth=np.inf, threshold=np.inf)
    pn.set_option('expand_frame_repr', False)
    pn.set_option('display.max_rows', 20)

    dataFrame = pn.read_csv(fileName, na_values=[], keep_default_na=False,  na_filter=False, dtype=np.str)

    #get rid of unwanted data
    redundantColumns = ["MainDiagnoseName", "AlternativeDiagnoseCode", "AlternativeDiagnoseName", "Name"]
    dataFrame = dataFrame.drop(columns=redundantColumns)
    claimsWithoutCountryCode = dataFrame.loc[dataFrame["IncidentCountryCode"] == ""].index
    dataFrame = dataFrame.drop(index=claimsWithoutCountryCode)
    notPaidClaims = dataFrame.loc[(dataFrame["TotalClaimCost"] == "NULL") | (dataFrame["TotalClaimCost"] == "0")].index
    dataFrame = dataFrame.drop(index=notPaidClaims)
    deniedClaims = dataFrame.loc[dataFrame["ClaimStatus"] == "Denied"].index
    dataFrame = dataFrame.drop(index=deniedClaims)
    dataFrame = dataFrame.drop(columns="ClaimStatus")

    #replace necessary fields
    dataFrame["PreviousClaimCost"] = dataFrame["PreviousClaimCost"].replace("NULL", 0)


    columnsContainingSubCategories = ["CoverCauseDisplayName", "MainDiagnoseCode", "BenefitCode", "IncidentCountryCode", "ClaimantIsSmoker"]

    #Unique values from each column
    uniqueColumnsSubCategories = {}
    for column in columnsContainingSubCategories:
        uniqueColumnsSubCategories[column] = __findColumnSubCategories(dataFrame, column)

    #Values from each column that occure more often, and (sum of their occuriences)/(all rows in the column) >= coverage. Left 1-conerage is discarded as being considered as marginal influence
    coverage = 0.96
    chosenColumnsSubCategoriesQuantities = {}
    for column in columnsContainingSubCategories:
        chosenColumnsSubCategoriesQuantities[column] = __getFeaturesByCoverage(dataFrame, column, coverage)

    #Names of new columns
    sanitisedColumnNames = ["id", "PreviousClaimCount", "PreviousClaimCost"]
    for columnSubCategoriesQuantities in chosenColumnsSubCategoriesQuantities.values():
        sanitisedColumnNames.extend(columnSubCategoriesQuantities.keys())
    sanitisedColumnNames.append("TotalClaimCost")

    #processedValues = __getProcessedMatrixWithSeparateCategoriesColumns(dataFrame, sanitisedColumnNames, chosenCoverCausesQuantities, chosenDiagnosesQuantities, chosenBenefitsQuantities, chosenCountriesQuantities, chosenIsSmokersQuantities)
    #processedDataFrame = pn.DataFrame(data=processedValues, columns=sanitisedColumnNames)
    #__printSummaryOfCreatingSeparateCategories(dataFrame, processedDataFrame, uniqueCoverCausesDisplayNames,
    #                                           uniqueMainDiagnoseCodes, uniqueBenefitCodes, uniqueIncidentCountryCodes,
    #                                           uniqueClaimantIsSmoker, sanitisedColumnNames,
    #                                           chosenCoverCausesQuantities, chosenDiagnosesQuantities,
    #                                           chosenBenefitsQuantities, chosenCountriesQuantities,
    #                                           chosenIsSmokersQuantities, coverage)
    #print(type(dataFrame[columnsContainingSubCategories[0]].iloc[0]))
    #print(type(dataFrame[columnsContainingSubCategories[1]].iloc[1]))
    #print(type(dataFrame[columnsContainingSubCategories[2]].iloc[2]))
    #print(type(dataFrame[columnsContainingSubCategories[3]].iloc[3]))
    #print(type(dataFrame[columnsContainingSubCategories[4]].iloc[4]))
    #print(dataFrame[columnsContainingSubCategories])
    #pn.unique(dataFrame[columnsContainingSubCategories])
    #print(__getUniqueSuperCategoriesAsSubCategoryArrays(dataFrame, columnsContainingSubCategories))
    coverage = 0.96
    #superCategories, superCategoriesQuantities = __getSuperCategoriesByCoverage(dataFrame, columnsContainingSubCategories, coverage)
    minSampleNumber = 10
    finalColumns = ["id", "PreviousClaimCount", "PreviousClaimCost", "SuperCategory", "CoverCauseDisplayName", "MainDiagnoseCode", "BenefitCode", "IncidentCountryCode", "ClaimantIsSmoker", "TotalClaimCost"]
    processedDataFrame, classificationCodes = __createSuperCategoryDataFrame(dataFrame, columnsContainingSubCategories, minSampleNumber, finalColumns)
    # = pn.DataFrame(data=processedValues, columns=["id", "PreviousClaimCount", "PreviousClaimCost", "SuperCategory", "TotalClaimCost"])

    #print(processedDataFrame)

    processedDataFrame.to_csv("sanitised_data3.csv", index=False)


def __findColumnSubCategories(dataFrame, columnName):
    return np.unique(dataFrame[columnName])

def __getUniqueSuperCategoriesAsSubCategoryArrays(dataFrame, columnNames):
    print(f"Searching for unique super categories of {columnNames}...")
    superCategories = {}

    for superCategory in dataFrame[columnNames].to_numpy()[:, :]:
        classificationCode = "".join(f"{subCategory}-&-" if list(superCategory).index(subCategory) != len(superCategory)-1 else f"{subCategory}" for subCategory in superCategory)
        if classificationCode not in superCategories:
            superCategories[classificationCode] = superCategory
    uniqueCategories = np.array(list(superCategories.values()))
    print(f"Found { uniqueCategories.shape[0]} unique super categories")
    return uniqueCategories

def __getFeaturesByCoverage(dataFrame, columnName, coverage):
    uniqueColumnRows = __findColumnSubCategories(dataFrame, columnName)

    uniqueColumnRowValuesQuantities = {}
    for columnRowValue in uniqueColumnRows:
        uniqueColumnRowValuesQuantities[columnRowValue] = dataFrame.loc[dataFrame[columnName] == columnRowValue].shape[0]
    uniqueColumnRowValuesQuantities = {k: v for k, v in sorted(uniqueColumnRowValuesQuantities.items(), key=lambda item: item[1], reverse=True)}
    columnRowsNumber = dataFrame.shape[0]
    while sum(uniqueColumnRowValuesQuantities.values())-list(uniqueColumnRowValuesQuantities.values())[-1] >= coverage * columnRowsNumber:
        uniqueColumnRowValuesQuantities.popitem()

    return uniqueColumnRowValuesQuantities

def __searchForSuperCategories(dataFrame, columnNames):
    superCategories = __getUniqueSuperCategoriesAsSubCategoryArrays(dataFrame, columnNames)
    print("Finding number of examples with super categories...")

    superCategoryByCode = {}
    examplesBySuperCategoryCode = {}
    superCategoryQuantitybyCode = {}

    index = 0
    for superCategory in superCategories:
        examplesWithSuperCategory = __getExamplesWithSuperCategory(dataFrame, columnNames, superCategory)
        #print(examplesWithSuperCategory)
        superCategoryCode = "".join(f"{subCategory}-&-" if np.where(superCategory == subCategory) != superCategory.shape[0]-1 else f"{subCategory}" for subCategory in superCategory)
        superCategoryByCode[superCategoryCode] = superCategory
        examplesBySuperCategoryCode[superCategoryCode] = examplesWithSuperCategory
        superCategoryQuantitybyCode[superCategoryCode] = examplesWithSuperCategory.shape[0]

        if index%1000 == 0:
            print(f"Finding number of examples with super categories in range {index+1}-{index+1000}")
        index += 1

    print("Ordering super categories by quantity in descending order...")
    superCategoryQuantitybyCode = {code: superCategoryAndQuantityList for code, superCategoryAndQuantityList in sorted(superCategoryQuantitybyCode.items(), key=lambda item: item[1], reverse=True)}
    tempSuperCategoryByCode = {}
    tempExamplesBySuperCategoryCode = {}
    for code in superCategoryQuantitybyCode.keys():
        tempSuperCategoryByCode[code] = superCategoryByCode[code]
        #print(examplesBySuperCategoryCode[code])
        #print(list(dataFrame.columns).index("TotalClaimCost"))
        sortedIndexes = examplesBySuperCategoryCode[code][:, list(dataFrame.columns).index("TotalClaimCost")].astype(np.float)[:].argsort()
        sortedIndexes = np.flip(sortedIndexes)
        #print(sortedIndexes)
        tempExamplesBySuperCategoryCode[code] = examplesBySuperCategoryCode[code][sortedIndexes, :]
        #print(tempExamplesBySuperCategoryCode[code])

    superCategoryByCode = tempSuperCategoryByCode
    examplesBySuperCategoryCode = tempExamplesBySuperCategoryCode

    return superCategoryByCode, examplesBySuperCategoryCode, superCategoryQuantitybyCode


def __selectSuperCategoriesWithCoverage(superCategoryByCode, examplesBySuperCategoryCode, superCategoryQuantitiesbyCode, coverage):
    print(f"Selecting super categories covering >={coverage * 100}% examples...")
    initialLength = sum(superCategoryQuantitiesbyCode.values())
    superCategoryByCodeCopy = superCategoryByCode.copy()
    examplesBySuperCategoryCopy = examplesBySuperCategoryCode.copy()
    superCategoryQuantitiesbyCodeCopy = superCategoryQuantitiesbyCode.copy()
    while sum(superCategoryQuantitiesbyCode.values()) - list(superCategoryQuantitiesbyCode.values())[-1] >= coverage * initialLength:
        superCategoryByCodeCopy.popitem()
        examplesBySuperCategoryCopy.popitem()
        superCategoryQuantitiesbyCodeCopy.popitem()
    print(f"Selected {len(superCategoryQuantitiesbyCode)}/{initialLength} ({100 * len(superCategoryQuantitiesbyCodeCopy) / initialLength}%) super categories")
    return superCategoryByCodeCopy, examplesBySuperCategoryCopy, superCategoryQuantitiesbyCodeCopy


def __selectSuperCategoriesWithMinSampleNumber(superCategoryByCode, examplesBySuperCategoryCode, superCategoryQuantitiesbyCode, minSampleNum):
    print(f"Selecting super categories with at least {minSampleNum} examples...")
    codes = list(superCategoryByCode.keys())
    codes.reverse()
    initialLength = len(codes)
    index = 0
    superCategoryByCodeCopy = superCategoryByCode.copy()
    examplesBySuperCategoryCopy = examplesBySuperCategoryCode.copy()
    superCategoryQuantitiesbyCodeCopy = superCategoryQuantitiesbyCode.copy()
    while len(superCategoryQuantitiesbyCode) > 0 and superCategoryQuantitiesbyCode[codes[index]] < minSampleNum:
        superCategoryByCodeCopy.popitem()
        examplesBySuperCategoryCopy.popitem()
        superCategoryQuantitiesbyCodeCopy.popitem()
        index += 1
    print(f"Selected {len(superCategoryQuantitiesbyCode)}/{initialLength} ({100 * len(superCategoryQuantitiesbyCodeCopy) / initialLength}%) super categories")
    return superCategoryByCodeCopy, examplesBySuperCategoryCopy, superCategoryQuantitiesbyCodeCopy


def __getExamplesWithSuperCategory(dataFrame, columnNames, superCategory):
    rowsWithSuperCategoryBoolList = dataFrame[columnNames[0]] == superCategory[0]
    for index in range(1, len(columnNames)):
        rowsWithSuperCategoryBoolList = rowsWithSuperCategoryBoolList & (dataFrame[columnNames[index]] == superCategory[index])
    return dataFrame.loc[rowsWithSuperCategoryBoolList].to_numpy()


def __createSuperCategoryDataFrame(dataFrame, columnsContainingSubCategories, minSampleNumber, finalColumns):
    processedValues = np.empty([dataFrame.shape[0], 10]).astype(np.str)
    classificationCodes = {}
    columnNames = dataFrame.columns
    size = 0

    allSuperCategoriesByCode, allExamplesBySuperCategoryCode, allSuperCategoriesQuantitybyCode = __searchForSuperCategories(dataFrame, columnsContainingSubCategories)
    chosenSuperCategoriesByCode, chosenExamplesBySuperCategoryCode, chosenSuperCategoriesQuantitiesbyCode = __selectSuperCategoriesWithMinSampleNumber(allSuperCategoriesByCode, allExamplesBySuperCategoryCode, allSuperCategoriesQuantitybyCode, minSampleNumber)

    for superCategoryCode, superCategory in chosenSuperCategoriesByCode.items():
        #classificationCode = "".join(f"{subCategory}-&-" if list(superCategory).index(subCategory) != len(superCategory)-1 else f"{subCategory}" for subCategory in superCategory)
        #examplesWithSuperCategory = __getExamplesWithSuperCategory(dataFrame, columnsContainingSubCategories, superCategory)

        additionalSize = chosenSuperCategoriesQuantitiesbyCode[superCategoryCode]
        #print(chosenExamplesBySuperCategoryCode[superCategoryCode])
        #print(chosenExamplesBySuperCategoryCode[superCategoryCode][:, 1])
        processedValues[size:size+additionalSize, :3] = chosenExamplesBySuperCategoryCode[superCategoryCode][:, [list(columnNames).index("id"), list(columnNames).index("PreviousClaimCount"), list(columnNames).index("PreviousClaimCost")]]
        classificationCodes[superCategoryCode] = len(classificationCodes)
        processedValues[size:size+additionalSize, 3] = classificationCodes[superCategoryCode]
        processedValues[size:size + additionalSize, 4:9] = chosenExamplesBySuperCategoryCode[superCategoryCode][:, [list(columnNames).index("CoverCauseDisplayName"),list(columnNames).index("MainDiagnoseCode"),list(columnNames).index("BenefitCode"),list(columnNames).index("IncidentCountryCode"),list(columnNames).index("ClaimantIsSmoker")]]
        processedValues[size:size+additionalSize, 9] = chosenExamplesBySuperCategoryCode[superCategoryCode][:, list(columnNames).index("TotalClaimCost")]
        size += additionalSize

        print(f"Processed data: {size} (+{additionalSize})")

    processedValues = processedValues[:size, :]
    processedDataFrame = pn.DataFrame(data=processedValues, columns=finalColumns)
    __printSuperCategorySummary(dataFrame, processedDataFrame, list(allSuperCategoriesByCode.values()), list(chosenSuperCategoriesByCode.values()), columnsContainingSubCategories)
    return processedDataFrame, classificationCodes


def __printSuperCategorySummary(dataFrame, processedDataFrame, superCategories, chosenSuperCategories, columnsContainingSubCategories):
    dataSetSize = dataFrame.shape[0]
    coveredDataSamples = processedDataFrame.shape[0]
    print(f"Total number of super categories: {len(superCategories)}")
    print(f"Number of super categories after covering {coveredDataSamples}/{dataSetSize} ({100 * coveredDataSamples / dataSetSize}%) of data samples: {len(chosenSuperCategories)}/{len(superCategories)} ({100 * len(chosenSuperCategories) / len(superCategories)}%)")

    possibleSuperCategoriesNumber = 1
    for columnName in columnsContainingSubCategories:
         possibleSuperCategoriesNumber *= len(__findColumnSubCategories(dataFrame, columnName))

    print(f"Total possible number of super-categories: {possibleSuperCategoriesNumber}")
    print(f"Percentage of super-categories used by data set: {100 * len(superCategories)/possibleSuperCategoriesNumber}")
    print(f"Percentage of super-categories covered by super-categories present in {100 * coveredDataSamples / dataSetSize}% of data: {100 * len(chosenSuperCategories)/possibleSuperCategoriesNumber}%")


def __getProcessedMatrixWithSeparateCategoriesColumns(dataFrame, sanitisedColumnNames, chosenColumnsSubCategoriesQuantities):
    preProcessColumnNames = dataFrame.columns.tolist()
    dataToProcess = dataFrame.to_numpy()

    processedValues = np.empty([np.shape(dataToProcess)[0], len(sanitisedColumnNames)])
    index = 0
    for example in dataToProcess:
        for columnName, subCategoriesQuantities in chosenColumnsSubCategoriesQuantities.items():
            if example[preProcessColumnNames.index(columnName)] not in subCategoriesQuantities.keys():
                continue

        rowValues = [
            example[preProcessColumnNames.index("id")],
            example[preProcessColumnNames.index("PreviousClaimCount")],
            example[preProcessColumnNames.index("PreviousClaimCost")],
        ]
        for columnName, subCategoriesQuantities in chosenColumnsSubCategoriesQuantities.items():
            rowValues.extend(["1" if subCategory == example[preProcessColumnNames.index(columnName)] else "0" for subCategory in subCategoriesQuantities.keys()])
        rowValues.append(example[preProcessColumnNames.index("TotalClaimCost")])
        processedValues[index, :] = rowValues

        if index % 1000 == 0:
            print(f"Creating new data in range {index + 1} - {index + 1000} ")
        index += 1

    return processedValues[:index, :]


def __printSubCategorySummary(dataFrame, processedDataFrame, coverage, uniqueColumnsSubCategories, chosenColumnsSubCategoriesQuantities):
    print(f"Total coverage of examples after choosing {coverage * 100}% of most often occurring sub-categories: {100 * processedDataFrame.shape[0] / dataFrame.shape[0]}%")

    columnsNotRequiringSanitisation = 2
    allFeaturesNumberBeforeSanitisation = columnsNotRequiringSanitisation
    for columnName, columnSubCategories in uniqueColumnsSubCategories.items():
         allFeaturesNumberBeforeSanitisation += len(columnSubCategories)

    allFeaturesNumberAfterSanitisation = 0
    for columnName, subCategoriesQuantities in chosenColumnsSubCategoriesQuantities.items():
         allFeaturesNumberBeforeSanitisation += len(subCategoriesQuantities.items())
    print(f"Discarded features: {(allFeaturesNumberBeforeSanitisation - allFeaturesNumberAfterSanitisation) / allFeaturesNumberBeforeSanitisation}% ({allFeaturesNumberBeforeSanitisation - allFeaturesNumberAfterSanitisation})")

    possibleFeaturesCombinationsBeforeSanitization = 1
    for columnName, columnSubCategories in uniqueColumnsSubCategories.items():
         possibleFeaturesCombinationsBeforeSanitization *= len(columnSubCategories)

    possibleFeaturesCombinationsAfterSanitization = 1
    for columnName, subCategoriesQuantities in chosenColumnsSubCategoriesQuantities.items():
         possibleFeaturesCombinationsAfterSanitization *= len(subCategoriesQuantities.items())

    print(f"Total combinations coverage after choosing {coverage * 100}% of most often occurring sub-categories from columns \n { list(chosenColumnsSubCategoriesQuantities.keys())}: {100 * possibleFeaturesCombinationsAfterSanitization / possibleFeaturesCombinationsBeforeSanitization}%")
