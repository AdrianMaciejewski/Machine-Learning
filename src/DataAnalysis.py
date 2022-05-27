import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from src.DataProcessing import *


class DataAnalysis:
    def __init__(self, data, target_feature, numerical_features, categorical_features, labels=[]):
        self.data = data.astype('U30')
        self.target_feature = target_feature
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.labels = labels
        self.clusterLabels = []

    def plot_numerical_features_distribution(self):
        for feature in self.numerical_features:
            sns.displot(data=self.data[feature].to_frame().astype(float), x=feature, kde=True, rug=True, rug_kws={"height":-0.01, "clip_on": False})
            plt.show()

    def plot_categorical_features_distribution(self):
        quantity = self.data.shape[0]

        for feature in self.categorical_features:
            values = self.data[feature]
            unique_values = np.unique(values)


            quantity_by_value = np.empty((unique_values.shape[0], 2), dtype='U30')
            for i in range(0, unique_values.shape[0]):
                value = unique_values[i]
                value_quantity = np.where(values == value)[0].shape[0]
                label = f"{value}- {round(value_quantity/quantity*100, 2)}% (x{value_quantity})"
                quantity_by_value[i] = np.array([label, value_quantity])
            quantity_by_value = quantity_by_value[np.flip(quantity_by_value[:, 1].astype(int).argsort()), :]

            fig, ax = plt.subplots()
            ax.pie(quantity_by_value[:, 1], autopct='%1.1f%%', shadow=True, startangle=90)
            ax.legend(labels=quantity_by_value[:, 0], title="Categories", loc='center right')
            ax.set_title(feature)
            fig.canvas.set_window_title(f"{feature} - pie chart")
            plt.show()

    def correlationMatrix(self):
        corr = self.data[self.numerical_features].astype(float).corr(method="kendall").round(3)
        mask = np.zeros_like(corr)
        mask[np.triu_indices_from(mask)] = True
        cmap = sns.diverging_palette(224, 8, s=100, as_cmap=True)
        ax = sns.heatmap(corr, mask=mask, linewidths=.5, cmap=cmap, annot=True,
                         annot_kws={'fontsize': 20, "rotation": "horizontal"}, fmt="g", vmin=-1, vmax=1,
                         xticklabels=True, yticklabels=True)
        ax.tick_params(axis='both', which='major', labelsize=20)
        plt.title("Numerical features correlations", fontsize=20)
        plt.show()

    def plot_numerical_features_pair_plots(self):
        features_number = len(self.numerical_features)
        depth = 6

        previous_end = 0
        for index in range(0, features_number, depth):
            end = index+depth
            features_data = self.data[self.numerical_features].iloc[:, previous_end:end].astype(float)
            if len(self.labels) > 0:
                features_data["labels"] = self.labels
                sns.pairplot(features_data, hue="labels")
            else:
                sns.pairplot(features_data)

            plt.show()
            previous_end = end

    def plot_2D_PCA_numerical_features(self):
        numerical_data = self.data[self.numerical_features].to_numpy(dtype=float)
        pca2 = PCA(n_components=2, )
        principalComponents = pca2.fit_transform(numerical_data)

        PCA_df = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2'])

        hue = {}
        if len(self.labels) > 0:
            hue["hue"] = np.array(self.labels).astype("U30")

        fig, ax = plt.subplots()
        sns.scatterplot(data=PCA_df, x="principal component 1", y="principal component 2", ax=ax, **hue)
        plt.show()

    def plot_one_hot_feature_dependencies(self, column1, column2, comparable_column, yPlotSize, xPlotSize):
        count = self.data.shape[0]
        comparable = self.data[comparable_column].to_numpy(float)
        column1_encoded = oneHotEncode(self.data[column1].to_frame(), [column1])
        column2_encoded = oneHotEncode(self.data[column2].to_frame(), [column2])
        column1_encoded_array = column1_encoded.to_numpy(dtype=int).astype("U30")
        column2_encoded_array = column2_encoded.to_numpy(dtype=int).astype("U30")
        column1_feature_count = column1_encoded.shape[1]
        column2_feature_count = column2_encoded.shape[1]
        column1_columns = list(column1_encoded.columns)
        column2_columns = list(column2_encoded.columns)
        features_variations_number = column1_feature_count * column2_feature_count

        plot_data_array = np.empty((count, features_variations_number), dtype="U30")
        columns = np.empty((count, 2), dtype="U30")
        for y1 in range(0, column1_feature_count):
            for y2 in range(0, column2_feature_count):
                feature_index = y1 * column2_feature_count + y2
                plot_data_array[:, feature_index] = np.char.add(column1_encoded_array[:, y1], column2_encoded_array[:, y2])
                columns[feature_index] = f"{column1_columns[y1]}_&_{column2_columns[y2]}"

        for i in range(0, features_variations_number, yPlotSize * xPlotSize):
            fig, axs = plt.subplots(yPlotSize, xPlotSize)
            for y in range(0, yPlotSize):
                for x in range(0, xPlotSize):
                    feature_index = i * yPlotSize * xPlotSize + y * xPlotSize + x
                    if feature_index < features_variations_number:
                        ax = axs[y, x]
                        x_data = plot_data_array[:, feature_index]
                        sns.stripplot(x=x_data, y=comparable, ax=ax, order=["00", "01", "10", "11"])
                        ax.set_ylabel(comparable_column)
                        #ax.set_xlabel(columns[feature_index])
                        #ax.xaxis.set_title(featuresForBarChart[featureIndex])
                    else:
                        break
            #handles = axs[0, 0].get_legend_handles_labels()[0]
            #fig.legend(handles, self.clusterLabels, title="Mean cluster value", loc='center right')
            plt.show()

    def plot_diagnoses_with_hue(self, hue, max_length, dodge):
        data = self.data.copy()
        data["TotalClaimCost"] = data["TotalClaimCost"].astype(float)
        data["Age"] = data["Age"].astype(float).astype(int)
        quantity_by_value = calcFeaturesQuantity(data["MainDiagnoseCode"])
        ordered_values = list(quantity_by_value.keys())
        for i in range(0, len(ordered_values), max_length):
            fig, ax = plt.subplots()
            indexes_to_plot = []
            max_i = i + max_length if i + max_length <= len(quantity_by_value.items()) else i + len(quantity_by_value.items()) - i
            for i2 in range(i, max_i):
                value_indexes = np.where(data["MainDiagnoseCode"] == ordered_values[i2])[0]
                indexes_to_plot.extend(list(value_indexes))
            sns.stripplot(data=data.iloc[indexes_to_plot, :], x="MainDiagnoseCode", y="TotalClaimCost", ax=ax, hue=hue, dodge=dodge)
            ax.set_title(f"MainDiagnoseCode with {hue} hue - cost by category ({i}-{max_i})")
            fig.canvas.set_window_title(f"MainDiagnoseCode with {hue} hue - cost by category ({i}-{max_i})")
            plt.show()

    def plot_separate_diagnoses_with_chosen_column_and_age_hue(self, second_column, hue="Age"):
        data = self.data.copy()
        data["TotalClaimCost"] = data["TotalClaimCost"].astype(float)
        data["Age"] = data["Age"].astype(float).astype(int)
        diagnose_quantity_by_value = calcFeaturesQuantity(data["MainDiagnoseCode"])
        diagnose_ordered_values = list(diagnose_quantity_by_value.keys())
        for i in range(0, len(diagnose_ordered_values)):
            fig, ax = plt.subplots()
            indexes_to_plot = []
            value_indexes = list(np.where(data["MainDiagnoseCode"] == diagnose_ordered_values[i])[0])
            indexes_to_plot.extend(value_indexes)
            #fig.legend(handles, self.clusterLabels, title="Mean cluster value", loc='center right')
            if len(indexes_to_plot) > 0:
                sns.stripplot(data=data.iloc[indexes_to_plot, :], x=second_column, y="TotalClaimCost", hue=hue, ax=ax, palette="flare")
                ax.set_xlabel(f"{second_column} of diagnose code {diagnose_ordered_values[i]}")
                plt.show()

    def plot_cost_by_category(self, category, max_length):
        data = self.data.copy()[[category, "TotalClaimCost"]]
        data["TotalClaimCost"] = data["TotalClaimCost"].astype(float)
        quantity_by_value = calcFeaturesQuantity(data[category])
        ordered_values = list(quantity_by_value.keys())
        for i in range(0, len(ordered_values), max_length):
            fig, ax = plt.subplots()
            indexes_to_plot = []
            max_i = i + max_length if i + max_length <= len(quantity_by_value.items()) else i + len(quantity_by_value.items()) - i
            for i2 in range(i, max_i):
                value_indexes = np.where(data[category] == ordered_values[i2])[0]
                indexes_to_plot.extend(list(value_indexes))
                #fig.legend(handles, self.clusterLabels, title="Mean cluster value", loc='center right')
            if len(indexes_to_plot) > 0:
                sns.boxplot(data=data.iloc[indexes_to_plot, :], x=category, y="TotalClaimCost", ax=ax, color=".8", whis=np.inf)
                sns.stripplot(data=data.iloc[indexes_to_plot, :], x=category, y="TotalClaimCost", ax=ax)
                ax.set_title(f"{category} - cost by category ({i}-{max_i})")
                fig.canvas.set_window_title(f"{category} - cost by category ({i}-{max_i})")
                fig.autofmt_xdate(rotation=45)
                plt.show()

    def plot2DPCA(self, features=None):
        if features:
            columns = list(self.data.columns)
            featureIndexes = [columns.index(feature) for feature in features]
            normalisedData = self.kmeans.npArray[:, featureIndexes]
        else:
            normalisedData = self.kmeans.npArray
        pca2 = PCA(n_components=2, )
        principalComponents = pca2.fit_transform(normalisedData)

        principalDf = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2'])
        finalDf = principalDf
        principalDf.head()

        fig, ax = plt. subplots()
        sns.scatterplot(x=finalDf['principal component 1'], y=finalDf['principal component 2'], ax=ax, hue=self.kmeans.clusterisedSamples.astype(str), hue_order=np.array(list(self.kmeans.orderedValueByClusterIndex.keys())).astype(str), palette="magma")
        #sns.scatterplot(x=self.kmeans.clusterCentroids[:, 0], y=self.kmeans.clusterCentroids[:, 1], ax=ax, s=50, ec='black')
        labels = [f"{i + 1}. {np.around(list(self.kmeans.orderedValueByClusterIndex.values())[i], 0)} (x{np.sum(self.kmeans.clusterisedSamples == i)})" for i in range(0, self.kmeans.clusterNumber)]
        handles = ax.get_legend_handles_labels()[0]
        fig.legend(handles, labels, title="PCA 2D data distribution", loc='center right')
        plt.xlabel('pc1')
        plt.ylabel('pc2')
        plt.show()


    def distributionHistogram(self, featuresForBarChart, yPlotSize, xPlotSize):
        """
            Only numerical features should be used to plot this graph.
        """

        data = self.data[featuresForBarChart].astype(float)

        n = data.shape[1]
        figAxsList = []
        for i in range(0, n, yPlotSize * xPlotSize):
            figAxsList.append(plt.subplots(yPlotSize, xPlotSize))
            for y in range(0, yPlotSize):
                for x in range(0, xPlotSize):
                    featureIndex = i + y * xPlotSize + x
                    if featureIndex < n:
                        fig = figAxsList[-1][0]
                        ax = figAxsList[-1][1][y, x]
                        xData = data.iloc[:, featureIndex].to_numpy()

                        hue = {}
                        if self.kmeans is not None:
                            hue["hue"] = self.kmeans.clusterisedSamples.astype(str)
                            hue["hue_order"] = np.array(list(self.kmeans.orderedValueByClusterIndex.keys())).astype(str)
                        sns.histplot(x=xData, ax=ax, palette="magma", multiple="stack", **hue)
                        #ax.xaxis.set_title(featuresForBarChart[featureIndex])
                    else:
                        break
            handles = figAxsList[0][1][0, 0].get_legend_handles_labels()[0]
            figAxsList[-1][0].legend(handles, self.clusterLabels, title="Mean cluster value", loc='center right')
        plt.show()

    def distributionBarchart(self, featuresForBarChart, yPlotSize, xPlotSize):
        data = self.data[featuresForBarChart]
        n = data.shape[1]
        figAxsList = []
        for i in range(0, n, yPlotSize * xPlotSize):
            figAxsList.append(plt.subplots(yPlotSize, xPlotSize))
            for y in range(0, yPlotSize):
                for x in range(0, xPlotSize):
                    featureIndex = i + y * xPlotSize + x
                    if featureIndex < n:
                        ax = figAxsList[-1][1][y, x]
                        sns.histplot(x=data.iloc[:, featureIndex], ax=ax)
                        ax.xaxis.set_ticklabels([])
                    else:
                        break
        plt.show()

    def distributionPieChart(self, featuresForPieChart, coverageTheshold, yPlotSize, xPlotSize):
        categoryQuantitiesByFeature = {}
        for column in featuresForPieChart:
            categories = self.data[column].astype(str)
            uniqueCategories = np.unique(categories)

            quantitiesByCategory = np.empty((uniqueCategories.shape[0], 2), dtype='U30')
            for i in range(0, uniqueCategories.shape[0]):
                category = uniqueCategories[i]
                quantitiesByCategory[i] = np.array([category, np.where(categories == category)[0].shape[0]])
            quantitiesByCategory = quantitiesByCategory[np.flip(quantitiesByCategory[:, 1].astype(int).argsort()), :]
            categoryQuantitiesByFeature[column] = quantitiesByCategory

        figAxsList = []
        for i in range(0, len(featuresForPieChart), yPlotSize * xPlotSize):
            figAxsList.append(plt.subplots(yPlotSize, xPlotSize))
            for y in range(0, yPlotSize):
                for x in range(0, xPlotSize):
                    featureIndex = i + y * xPlotSize + x
                    if featureIndex < len(featuresForPieChart):
                        ax = figAxsList[-1][1]
                        if yPlotSize != 1 or xPlotSize != 1:
                            ax = ax[y, x]
                        column = featuresForPieChart[featureIndex]
                        sizes = categoryQuantitiesByFeature[column][:, 1].astype(int)
                        sizesSum = np.sum(sizes)
                        labels = categoryQuantitiesByFeature[column][:, 0]
                        for z in range(0, len(labels)):
                            labels[z] += f" - {np.around(100 * sizes[z] / sizesSum, 1)}%"
                        explode = []
                        coveredQuantity = 0
                        for z in range(0, len(sizes)):
                            if coveredQuantity / sizesSum < coverageTheshold:
                                coveredQuantity += sizes[z]
                                explode.append(0.1)
                            else:
                                explode.append(0)
                        ax.pie(sizes, explode=explode, autopct='%1.1f%%', shadow=True, startangle=90)
                        ax.axis('equal')
                        ax.legend(labels, loc='center right')
                        ax.set_title(column)
                        handles, originalLabels = figAxsList[0][1][0, 0].get_legend_handles_labels()
                    else:
                        break
        plt.show()

    def plot_pair_plot(self, yPlotSize, xPlotSize, columns, as_type):
        columns = columns.copy()
        data = self.data[columns].astype(as_type)
        n = data.shape[1]
        labels = []
        #if self.kmeans is not None:
        #    labels = [f"{i + 1}. {np.around(list(self.kmeans.orderedValueByClusterIndex.values())[i], 0)} (x{np.sum(self.kmeans.clusterisedSamples == i)})" for i in range(0, self.kmeans.clusterNumber)]
        figAxsList = []
        for y in range(0, n, yPlotSize):
            for x in range(0, n, xPlotSize):
                y_fig_size = yPlotSize if n - y >= yPlotSize else n - y
                x_fig_size = xPlotSize if n - x >= xPlotSize else n - x
                figAxsList.append(plt.subplots(y_fig_size, x_fig_size))
                for iy in range(0, y_fig_size):
                    for ix in range(0, x_fig_size):
                        yFeatureIndex = iy + y
                        xFeatureIndex = ix + x
                        if yFeatureIndex < len(columns) and xFeatureIndex < len(columns):
                            ax = figAxsList[-1][1]
                            if y_fig_size != 1 and x_fig_size != 1:
                                ax = ax[iy, ix]
                            elif y_fig_size != 1:
                                ax = ax[iy]
                            elif x_fig_size != 1:
                                ax = ax[ix]
                            hue = {}
                            #if self.kmeans is not None:
                            #    hue["hue"] = self.kmeans.clusterisedSamples.astype(str)
                            #    hue["hue_order"] = np.array(list(self.kmeans.orderedValueByClusterIndex.keys())).astype(str)
                            sns.scatterplot(y=data[columns[yFeatureIndex]], x=data[columns[xFeatureIndex]], ax=ax, palette="magma", **hue)
                            #if self.kmeans is not None:
                            #    ax.get_legend().remove()
                            if iy != y_fig_size - 1:
                                ax.set(xlabel='')
                                ax.xaxis.set_ticklabels([])
                            if ix != 0:
                                ax.set(ylabel='')
                                ax.yaxis.set_ticklabels([])
                #handles, originalLabels = figAxsList[0][1][0, 0].get_legend_handles_labels()
                #if self.kmeans is None:
                #    labels = originalLabels
                #figAxsList[-1][0].legend(handles, labels, title="Mean cluster value", loc='center right')
        plt.show()

    def featuresByClusterDictibution(self, plotSize):
        n = self.data.shape[1]
        labels = [f"{i + 1}. {np.around(list(self.kmeans.orderedValueByClusterIndex.values())[i], 0)} (x{np.sum(self.kmeans.clusterisedSamples == i)})" for i in range(0, self.kmeans.clusterNumber)]
        for i in range(0, n, plotSize ** 2):
            figBx, axsBx = plt.subplots(plotSize, plotSize)
            figH, axsH = plt.subplots(plotSize, plotSize)
            for y in range(0, plotSize):
                for x in range(0, plotSize):
                    featureIndex = i + y * plotSize + x
                    if featureIndex < n:
                        sns.boxplot(x=self.kmeans.clusterisedSamples, y=self.data.iloc[:, featureIndex], ax=axsBx[y, x], palette="magma")
                        sns.histplot(x=self.data.iloc[:, featureIndex], ax=axsH[y, x], hue=self.kmeans.clusterisedSamples.astype(str), hue_order=np.array(list(self.kmeans.orderedValueByClusterIndex.keys())).astype(str), palette="magma", multiple="stack")
                    else:
                        break
            handles = axsH[0, 0].get_legend_handles_labels()[0]
            figH.legend(handles, labels, title="Mean cluster value", loc='center right')
            handles = axsH[0, 0].get_legend_handles_labels()[0]
            figBx.legend(handles, labels, title="Mean cluster value", loc='center right')
            plt.show()

