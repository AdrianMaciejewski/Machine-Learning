import numpy as np
import matplotlib.pyplot as plt


class KMeans:
    def __init__(self, npArray, clusterNumber, iterationNumber):
        self.y = npArray[:, -1]
        self.npArray = npArray
        self.clusterNumber = clusterNumber
        self.iterationNumber = iterationNumber
        self.clusterCentroids = self.npArray[np.random.randint(self.npArray.shape[0], size=self.clusterNumber), :]
        self.clusterisedSamples = np.zeros(self.npArray.shape[0], dtype=int)
        self.orderedValueByClusterIndex = {}

    def bestKMeans(self, iterationsNumber):
        bestClusterisedSamples = self.clusterisedSamples
        bestClusterCentroids = self.clusterCentroids
        bestCost = float('inf')
        for i in range(0, iterationsNumber):
            print(f"KMeans nr.{i+1}")
            clusterisedSamples, clusterCentroids = self.__runKMeans(self.clusterNumber)
            cost = self.__calculateClustersCost(clusterisedSamples, clusterCentroids)
            if cost < bestCost:
                print(f"Better cluster cost found: {cost}")
                bestClusterisedSamples = clusterisedSamples
                bestClusterCentroids = clusterCentroids
                bestCost = cost
        self.clusterisedSamples = bestClusterisedSamples
        self.clusterCentroids = bestClusterCentroids
        return self.clusterisedSamples

    def runKMeans(self):
        self.clusterisedSamples, self.clusterCentroids = self.__runKMeans(self.clusterNumber)
        return self.clusterisedSamples

    def __runKMeans(self, clusterNumber):
        clusterCentroids = self.npArray[np.random.randint(self.npArray.shape[0], size=clusterNumber), :]
        clusterisedSamples = np.empty(self.npArray.shape[0], dtype=int)

        for i in range(0, self.iterationNumber):
            print(f"K-means iteration: {i}")
            for s in range(0, self.npArray.shape[0]):
                minDistance = float('inf')
                cluserIndex = 0
                for c in range(0, clusterCentroids.shape[0]):
                    distance = np.sqrt(np.sum(np.power(clusterCentroids[c] - self.npArray[s], 2)))
                    if distance < minDistance:
                        minDistance = distance
                        cluserIndex = c
                clusterisedSamples[s] = cluserIndex

            for c in range(0, clusterCentroids.shape[0]):
                newClusterCentroid = np.mean(self.npArray[np.where(clusterisedSamples == c)[0]], axis=0)
                clusterCentroids[c, :] = newClusterCentroid
        return clusterisedSamples, clusterCentroids

    def orderClusterInxedingByMeanValue(self):
        valueByClusterIndex = {}
        for i in range(0, self.clusterNumber):
            valueByClusterIndex[i] = np.mean(self.y[np.where(self.clusterisedSamples == i)[0]])
        valueByClusterIndex = {k: v for k, v in sorted(valueByClusterIndex.items(), key=lambda item: item[1], reverse=True)}
        clustersOrder = list(p for p in valueByClusterIndex.keys())
        mapping = {clustersOrder[i]: i for i in range(0, len(clustersOrder))}
        self.clusterisedSamples = np.array([mapping[x] for x in self.clusterisedSamples])
        self.orderedValueByClusterIndex = {mapping[k]: v for k, v in valueByClusterIndex.items()}
        return self.clusterisedSamples

    def calculateClustersCost(self):
        return self.__calculateClustersCost(self.clusterisedSamples, self.clusterCentroids)

    def __calculateClustersCost(self, clusterisedSamples, clusterCentroids):
        cost = 0
        for i in range(0, self.npArray.shape[0]):
            cost += np.sqrt(np.sum(np.power(clusterCentroids[clusterisedSamples[i]] - self.npArray[i], 2)))
        return cost

    def plotElbowCurve(self, maxClusterNumber):
        print("Plotting elbow curve")
        fig, ax = plt.subplots()

        clustersNumber = np.empty(maxClusterNumber)
        clustersCost = np.empty(maxClusterNumber)
        for i in range(1, maxClusterNumber+1):
            print(f"\tCalculating {i}/{maxClusterNumber}")
            clustersCost[i-1] = self.__calculateClustersCost(*self.__runKMeans(i))
            clustersNumber[i-1] = i

        ax.plot(clustersNumber, clustersCost)
        ax.set_xlabel('Clusters number')
        ax.set_ylabel('Clusters cost')
        ax.set_title('Elbow curve')
        fig.canvas.manager.set_window_title('K-Means elbow curve')
        plt.show()