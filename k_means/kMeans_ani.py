
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import cnames

# set K
k = 7

# set the random seed
np.random.seed(666)


def createOneDataSet(n=200, mean=[0, 0], cov=[[1, 0], [0, 1]], tag=1):
    x, y = np.random.multivariate_normal(mean, cov, n).T
    dataSet = []
    for i in range(n):
        curData = []
        curData.append(x[i])
        curData.append(y[i])
        dataSet.append(curData)
    return dataSet


# createDataSet
def createDataSet(k):
    dataSet = []
    for i in range(k):
        mean = [np.random.uniform(-10, 10), np.random.uniform(-10, 10)]
        cov = np.random.randn(2, 2)
        n = np.random.randint(50, 500)
        dataSet.extend(createOneDataSet(n, mean, cov, i))
    return dataSet


# calculate the distance
def distEclud(vacA, vacB):
    return np.sqrt(sum(np.power(vacA-vacB, 2).tolist()[0]))


# set centroids
# robust when data is not 2 dimensional
# return a k*n matrix
# n for num of dimension
def randCent(dataSet, k):
    featNum = dataSet.shape[1]
    centroid = np.mat(np.zeros((k, featNum)))
    for feat_i in range(featNum):
        min_i = np.min(dataSet[:, feat_i])
        range_i = float(np.max(dataSet[:, feat_i])-min_i)
        centroid[:, feat_i] = min_i + range_i * np.random.rand(k, 1)
    return centroid


# kMeans
def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    numOfData = dataSet.shape[0]
    # cluster assignment:
    # [0] rf which cluster this data is assigned to
    # [1] rf the distance between data and cluster center(for further calculation)
    clusterAssment = np.mat(np.zeros((numOfData, 2)))
    centroid = createCent(dataSet, k)
    clusterChanged = True
    step = 0
    iterTime = 10
    while clusterChanged and (step < iterTime):
        # show
        x = dataSet[:, 0].T.tolist()[0]
        y = dataSet[:, 1].T.tolist()[0]
        plt.hist2d(x, y, bins=300)
        centX = centroid[:, 0].T.tolist()[0]
        centY = centroid[:, 1].T.tolist()[0]
        plt.scatter(centX, centY,c='#ffff00',marker='+',s=20)
        figDict = 'fig' + str(step)
        plt.savefig(figDict)
        plt.show()
        clusterChanged = False
        for indexOfData in range(numOfData):
            minDist = np.inf
            minIndex = -1
            for j in range(k):
                # the j_th centroid's coordinates
                # the i_th data's coordinates
                distJI = distMeas(centroid[j, :], dataSet[indexOfData, :])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[indexOfData, 0] != minIndex:
                clusterChanged = True
            clusterAssment[indexOfData, :] = minIndex, minDist**2
        step += 1
        for center in range(k):
            ptsInCluster = dataSet[np.nonzero(
                clusterAssment[:, 0].A == center)[0]]
            if len(ptsInCluster) != 0:
                centroid[center, :] = np.mean(ptsInCluster, axis=0)
    return centroid, clusterAssment


# createDataSet and turn it into a matrix
dataSet = createDataSet(k)
dataSet = np.mat(dataSet)

# show the data
x = dataSet[:, 0].T.tolist()[0]
y = dataSet[:, 1].T.tolist()[0]
plt.hist2d(x, y, bins=300)

# scatter the centroid
centroid, _ = kMeans(dataSet, k)


