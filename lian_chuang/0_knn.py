
import numpy as np
import operator
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class knn:
    filename = None
    testFeat = None
    testLabels = None
    data = None
    labels = None
    k = 7
    predictLabel = None

    def __init__(self, filename):
        self.filename = filename
        self.data, self.labels = self.normalize()
    
    def predict(self):
        retLabels = []
        for testData in self.testFeat:
            retLabels.append(self.findMax(testData))
        self.predictLabel = retLabels
        self.plot()

    def line2Data(self, line):
        line = line.strip('\n')
        line = line.split(',')
        curLine = []
        for feat in line[:-1]:
            curLine.append(float(feat))
        if str(line[-1]) == 'Iris-setosa':
            curLine.append('r')
        elif str(line[-1]) == 'Iris-versicolor':
            curLine.append('g')
        elif str(line[-1]) == 'Iris-virginica':
            curLine.append('b')
        return curLine
    # 处理数据 done

    def file2matrix(self):
        with open(self.filename) as fr:
            arrayOLines = fr.readlines()[:-1]
            numOflines = len(arrayOLines)
            numOfTestData = int(numOflines*0.1)
            np.random.seed(7)

            indexOfTest = np.random.randint(0, numOflines, numOfTestData)
            testFeat = []
            testLabels = []

            for index in indexOfTest:
                line = self.line2Data(arrayOLines[index])
                testFeat.append(line[:-1])
                testLabels.append(line[-1])
            self.testFeat = np.array(testFeat)
            self.testLabels = np.array(testLabels)

            retMat = []
            retLabels = []
            for index in range(numOflines):
                if index in indexOfTest:
                    continue
                line = self.line2Data(arrayOLines[index])
                retMat.append(line[:-1])
                retLabels.append(line[-1])
            retMat = np.array(retMat)
            retLabels = np.array(retLabels)
            return retMat, retLabels

    def normalize(self):
        mat, labels = self.file2matrix()
        mean = mat.mean(axis=0)
        mat -= mean
        std = mat.std(axis=0)
        mat /= std

        self.testFeat -= mean
        self.testFeat /= std
        return mat, labels

    def findMax(self, testData):
        diffMat = self.data-testData
        sqdiffMat = diffMat ** 2
        sqDistance = sqdiffMat.sum(axis=1)
        distances = sqDistance ** 0.5
        sortedDist = distances.argsort()
        classCount = {}
        for i in range(self.k):
            voteIlable = self.labels[sortedDist[i]]
            classCount[voteIlable] = classCount.get(voteIlable, 0) + 1
        sortedClassCount = sorted(classCount.items(),
                                  key=operator.itemgetter(1), reverse=True)
        return sortedClassCount[0][0]

    def plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        featX = self.data[:, 0].tolist()
        featY = self.data[:, 1].tolist()
        featZ = self.data[:, 2].tolist()
        originLabels = list(self.labels)
        ax.scatter3D(featX, featY, featZ, c=originLabels, alpha=0.2)
        testFeatX = self.testFeat[:, 0].tolist()
        testFeatY = self.testFeat[:, 1].tolist()
        testFeatZ = self.testFeat[:, 2].tolist()
        predictLabel = list(self.predictLabel)
        ax.scatter3D(testFeatX, testFeatY, testFeatZ,
                     c=predictLabel, marker='+', alpha=1)
        plt.show()


filename = 'lian_chuang\\data\\iris.data'
k = knn(filename)
k.predict()