
import numpy as np
import operator
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
np.random.seed(7777)


class KdNode:
    def __init__(self, storePt, splitInd, lChild, rChild, label):
        self.storePt = storePt
        self.splitInd = splitInd
        self.lChild = lChild
        self.rChild = rChild
        self.label = label


class KdTree:
    def __init__(self, data):
        k = len(data[0])-1

        def CreateNode(split, dataSet):
            if len(dataSet) == 0:
                return None
            dataSet.sort(key=lambda x: x[split])
            splitInd = len(dataSet) // 2
            midData = dataSet[splitInd]
            splitNext = (split + 1) % k

            return KdNode(midData, split,
                          CreateNode(splitNext, dataSet[:splitInd]),
                          CreateNode(splitNext, dataSet[splitInd + 1:]), midData[-1])

        self.root = CreateNode(0, data)


class Result:
    def __init__(self, nearestPoint, nearestDist):
        self.nearestPoint = nearestPoint
        self.nearestDist = nearestDist


class knn:
    filename = None
    testMat = None
    data = None
    k = 7
    kdTree = None

    def __init__(self, filename):
        self.filename = filename
        self.data = self.normalize()

    def predict(self, testData=None):
        predictLabel = []
        if not testData:
            testData = self.testMat
        for data in testData:
            predictLabel.append(self.findNearest(
                self.kdTree, data).nearestPoint[-1])
        self.plot(testData, predictLabel)

    def line2Data(self, line):
        line = line.strip('\n')
        line = line.split(',')
        curLine = []
        for feat in line[:-1]:
            curLine.append(float(feat))
        if str(line[-1]) == 'Iris-setosa':
            curLine.append(1)
        elif str(line[-1]) == 'Iris-versicolor':
            curLine.append(2)
        elif str(line[-1]) == 'Iris-virginica':
            curLine.append(3)
        return curLine
    # 处理数据 done

    def file2matrix(self):
        with open(self.filename) as fr:
            arrayOLines = fr.readlines()[:-1]
            numOflines = len(arrayOLines)
            numOfTestData = int(numOflines*0.15)
            # 随机抽测试数据
            indexOfTest = np.random.randint(0, numOflines, numOfTestData)
            testMat = []
            # 将测试数据整合成矩阵
            for index in indexOfTest:
                line = self.line2Data(arrayOLines[index])
                testMat.append(line)
            self.testMat = testMat
            retMat = []
            for index in range(numOflines):
                if index in indexOfTest:
                    continue
                line = self.line2Data(arrayOLines[index])
                retMat.append(line)
            retMat = np.array(retMat)
            return retMat

    def normalize(self):
        mat = self.file2matrix()
        mean = mat[:, :-1].mean(axis=0)
        mean = np.concatenate((mean, np.zeros(1)))
        mat -= mean
        std = mat[:, :-1].std(axis=0)
        std = np.concatenate((std, np.ones(1)))
        mat /= std

        self.testMat -= mean
        self.testMat /= std
        return mat.tolist()

    def train(self):
        self.kdTree = KdTree(self.data)

    def findNearest(self, tree, point):
        k = len(point)

        def travel(kdNode, target, maxDist):
            if kdNode is None:
                result = Result([0] * k, float("inf"))
                return result

            s = kdNode.splitInd
            curPt = kdNode.storePt
            if target[s] <= curPt[s]:
                nearerNode = kdNode.lChild
                furtherNode = kdNode.rChild
            else:
                nearerNode = kdNode.rChild
                furtherNode = kdNode.lChild

            temp1 = travel(nearerNode, target, maxDist)

            nearest = temp1.nearestPoint
            dist = temp1.nearestDist

            if dist < maxDist:
                maxDist = dist

            tempDist = abs(curPt[s] - target[s])
            if maxDist < tempDist:
                result = Result(nearest, dist)
                return result

            tempDist = np.sqrt(
                sum((p1 - p2) ** 2 for p1, p2 in zip(curPt, target)))

            if tempDist < dist:
                nearest = curPt
                dist = tempDist
                maxDist = dist

            temp2 = travel(furtherNode, target, maxDist)

            if temp2.nearestDist < dist:
                nearest = temp2.nearestPoint
                dist = temp2.nearestDist
            result = Result(nearest, dist)
            return result

        return travel(tree.root, point, float("inf"))

    def plot(self, testData, predictLabel):
        plotData = np.array(self.data)
        testData = np.array(testData)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        featX = plotData[:, 0].tolist()
        featY = plotData[:, 1].tolist()
        featZ = plotData[:, 2].tolist()
        originLabels = plotData[:, 3].tolist()
        ax.scatter3D(featX, featY, featZ, c=originLabels,
                     alpha=0.2, cmap='cool')
        testMatX = testData[:, 0].tolist()
        testMatY = testData[:, 1].tolist()
        testMatZ = testData[:, 2].tolist()
        ax.scatter3D(testMatX, testMatY, testMatZ,
                     c=predictLabel, marker='+', alpha=1, cmap='cool')
        plt.show()


k = knn('lian_chuang\\data\\iris.data')
k.train()
k.predict()
