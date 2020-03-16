
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class TreeNode:
    lChild = None
    rChild = None
    spVal = None
    spInd = None


def isTree(tree):
    if tree.__class__ == TreeNode:
        return True
    return False


def regLeaf(dataSet):
    return np.mean(dataSet[:, 0])


def regErr(dataSet):
    # total error, not MSE
    return np.var(dataSet[:, 0])*len(dataSet)


def regTreeEval(model, inDat):
    return float(model)


class CART:
    Tree = None
    dataSet = None
    testData = None

    def __init__(self, filename):
        df = pd.read_csv(filename, header=0, sep=",", dtype=float)
        self.dataSet = np.mat(df)
        testInd = np.random.randint(0, len(self.dataSet), 100)
        trainInd = []
        self.testData = self.dataSet[testInd]
        for i in range(len(self.dataSet)):
            if i in testInd:
                continue
            else:
                trainInd.append(i)
        self.dataSet = self.dataSet[trainInd]

    def binSplitDataSet(self, dataSet, feature, value):
        mat0 = dataSet[np.nonzero(dataSet[:, feature] > value)[0], :]
        mat1 = dataSet[np.nonzero(dataSet[:, feature] <= value)[0], :]
        return mat0, mat1

    def createTree(self, dataSet, ops=(5, 25)):
        dataSet = np.mat(dataSet)
        feat, val = self.chooseBestSplit(dataSet, ops)
        if feat == None:
            return val
        retTree = TreeNode()
        retTree.spInd = feat
        retTree.spVal = val
        lSet, rSet = self.binSplitDataSet(dataSet, feat, val)
        retTree.lChild = self.createTree(lSet, ops)
        retTree.rChild = self.createTree(rSet, ops)
        return retTree

    def chooseBestSplit(self, dataSet, ops):
        # minimum step of descent
        tolS = ops[0]
        # minimum num of samples to split
        tolN = ops[1]
        if len(set(dataSet[:, 0].T.tolist()[0])) == 1:
            return None, regLeaf(dataSet)
        _, n = np.shape(dataSet)
        S = regErr(dataSet)
        bestS = np.inf
        bestIndex = 0
        bestValue = 0
        for featIndex in range(1, n):
            for splitVal in set(dataSet[:, featIndex].T.tolist()[0]):
                mat0, mat1 = self.binSplitDataSet(dataSet, featIndex, splitVal)
                if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):
                    continue
                newS = regErr(mat0) + regErr(mat1)
                if newS < bestS:
                    bestIndex = featIndex
                    bestValue = splitVal
                    bestS = newS
        if (S - bestS) < tolS:
            return None, regLeaf(dataSet)
        mat0, mat1 = self.binSplitDataSet(dataSet, bestIndex, bestValue)
        if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):
            return None, regLeaf(dataSet)
        return bestIndex, bestValue

    def createFore(self, tree):
        m = len(self.testData)
        yHat = np.mat(np.zeros((m, 1)))
        for i in range(m):
            yHat[i, 0] = self.treeForecast(tree, self.testData[i])
        return yHat

    def treeForecast(self, tree, inData):
        if not isTree(tree):
            return regTreeEval(tree, inData)
        # inData is a 1*m matrix
        if inData.tolist()[0][tree.spInd] > tree.spVal:
            if isTree(tree.lChild):
                return self.treeForecast(tree.lChild, inData)
            else:
                return regTreeEval(tree.lChild, inData)
        else:
            if isTree(tree.rChild):
                return self.treeForecast(tree.rChild, inData)
            else:
                return regTreeEval(tree.rChild, inData)

    def boostStrap(self):
        randSet = set()
        trainData = []
        for _ in range(int(len(self.dataSet)*1.5)):
            rand = np.random.randint(0, len(self.dataSet))
            randSet.add(rand)
            trainData.append(self.dataSet.tolist()[rand])
        pruneData = []
        for i in range(len(self.dataSet)):
            if i in randSet:
                continue
            pruneData.append(self.dataSet.tolist()[i])
        trainData = np.array(trainData)
        pruneData = np.array(pruneData)
        return trainData, pruneData

    def crossVal(self, i):
        numPerFold = len(self.dataSet)//5
        testData = self.dataSet[i*numPerFold:(i+1)*numPerFold]
        return testData

    def predict(self):
        numOfTrees = 5
        forest = []
        for i in range(numOfTrees):
            trainData, pruneData = self.boostStrap()
            tree = self.createTree(trainData)
            tree = self.prune(tree, pruneData)
            forest.append(tree)
        forestPred = []
        for i in range(numOfTrees):
            predVec = self.createFore(tree)
            forestPred.append(predVec)
        voteResult = np.array(forestPred).mean(axis=0)
        testLabel = self.testData[:, 0]
        return voteResult, testLabel.A

    def prune(self, tree, testData):
        if np.shape(testData)[0] == 0:
            return self.getMean(tree)
        if isTree(tree.lChild) or isTree(tree.rChild):
            lSet, rSet = self.binSplitDataSet(testData,
                                              tree.spInd, tree.spVal)
        if isTree(tree.lChild):
            tree.lChild = self.prune(tree.lChild, lSet)
        if isTree(tree.rChild):
            tree.rChild = self.prune(tree.rChild, rSet)
        if (not isTree(tree.lChild)) and (not isTree(tree.rChild)):
            lSet, rSet = self.binSplitDataSet(testData,
                                              tree.spInd, tree.spVal)
            errorNoMerge = sum(np.power(lSet[:, 0]-tree.lChild, 2)) +\
                sum(np.power(rSet[:, 0]-tree.rChild, 2))
            treeMean = (tree.lChild+tree.rChild)/2
            errorMerge = sum(np.power(testData[:, 0]-treeMean, 2))
            if errorNoMerge > errorMerge:
                return treeMean
            else:
                return tree
        else:
            return tree

    def getMean(self, tree):
        if isTree(tree.rChild):
            tree.rChild = self.getMean(tree.rChild)
        if isTree(tree.lChild):
            tree.lChild = self.getMean(tree.lChild)
        return (tree.lChild+tree.rChild)/2.0

# following are on Titanic
# for i in range(10):
#     c = CART(R'lian_chuang\data\myTitanic.csv')
#     c.predict()


c = CART(R'lian_chuang\data\myForestFire.csv')
predict, testLabel = c.predict()
print(predict)
print(predict.shape)
print(testLabel)
print(testLabel.shape)
x = np.linspace(0, len(predict), len(predict))
plt.plot(x, np.clip(np.exp(testLabel) -
                    np.exp(predict), -200, 200), c='b', alpha=0.7)
plt.show()
