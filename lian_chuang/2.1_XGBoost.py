
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


def regLeaf(dataSet, originData):
    mean = np.mean(dataSet[:, 0])
    changeList = dataSet[:, -1]
    for changeInd in changeList.A:
        originData[int(changeInd), 0] -= mean
    return mean


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
        df['counter'] = range(len(df))
        self.dataSet = np.mat(df)
        self.testData = self.dataSet[-100:]
        self.dataSet = self.dataSet[:-100]

    def binSplitDataSet(self, dataSet, feature, value):
        mat0 = dataSet[np.nonzero(dataSet[:, feature] > value)[0], :]
        mat1 = dataSet[np.nonzero(dataSet[:, feature] <= value)[0], :]
        return mat0, mat1

    def createTree(self, dataSet, originData, ops=(0.05, 10)):
        dataSet = np.mat(dataSet)
        feat, val = self.chooseBestSplit(dataSet, originData, ops)
        if feat == None:
            return val
        retTree = TreeNode()
        retTree.spInd = feat
        retTree.spVal = val
        lSet, rSet = self.binSplitDataSet(dataSet, feat, val)
        retTree.lChild = self.createTree(lSet, originData, ops)
        retTree.rChild = self.createTree(rSet, originData, ops)
        return retTree

    def chooseBestSplit(self, dataSet, originData, ops=(1.5, 5)):
        # minimum step of descent
        tolS = ops[0]
        # minimum num of samples to split
        tolN = ops[1]
        if len(set(dataSet[:, 0].T.tolist()[0])) == 1:
            return None, regLeaf(dataSet, originData)
        _, n = np.shape(dataSet)
        S = regErr(dataSet)
        bestS = np.inf
        bestIndex = 0
        bestValue = 0
        for featIndex in range(1, n-1):
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
            return None, regLeaf(dataSet, originData)
        mat0, mat1 = self.binSplitDataSet(dataSet, bestIndex, bestValue)
        if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):
            return None, regLeaf(dataSet, originData)
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

    def predict(self):
        numOfTrees = 5
        forest = []
        for i in range(numOfTrees):
            tree = self.createTree(self.dataSet, self.dataSet)
            forest.append(tree)
        forestPred = np.zeros((len(self.testData),1))
        for i in range(numOfTrees):
            predVec = self.createFore(forest[i])
            forestPred += predVec.A
            plt.plot(forestPred,label='{}'.format(i+1))
        acc = 0
        plt.legend()
        plt.show()
        for i in range(len(forestPred)):
            if forestPred[i] > 0.5:
                forestPred[i] = 1
            else:
                forestPred[i] = 0
            if forestPred[i] == self.testData[i, 0]:
                acc += 1
        acc /= len(self.testData)
        print('acc: ', acc)

# 这里用的是做过特征工程的数据
c = CART(R'lian_chuang\data\myTitanic.csv')
c.predict()
