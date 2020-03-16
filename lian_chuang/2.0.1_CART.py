
import operator
from math import log
import numpy as np
import pandas as pd


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
        df = pd.read_table(filename, header=0, sep=" ", dtype=str)
        df.drop(columns=['PassengerId', 'Name',
                         'Ticket', 'Cabin'], axis=1, inplace=True)
        df.replace(to_replace=np.nan, value='40', regex=True, inplace=True)
        df.replace(to_replace='C', value='0', regex=True, inplace=True)
        df.replace(to_replace='S', value='1', regex=True, inplace=True)
        df.replace(to_replace='Q', value='-1', regex=True, inplace=True)
        df.replace(to_replace='female', value='0', regex=True, inplace=True)
        df.replace(to_replace='male', value='1', regex=True, inplace=True)
        self.dataSet = np.mat(df).astype('float32')
        self.testData = self.dataSet[:150]
        self.dataSet = self.dataSet[150:]

    def binSplitDataSet(self, dataSet, feature, value):
        mat0 = dataSet[np.nonzero(dataSet[:, feature] > value)[0], :]
        mat1 = dataSet[np.nonzero(dataSet[:, feature] <= value)[0], :]
        return mat0, mat1

    def createTree(self, dataSet,   ops=(1, 4)):
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

    def chooseBestSplit(self, dataSet,   ops=(1, 4)):
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

    def predict(self):
        tree = self.createTree(self.dataSet)
        tree = self.prune(tree, self.testData)
        predVec = self.createFore(tree)
        pred = []
        acc = 0
        for i in range(len(self.testData)):
            if predVec[i][0] > 0.5:
                pred.append(1)
            else:
                pred.append(0)
            if pred[i] == self.testData[i, 0]:
                acc += 1
        acc /= len(self.testData)
        print('acc: ', acc)

    def prune(self, tree, testData):
        if np.shape(testData)[0] == 0:
            return getMean(tree)
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
                print("merge")
                return treeMean
            else:
                return tree
        else:
            return tree


def getMean(tree):
    if isTree(tree.rChild):
        tree.rChild = getMean(tree.rChild)
    if isTree(tree.lChild):
        tree.lChild = getMean(tree.lChild)
    return (tree.lChild+tree.rChild)/2.0


##############################################
############ classification tree #############
##############################################

def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVector in dataSet:
        currentLabel = featVector[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob*log(prob, 2)
    return shannonEnt


def chooseBestFeatureToSplit(dataSet):
    # in this case, the dataSet's last value is the y value
    # no need to be taken into consideration
    numFeatures = len(dataSet[0])-1
    # set original value
    bestEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        # set the feature in i axis
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            # method binSplitDataSet is in class CART
            subDataSet = binSplitDataSet(dataSet, i, value)
            # calculate posterior prob
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob*calcShannonEnt(subDataSet)
        infoGain = bestEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(),
                              key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


c = CART(R'lian_chuang\data\titanic.txt')
c.predict()
