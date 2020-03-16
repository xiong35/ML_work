
import numpy as np
import random
from math import sin,log


def binSplitDataSet(dataSet, feature, value):
    mat0 = dataSet[np.nonzero(dataSet[:, feature] > value)[0], :]
    mat1 = dataSet[np.nonzero(dataSet[:, feature] <= value)[0], :]
    return mat0, mat1


def regLeaf(dataSet):
    return np.mean(dataSet[:, -1])


def regErr(dataSet):
    # total error, not MSE
    return np.var(dataSet[:, -1])*np.shape(dataSet)[0]


def modelLeaf(dataSet):
    ws, X, Y = linearSolve(dataSet)
    return ws


def modelErr(dataSet):
    ws, X, Y = linearSolve(dataSet)
    yHat = np.dot(X, ws)
    return np.var(yHat-Y)


def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    dataSet = np.mat(dataSet)
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    if feat == None:
        return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree


def linearSolve(dataSet):
    m, n = np.shape(dataSet)
    X = np.mat(np.ones((m, n)))
    Y = np.mat(np.ones((m, 1)))
    X[:, 1:n] = dataSet[:, 0:n-1]
    Y = dataSet[:, -1]
    xTx = np.dot(X.T, X)
    if np.linalg.det(xTx) == 0:
        raise NameError('This matrix is singular, cannot do inverse, \
            try to increase the second value of ops')
    ws = np.dot(xTx.I, np.dot(X.T, Y))
    return ws, X, Y


# detailed illustration in decision_tree.py


def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    # minimum step of descent
    tolS = ops[0]
    # minimum num of samples to split
    tolN = ops[1]
    if len(set(dataSet[:, -1].T.tolist()[0])) == 1:
        return None, leafType(dataSet)
    m, n = np.shape(dataSet)
    S = errType(dataSet)
    bestS = np.inf
    bestIndex = 0
    bestValue = 0
    for featIndex in range(n-1):
        for splitVal in set(dataSet[:, featIndex].T.tolist()[0]):
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):
                continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    if (S - bestS) < tolS:
        return None, leafType(dataSet)
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):
        return None, leafType(dataSet)
    return bestIndex, bestValue


### prune the tree ###

def isTree(obj):
    return (type(obj).__name__ == 'dict')


def getMean(tree):
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    return (tree['left']+tree['right'])/2.0


def prune(tree, testData):
    if np.shape(testData)[0] == 0:
        return getMean(tree)
    if isTree(tree['left']) or isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData,
                                     tree['spInd'], tree['spVal'])
    if isTree(tree['left']):
        tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']):
        tree['right'] = prune(tree['right'], rSet)
    if (not isTree(tree['left'])) and (not isTree(tree['right'])):
        lSet, rSet = binSplitDataSet(testData,
                                     tree['spInd'], tree['spVal'])
        errorNoMerge = sum(np.power(lSet[:, -1]-tree['left'], 2)) +\
            sum(np.power(rSet[:, -1]-tree['right'], 2))
        treeMean = (tree['left']+tree['right'])/2
        errorMerge = sum(np.power(testData[:, -1]-treeMean, 2))
        if errorNoMerge > errorMerge:
            print("merge")
            return treeMean
        else:
            return tree
    else:
        return tree


# train model
# random.seed(7)

def y(x):
    y = random.uniform(-0.05, 0.05) + sin(x/10.0)+log(1+x/10.0)
    return y


myData = []
for i in range(100):
    listArr = []
    listArr.append(i)
    listArr.append(y(i))
    myData.append(listArr)
myData = np.mat(myData)
# print(myData)

testData = []
for i in range(0, 100, 3):
    listArr = []
    listArr.append(i)
    listArr.append(y(i))
    testData.append(listArr)
testData = np.mat(testData)


def regTreeEval(model, inDat):
    return float(model)


def modelTreeEval(model, inDat):
    n = np.shape(inDat)[1]
    X = np.mat(np.ones((1, n+1)))
    X[:, 1:n+1] = inDat
    return float(np.dot(X[:, :-1], model))


def treeForecast(tree, inData, modelEval):
    if not isTree(tree):
        return modelEval(tree, inData)
    # inData is a 1*m matrix
    if inData.tolist()[0][tree['spInd']] > tree['spVal']:
        if isTree(tree['left']):
            return treeForecast(tree['left'], inData, modelEval)
        else:
            return modelEval(tree['left'], inData)
    else:
        if isTree(tree['right']):
            return treeForecast(tree['right'], inData, modelEval)
        else:
            return modelEval(tree['right'], inData)


def createFore(tree, testData, modelEval):
    m = len(testData)
    yHat = np.mat(np.zeros((m, 1)))
    for i in range(m):
        yHat[i, 0] = treeForecast(tree, testData[i], modelEval)
    return yHat


myTree = createTree(myData, modelLeaf, modelErr, ops=(0.1, 2))
# print(myTree)

model = modelTreeEval  # /regTreeEval
# remenber to change the leaf/err too

yHat = createFore(myTree, testData, model)
# print(yHat)


# prune(myTree, testData)
