
# Python+Numpy构建分类回归树

## 引入相关依赖

    import numpy as np
    import random
    from math import sin,log

## 定义划分数据函数

根据所给的feature编号和对应阈值划分数据集

    def binSplitDataSet(dataSet, feature, value):
        mat0 = dataSet[np.nonzero(dataSet[:, feature] > value)[0], :]
        mat1 = dataSet[np.nonzero(dataSet[:, feature] <= value)[0], :]
        return mat0, mat1

## 定义普通模型(1.0)

定义普通叶结点，返回当前被分配到的数据的平均值

    def regLeaf(dataSet):
        # 返回平行X轴的直线
        return np.mean(dataSet[:, -1])

定义普通误差，返回MSE

    def regErr(dataSet):
        # total error, not MSE
        return np.var(dataSet[:, -1])*np.shape(dataSet)[0]

## 定义线性模型(2.0)

准备工作：

    def linearSolve(dataSet):
        # m是date数，n是feature数
        m, n = np.shape(dataSet)
        X = np.mat(np.ones((m, n)))
        Y = np.mat(np.ones((m, 1)))
        # 把x设置成第一列全为1(i.e. x_0)
        # 后面为正常x_1, x_2之类的值
        X[:, 1:n] = dataSet[:, 0:n-1]
        # Y即是labels
        Y = dataSet[:, -1]
        xTx = np.dot(X.T, X)
        if np.linalg.det(xTx) == 0:
            raise NameError('This matrix is singular, cannot do inverse, \
                try to increase the second value of ops')
        # 用玄学的线代知识直接解出最优直线
        ws = np.dot(xTx.I, np.dot(X.T, Y))
        # 返回这条直线的参数，加了1的feature，label
        return ws, X, Y

定义线性模型叶结点

    def modelLeaf(dataSet):
        ws, X, Y = linearSolve(dataSet)
        # 返回直线的参数
        return ws

    def modelErr(dataSet):
        ws, X, Y = linearSolve(dataSet)
        yHat = np.dot(X, ws)
        # 返回预测值和实际值的MSE
        return np.var(yHat-Y)

## 定义选取最佳切分方案的函数

详见[这篇文章](http://101.133.217.104/python-numpy%E5%AE%9E%E7%8E%B0adaboost)的[定义单个划分结点]小节  

不同之处在于在这个模型里我们要进行剪枝操作，如果一个划分方案达不到理想的效果，那干脆就不划分了  
此外此例还可以在调用函数时选择模型（普通模型/线性模型）  

    def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
        # minimum step of descent
        tolS = ops[0]
        # minimum num of samples to split
        tolN = ops[1]
        # 如果剩下的结点都是同一类，就返回模型给出的预测方案
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
        # 如果变化太小
        if (S - bestS) < tolS:
            return None, leafType(dataSet)
        mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
        # 或者切出来的数据太少，视为已经是同一类了
        if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):
            return None, leafType(dataSet)
        # 如果的确分的很好就返回划分的数据集
        return bestIndex, bestValue

## 定义训练树

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

## 定义树的后剪枝

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

## 产生数据

定义产生数据的函数

    def y(x):
        y = random.uniform(-0.05, 0.05) + sin(x/10.0)+log(1+x/10.0)
        return y

产生一些训练数据

    myData = []
    for i in range(100):
        listArr = []
        listArr.append(i)
        listArr.append(y(i))
        myData.append(listArr)
    myData = np.mat(myData)

再产生一些测试数据  

    testData = []
    for i in range(0, 100, 3):
        listArr = []
        listArr.append(i)
        listArr.append(y(i))
        testData.append(listArr)
    testData = np.mat(testData)

## 定义两个模型的预测函数

    def regTreeEval(model, inDat):
        return float(model)

    def modelTreeEval(model, inDat):
        n = np.shape(inDat)[1]
        X = np.mat(np.ones((1, n+1)))
        X[:, 1:n+1] = inDat
        return float(np.dot(X[:, :-1], model))

## 定义利用模型进行预测的函数

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

## 正式开始

建立模型

    myTree = createTree(myData, modelLeaf, modelErr, ops=(0.1, 2))
    # print(myTree)

预测

    model = modelTreeEval  # 或者换成regTreeEval
    # remenber to change the leaf/err too
    yHat = createFore(myTree, testData, model)
    # print(yHat)
