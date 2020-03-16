
# Python + Numpy实现adaboost

## 导入Numpy

    import numpy as np

## 定义导入数据函数（1.0）

自己生成少量数据:  

    def loadSimpData():
        dataMatrix = np.matrix([[1., 2.1],
                                [2., 1.1],
                                [1.3, 1.],
                                [1., 1.],
                                [2., 1.]])
        classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
        return dataMatrix, classLabels

    dataMatrix, classLabels = loadSimpData()

## 定义导入数据函数(2.0)

用这段程序替换上一节  

从和鲸社区上下载数据  

下载地址：[电离层数据](https://www.kesci.com/home/dataset/5dc28bd6080dc30037200775)  

    filename = '/home/ylxiong/Documents/ionosphere.data'

    def loadDataSet(filename, begin=0, end=200):
        dataMatrix = []
        labelMatrix = []
        with open(filename) as fr:
            numFeat = len(fr.readline().split(','))
            for line in fr.readlines()[begin:end]:
                lineArr = []
                curLine = line.strip().split(',')
                for i in range(numFeat - 1):
                    lineArr.append(float(curLine[i]))
                dataMatrix.append(lineArr)
                # change the label into 1/-1
                # so the following calculation will be easier
                if curLine[-1] == 'g':
                    labelMatrix.append(1.0)
                else:
                    labelMatrix.append(-1.0)
        return dataMatrix, labelMatrix

## 定义划分单元

给定划分维度、阈值、方式（大于阈值/小于阈值？）  
将数据集划分成+1/-1两边  
将在后续主函数中调用这个函数  

    def stumpClassify(dataMatrix, dimension, threshVal, threshIneq):
        retArray = np.ones((np.shape(dataMatrix)[0], 1))
        # less than / greater than
        if threshIneq == 'lt':
            # dataMatrix[:, dimension] <= threshVal
            # means the certain index of [datas in
            # dataMatrix's x dimension that is smaller than threshVal]
            retArray[dataMatrix[:, dimension] <= threshVal] = -1.0
        else:
            retArray[dataMatrix[:, dimension] > threshVal] = -1.0
        return retArray

## 定义单个划分结点

对于给定数据集，找出最适合的划分方式  

何为[最适合]？  
算法之一：  
运用所测试的方法划分，看分错了多少个，对每一笔错误乘上一个weight，选取总和最小的划分方式  

怎么找？  
遍历所有划分可能，计算上述值  

    ##### - build single decision tree - #####
    # D for weights
    # three loops
    # first: for all the features
    # second: for all the values
    # third: for > / <=

    def buildStump(dataArray, classLabels, D):
        dataMatrix = np.mat(dataArray)
        labelMatrix = np.mat(classLabels).T
        m, n = np.shape(dataMatrix)
        numSteps = 10.0
        bestStump = {}
        bestClasEst = np.mat(np.zeros((m, 1)))
        minError = np.inf
        for i in range(n):
            rangeMin = dataMatrix[:, i].min()
            rangeMax = dataMatrix[:, i].max()
            stepSize = (rangeMax - rangeMin)/numSteps
            for j in range(-1, int(numSteps)+1):
                for inequal in ['lt', 'gt']:
                    threshVal = (rangeMin + float(j)*stepSize)
                    predictedVals = stumpClassify(
                        dataMatrix, i, threshVal, inequal)
                    errArr = np.mat(np.ones((m, 1)))
                    errArr[predictedVals == labelMatrix] = 0
                    # D is the weights of each error
                    # errArr rf weather the model make a mistake in certain index
                    weightedError = np.dot(D.T, errArr)
                    print('split: dim % d, thresh: % .2f, inequal: % s,\
                        weightedError: %.3f' % (i, threshVal, inequal, weightedError))
                    if weightedError < minError:
                        minError = weightedError
                        bestClasEst = predictedVals.copy()
                        bestStump['dim'] = i
                        bestStump['thresh'] = threshVal
                        bestStump['ineq'] = inequal
        return bestStump, minError, bestClasEst

## 定义主函数

递归地使用上述函数进行划分  
详见注释  

    ##### - combine nodes together - #####
    # DS: decision stump, a weak classifier
    # not the only one, but the most popular one

    def adaBoostTrainDS(dataArray, classLabels, numIter=40):
        weakClassArr = []
        m = np.shape(dataArray)[0]
        D = np.mat(np.ones((m, 1))/m)
        aggClassEst = np.mat(np.zeros((m, 1)))
        # 设置最大迭代次数
        for i in range(numIter):
            # 找出当前最好的划分方案
            bestStump, error, classEst = buildStump(
                dataArray, classLabels, D)
            print('D: ', D.T)
            # alpha可看作当前划分对结果的置信度，在后面加权要用
            alpha = float(0.5*np.log((1.0-error)/max(error, 1e-16)))# prvent 0 deviation
            bestStump['alpha'] = alpha
            weakClassArr.append(bestStump)
            print('classEst:', classEst.T)
            # 预测和结果相同得1分，不同得-1分，乘上-alpha取指数
            expon = np.multiply(-1*alpha*np.mat(classLabels).T, classEst)
            # 将上述值乘上权重D
            D = np.multiply(D, np.exp(expon))
            # 更新权重
            D = D/D.sum()
            # 更新预测
            aggClassEst += alpha*classEst
            print('aggClassEst:', aggClassEst.T)
            aggErrors = np.multiply(np.sign(aggClassEst) !=
                                    np.mat(classLabels).T, np.ones((m, 1)))
            # 计算错误率
            errorRate = aggErrors.sum()/m
            print('errorRate:', errorRate)
            # 为0就不用继续了
            if errorRate == 0:
                break
        return weakClassArr, aggClassEst

至此模型构建完毕  

## 定义分类函数

根据预测结果，最后得到大于0的结果就分为正类  

    ### - classify - ###

    def adaClassify(datToClassify, classifierArr):
        dataMatrix = np.mat(datToClassify)
        m = np.shape(dataMatrix)[0]
        aggClassEst = np.mat(np.zeros((m, 1)))
        for i in range(len(classifierArr)):
            classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'],
                                    classifierArr[i]['thresh'],
                                    classifierArr[i]['ineq'])
            aggClassEst += classifierArr[i]['alpha']*classEst
            print(aggClassEst)
        return np.sign(aggClassEst)

## 定义绘制ROC曲线

    def plotROC(predStrengths, classLabels):
        import matplotlib.pyplot as plt
        cur = [1.0, 1.0]
        ySum = 0.0
        # pos for positive
        numPosClass = sum(np.array(classLabels) == 1.0)
        yStep = 1/float(numPosClass)
        xStep = 1/float(len(classLabels)-numPosClass)
        sortedIndices = predStrengths.argsort()
        fig = plt.figure()
        fig.clf()
        ax = plt.subplot(111)
        for index in sortedIndices.tolist()[0]:
            if classLabels[index] == 1.0:
                delX = 0
                delY = yStep
            else:
                delX = xStep
                delY = 0
            ySum += cur[1]
            ax.plot([cur[0], cur[0]-delX], [cur[1], cur[1]-delY], c='b')
            cur = [cur[0]-delX, cur[1]-delY]
        ax.plot([0, 1], [0, 1], 'b--')
        plt.xlabel('False Positive')
        plt.ylabel('True Positive')
        plt.show()

## 正式开始

定义相关参数

    numIter = 40
    begin = 230
    end = 250

导入数据

    dataMatrix, labelMatrix = loadDataSet(filename)
    testMatrix, testLabelMatrix = loadDataSet(filename, begin, end)

训练模型

    classifierArr, aggClassEst = adaBoostTrainDS(dataMatrix, labelMatrix, numIter)

预测结果

    prediction = adaClassify(testMatrix, classifierArr)
    errArr = np.mat(np.ones((end-begin, 1)))
    rate = errArr[prediction != np.mat(testLabelMatrix).T].sum()/(end - begin)
    print('testing error: ',rate*100,'%')

绘制ROC

    plotROC(aggClassEst.T, labelMatrix)

结果如下：

![ROC plot](http://q5ioolwed.bkt.clouddn.com/ROC_mpl.png)

就是这样了😉
