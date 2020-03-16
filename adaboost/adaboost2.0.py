
import numpy as np

filename = '/home/ylxiong/Documents/ionosphere.data'


def loadSimpData():
    dataMatrix = np.matrix([[1., 2.1],
                            [2., 1.1],
                            [1.3, 1.],
                            [1., 1.],
                            [2., 1.]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return dataMatrix, classLabels

# dataMatrix, classLabels = loadSimpData()


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


##### - build single decision tree - #####

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


##### - combine nodes together - #####
# DS: decision stump, a weak classifier
# not the only one, but the most popular one

def adaBoostTrainDS(dataArray, classLabels, numIter=40):
    weakClassArr = []
    m = np.shape(dataArray)[0]
    D = np.mat(np.ones((m, 1))/m)
    aggClassEst = np.mat(np.zeros((m, 1)))
    for _ in range(numIter):
        bestStump, error, classEst = buildStump(
            dataArray, classLabels, D)
        print('D: ', D.T)
        # prvent 0 deviation
        alpha = float(0.5*np.log((1.0-error)/max(error, 1e-16)))
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        print('classEst:', classEst.T)
        expon = np.multiply(-1*alpha*np.mat(classLabels).T, classEst)
        D = np.multiply(D, np.exp(expon))
        D = D/D.sum()
        aggClassEst += alpha*classEst
        print('aggClassEst:', aggClassEst.T)
        aggErrors = np.multiply(np.sign(aggClassEst) !=
                                np.mat(classLabels).T, np.ones((m, 1)))
        errorRate = aggErrors.sum()/m
        print('errorRate:', errorRate)
        if errorRate == 0:
            break
    return weakClassArr, aggClassEst


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


numIter = 40
begin = 230
end = 250

dataMatrix, labelMatrix = loadDataSet(filename)
testMatrix, testLabelMatrix = loadDataSet(filename, begin, end)

classifierArr, aggClassEst = adaBoostTrainDS(dataMatrix, labelMatrix, numIter)
prediction = adaClassify(testMatrix, classifierArr)

errArr = np.mat(np.ones((end-begin, 1)))
rate = errArr[prediction != np.mat(testLabelMatrix).T].sum()/(end - begin)

print('testing error: ',rate*100,'%')

plotROC(aggClassEst.T, labelMatrix)
