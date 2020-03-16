
import numpy as np
import pandas as pd
np.random.seed(777)


class NaiveBayes:
    dataSet = []
    labels = []
    posibilityMatOf0 = []
    posibilityMatOf1 = []
    NumOf1 = 0
    NumOf0 = 0

    def __init__(self, filename):
        df = pd.read_table(filename, header=0, sep=" ", dtype=str)
        df.drop(columns=['PassengerId', 'Name',
                         'Ticket', 'Cabin'], axis=1, inplace=True)
        df.replace(to_replace=np.nan, value='40', regex=True, inplace=True)
        self.labels = df.values[:, 0]
        self.dataSet = df.values[:, 1:]

    def holdOutSplit(self):
        numOfDate = len(self.dataSet)
        numOfTestData = int(numOfDate*0.2)
        testLabel = self.labels[:numOfTestData]
        testData = self.dataSet[:numOfTestData]
        trainLabel = self.labels[numOfTestData:]
        trainData = self.dataSet[numOfTestData:]
        return (trainData, trainLabel), (testData, testLabel)

    def crossVal(self, i):
        numPerFold = len(self.dataSet)//5
        testData = self.dataSet[i*numPerFold:(i+1)*numPerFold]
        trainData = np.concatenate((self.dataSet[:i*numPerFold],
                                    self.dataSet[(i+1)*numPerFold:]))
        testLabel = self.labels[i*numPerFold:(i+1)*numPerFold]
        trainLabel = np.concatenate((self.labels[:i*numPerFold],
                                     self.labels[(i+1)*numPerFold:]))
        return (trainData, trainLabel), (testData, testLabel)

    def boostStrap(self):
        randSet = set()
        trainData = []
        trainLabel = []
        for _ in range(len(self.dataSet)):
            rand = np.random.randint(0, len(self.dataSet))
            randSet.add(rand)
            trainData.append(self.dataSet.tolist()[rand])
            trainLabel.append(self.labels[rand])
        testData = []
        testLabel = []
        for i in range(len(self.dataSet)):
            if i in randSet:
                continue
            testData.append(self.dataSet.tolist()[i])
            testLabel.append(self.labels[i])
        trainLabel = np.array(trainLabel)
        trainData = np.array(trainData)
        testLabel = np.array(testLabel)
        testData = np.array(testData)
        return (trainData, trainLabel), (testData, testLabel)

    def strPosibility(self, oneColOfData, trainLabel):
        strSet = set()
        oneColOfData = oneColOfData.tolist()
        for string in oneColOfData:
            if string not in strSet:
                strSet.add(string)
        strSet = list(strSet)
        p0Num = dict()
        p1Num = dict()
        for string in strSet:
            p0Num[string] = 1
            p1Num[string] = 1
        for index, string in enumerate(oneColOfData):
            if trainLabel[index] == '1':
                p1Num[string] += 1
            else:
                p0Num[string] += 1
        for string in strSet:
            p0Num[string] = np.log(p0Num[string]/(self.NumOf0+1))
            p1Num[string] = np.log(p1Num[string]/(self.NumOf1+1))
        self.posibilityMatOf0.append(p0Num)
        self.posibilityMatOf1.append(p1Num)

    def numPosibility(self, oneColOfData, trainLabel):
        oneColOfData = oneColOfData.astype('float32')
        minNum = oneColOfData.min()
        maxNum = oneColOfData.max()
        step = (maxNum - minNum)/10
        p0Num = []
        p1Num = []
        for i in range(10):
            p0Num.append([int(minNum+i*step), 1])
            p1Num.append([int(minNum+i*step), 1])
        p0Num.append([float('inf'), 1])
        p1Num.append([float('inf'), 1])
        for index, num in enumerate(oneColOfData):
            if trainLabel[index] == '1':
                for i in range(11):
                    if num < p1Num[i][0]:
                        p1Num[i][1] += 1
                        break
            else:
                for i in range(11):
                    if num < p0Num[i][0]:
                        p0Num[i][1] += 1
                        break
        for i in range(len(p0Num)):
            p0Num[i][1] = np.log(p0Num[i][1]/(self.NumOf0+1))
        for i in range(len(p1Num)):
            p1Num[i][1] = np.log(p1Num[i][1]/(self.NumOf1+1))
        self.posibilityMatOf0.append(p0Num)
        self.posibilityMatOf1.append(p1Num)

    def numPred(self, data, index):
        data = float(data)
        pos0 = 0
        pos1 = 0
        for i in range(len(self.posibilityMatOf0[index])):
            if data < self.posibilityMatOf0[index][i][0]:
                pos0 = self.posibilityMatOf0[index][i][1]
                break
        for i in range(len(self.posibilityMatOf0[index])):
            if data < self.posibilityMatOf1[index][i][0]:
                pos1 = self.posibilityMatOf1[index][i][1]
                break
        return pos0, pos1

    def strPred(self, data, index):
        try:
            pos0 = self.posibilityMatOf0[index][data]
            pos1 = self.posibilityMatOf1[index][data]
        except KeyError:
            pos0, pos1 = -0.5, -0.5
        return pos0, pos1

    def train(self):
        # code for training your naive bayes algorithm
        (trainData, trainLabel), (testData, testLabel) = self.boostStrap()
        for i in range(len(trainLabel)):
            if trainLabel[i] == '1':
                self.NumOf1 += 1
            else:
                self.NumOf0 += 1
        for i in range(len(trainData[0])):
            if i in [2, 5]:
                self.numPosibility(trainData[:, i], trainLabel)
            else:
                self.strPosibility(trainData[:, i], trainLabel)
        return testData, testLabel

    def predict(self):
        # code for predicting
        testData, testLabel = self.train()
        predLabel = []
        predStrength = []
        for data in testData:
            pos0 = self.NumOf0/(self.NumOf0+self.NumOf1)
            pos1 = self.NumOf1/(self.NumOf0+self.NumOf1)
            for featIndex in range(len(data)):
                if featIndex in [2, 5]:
                    cur0, cur1 = self.numPred(data[featIndex], featIndex)
                else:
                    cur0, cur1 = self.strPred(data[featIndex], featIndex)
                pos1 += cur1
                pos0 += cur0
            if pos0 > pos1:
                predLabel.append('0')
                predStrength.append(pos0)
            else:
                predLabel.append('1')
                predStrength.append(pos1)
        tp, tn, fp, fn = 0, 0, 0, 0
        for i in range(len(predLabel)):
            if predLabel[i] == testLabel[i] == '1':
                tp += 1
            elif predLabel[i] == testLabel[i] == '0':
                tn += 1
            elif predLabel[i] == '0' and testLabel[i] == '1':
                fn += 1
            elif predLabel[i] == '1' and testLabel[i] == '0':
                fp += 1
        acc = (tp + tn)/(tp+fn+tn+fp)
        precision = tp/(tp+fp)
        recall = tp/(tp+fn)
        F1 = 2/(1/precision+1/recall)

        print('acc: ', acc)
        print('precision: ', precision)
        print('recall: ', recall)
        print('F1: ', F1)

        return predStrength, testLabel

    def plotROC(self, predStrengths, classLabels):
        import matplotlib.pyplot as plt
        predStrengths = np.array(predStrengths)
        cur = [1.0, 1.0]
        ySum = 0.0
        numPosClass = sum(np.array(classLabels) == '1')
        yStep = 1/float(numPosClass)
        xStep = 1/float(len(classLabels)-numPosClass)
        sortedIndices = predStrengths.argsort()
        fig = plt.figure()
        fig.clf()
        ax = plt.subplot(111)
        for index in sortedIndices[::-1]:
            if classLabels[index] == '1':
                delX = 0
                delY = yStep
            else:
                delX = xStep
                delY = 0
            if delX != 0:
                ySum += cur[1]*xStep
            ax.plot([cur[0], cur[0]-delX], [cur[1], cur[1]-delY], c='b')
            cur = [cur[0]-delX, cur[1]-delY]
        print('ROC: ', ySum)
        ax.plot([0, 1], [0, 1], 'b--')
        plt.xlabel('False Positive')
        plt.ylabel('True Positive')
        plt.show()


n = NaiveBayes(R'lian_chuang\data\titanic.txt')
predStrength, testLabel = n.predict()
n.plotROC(predStrength, testLabel)
