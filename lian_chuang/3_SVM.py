
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(7)


class opStruct:
    def __init__(self, dataMatIn, classLabels, C, toler):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = len(dataMatIn)
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        self.eCache = np.mat(np.zeros((self.m, 2)))


class SVM:
    trainLabel = None
    trainData = None
    testLabel = None
    testData = None
    b = None
    w = None

    def __init__(self):
        filename = R'lian_chuang\data\myTitanic.csv'
        df = pd.read_csv(filename, header=0, sep=",", dtype=float)
        dataSet = np.mat(df).astype('float32')
        classLabels = dataSet[:, 0]
        for i in range(len(classLabels)):
            if classLabels[i] == 0:
                classLabels[i] = -1
        print(df)
        self.boostStrap(dataSet[:, 1:].tolist(), classLabels.tolist())

    def boostStrap(self, dataSet, classLabels):
        randSet = set()
        trainData = []
        trainLabel = []
        for _ in range(len(dataSet)):
            rand = np.random.randint(0, len(dataSet))
            randSet.add(rand)
            trainData.append(dataSet[rand])
            trainLabel.append(classLabels[rand])
        testData = []
        testLabel = []
        for i in range(len(dataSet)):
            if i in randSet:
                continue
            testData.append(dataSet[i])
            testLabel.append(classLabels[i])
        self.trainLabel = np.mat(trainLabel)
        self.trainData = np.mat(trainData)
        self.testLabel = np.mat(testLabel)
        self.testData = np.mat(testData)

    def calEk(self, oS, k):
        fXk = float(np.multiply(oS.alphas, oS.labelMat).T
                    * (oS.X*oS.X[k, :].T))+oS.b
        Ek = fXk - float(oS.labelMat[k])
        return Ek

    def selectJ(self, i, oS, Ei):
        maxK, maxDeltaE, Ej = -1, 0, 0
        oS.eCache[i] = [1, Ei]
        validEcacheList = np.nonzero(oS.eCache[:, 0].A)[0]
        if len(validEcacheList) > 1:
            for k in validEcacheList:
                if k == i:
                    continue
                Ek = self.calEk(oS, k)
                deltaE = abs(Ei-Ek)
                if deltaE > maxDeltaE:
                    maxK, maxDeltaE, Ej = k, deltaE, Ek
            return maxK, Ej
        else:
            j = i
            while j == i:
                j = np.random.randint(0, oS.m)
            Ej = self.calEk(oS, j)
        return j, Ej

    def updateEk(self, oS, k):
        Ek = self.calEk(oS, k)
        oS.eCache[k] = [1, Ek]

    def clipAlpha(self, aj, H, L):
        if aj > H:
            aj = H
        if L > aj:
            aj = L
        return aj

    def innerL(self, i, oS):
        Ei = self.calEk(oS, i)
        if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C))or\
                ((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] > 0)):
            j, Ej = self.selectJ(i, oS, Ei)
            alphaIold = oS.alphas[i].copy()
            alphaJold = oS.alphas[j].copy()
            if (oS.labelMat[i] != oS.labelMat[j]):
                L = max(0, oS.alphas[j]-oS.alphas[i])
                H = min(oS.C, oS.C+oS.alphas[j]-oS.alphas[i])
            else:
                L = max(0, oS.alphas[j]+oS.alphas[i]-oS.C)
                H = min(oS.C, oS.alphas[j]+oS.alphas[i])
            if L == H:
                return 0
            eta = 2*oS.X[i, :]*oS.X[j, :].T - oS.X[i, :] * \
                oS.X[i, :].T-oS.X[j, :]*oS.X[j, :].T
            if eta >= 0:
                return 0
            oS.alphas[j] -= oS.labelMat[j]*(Ei-Ej)/eta
            oS.alphas[j] = self.clipAlpha(oS.alphas[j], H, L)
            self.updateEk(oS, j)
            if abs(oS.alphas[j]-alphaJold) < 1e-5:
                return 0
            oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i] *\
                (alphaJold-oS.alphas[j])
            self.updateEk(oS, i)
            b1 = oS.b - Ei - \
                oS.labelMat[i]*(oS.alphas[i]-alphaIold) * oS.X[i, :]*oS.X[i, :].T -\
                oS.labelMat[j]*(oS.alphas[j]-alphaJold) * \
                oS.X[i, :]*oS.X[j, :].T
            b2 = oS.b - Ej - \
                oS.labelMat[i]*(oS.alphas[i]-alphaIold) * oS.X[i, :]*oS.X[j, :].T -\
                oS.labelMat[j]*(oS.alphas[j]-alphaJold) * \
                oS.X[j, :]*oS.X[j, :].T
            if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
                oS.b = b1
            elif (0 < oS.alphas[j])and (oS.C > oS.alphas[j]):
                oS.b = b2
            else:
                oS.b = (b1+b2)/2
            return 1
        else:
            return 0

    def smoP(self, dataMatIn, classLabels, C, toler, maxIter):
        oS = opStruct(dataMatIn, classLabels, C, toler)
        iter = 0
        entireSet = True
        alphaPairsChanged = 0
        while (iter < maxIter) and ((alphaPairsChanged > 0)or entireSet):
            alphaPairsChanged = 0
            if entireSet:
                for i in range(oS.m):
                    alphaPairsChanged += self.innerL(i, oS)
                iter += 1
            else:
                nonBoundIs = np.nonzero((oS.alphas.A > 0)*(oS.alphas.A < C))[0]
                for i in nonBoundIs:
                    alphaPairsChanged += self.innerL(i, oS)
                iter += 1
            if entireSet:
                entireSet = False
            elif alphaPairsChanged == 0:
                entireSet = True
        return oS.b, oS.alphas

    def calcWs(self, alphas, data, classLabels):
        X = data
        labelMat = classLabels
        m, n = np.shape(X)
        w = np.zeros((n, 1))
        for i in range(m):
            w += np.multiply(alphas[i]*labelMat[i], X[i, :].T)
        return w

    def test(self):
        acc = 0
        ws = np.mat(self.w)
        for i in range(100):
            if self.testData[i]*ws+self.b > 0:
                pred = 1
            else:
                pred = -1
            if pred == self.testLabel[i]:
                acc += 1
        print(acc)

    def train(self):
        b, alphas = self.smoP(
            self.trainData, self.trainLabel, 0.005, 0.001, 40)
        w = self.calcWs(alphas, self.trainData, self.trainLabel)
        self.b, self.w = b, w


s = SVM()
s.train()
s.test()
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# featX = data[:, 0].T.tolist()[0]
# featY = data[:, 1].T.tolist()[0]
# featZ = data[:, 2].T.tolist()[0]
# labels = classLabels.T.tolist()[0]
# ax.scatter3D(featX, featY, featZ, c=labels, alpha=0.2, cmap='viridis')
# x = np.linspace(-5, 5, 50)
# y = np.linspace(-5, 5, 50)
# X, Y = np.meshgrid(x, y)
# z = (-b - w[0]*x-w[1]*y)/w[2]
# ax.plot_surface(X, Y, z, rstride=1, cstride=1, cmap='rainbow')
# plt.show()
