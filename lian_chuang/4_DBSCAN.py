
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import sin

np.random.seed(77)


class DBSCAN:

    dataMat = None
    nDim = 4
    clusters = 4
    tolN = None
    tolDist = None
    numOfPoints = None
    labels = None

    def __init__(self, tolDist=4.5, tolN=7):
        self.dataMat = self.genRandData()
        for _ in range(self.clusters-1):
            mat = self.genRandData()
            self.dataMat = np.concatenate((self.dataMat, mat), axis=1,)
        self.tolDist = tolDist
        self.tolN = tolN
        self.numOfPoints = self.dataMat.shape[1]
        self.labels = np.zeros((self.numOfPoints))

    def genRandData(self):
        a = np.random.uniform(-4, 4)
        line = np.linspace(a-1, a+1, 50)
        thisCluster = []
        for _ in range(self.nDim):
            fn = self.genRandFunc()
            yi = []
            for x in line:
                yi.append(fn(x)+np.random.uniform(-3, 3))
            thisCluster.append(yi)
        return np.mat(thisCluster)

    def genRandFunc(self):
        # ParameterList
        pL = []
        for _ in range(4):
            pL.append(np.random.uniform(-4, 4))

        def fn(i): return pL[0]*pow(i*1.3, 3)/100 + \
            pL[1]*pow(i*1.4, 2)/10+pL[2]*i+pL[3]
        return fn

    def calcDist(self, vec1, vec2):
        return np.sqrt(sum((vec1-vec2).A**2))

    def findCenter(self):
        centerPt = []
        distList = np.zeros((self.numOfPoints, self.numOfPoints))
        for curPoint in range(self.numOfPoints):
            curCount = 0
            curVec = self.dataMat[:, curPoint]
            for another in range(self.numOfPoints):
                if distList[curPoint, another] != 0 or distList[another, curPoint] != 0:
                    curDist = distList[curPoint, another]
                else:
                    anotherVec = self.dataMat[:, another]
                    curDist = self.calcDist(curVec, anotherVec)
                    distList[curPoint, another] = curDist
                    distList[another, curPoint] = curDist
                if self.tolDist*0.1 < curDist < self.tolDist:
                    curCount += 1
            if curCount > self.tolN:
                centerPt.append(curPoint)
        return centerPt, distList

    def cluster(self):
        centerPt, distList = self.findCenter()
        classLabel = 1
        while centerPt:
            centerIndex = centerPt.pop()
            curCluster = self.findSmallCluster(distList[:, centerIndex])
            i = 0
            while True:
                try:
                    curCluster[i]
                except IndexError:
                    break
                if curCluster[i] in centerPt:
                    newCluster = self.findSmallCluster(
                        distList[:, curCluster[i]])
                    centerPt.remove(curCluster[i])
                    for pt in newCluster:
                        if pt not in curCluster:
                            curCluster.append(pt)
                i += 1
            for index in curCluster:
                if self.labels[index] == 0:
                    self.labels[index] = classLabel
            classLabel += 1
        print(self.labels)

    def findSmallCluster(self, distMat):
        distVec = distMat
        smallCluster = []
        for i in range(self.numOfPoints):
            if self.tolDist*0.1 < distVec[i] < self.tolDist:
                smallCluster.append(i)
        return smallCluster

    def plot3D(self):
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter3D(self.dataMat[0], self.dataMat[1],
                     self.dataMat[2], c=self.labels)
        plt.show()

    def plot2D(self):
        print(self.labels)
        print(self.dataMat[0].tolist()[0])
        plt.scatter(self.dataMat[1].tolist()[0],
                    self.dataMat[2].tolist()[0], c=self.labels)
        plt.show()


d = DBSCAN()
d.cluster()
d.plot3D()
