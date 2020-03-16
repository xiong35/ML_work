
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


class GMM:
    def __init__(self, nDim=3, nCluster=7):
        self.nCluster = nCluster

        dataSet = self.createDataSet(int(nCluster*1.5), nDim)
        self.trainData, self.testData = self.randSample(dataSet)
        self.nData, self.nDim = self.trainData.shape

        # init random gaussian distribute
        self.mu = np.random.randint(self.trainData.min()/2, self.trainData.max() /
                                    2, size=(nCluster, nDim))
        self.cov = np.zeros((nCluster, nDim, nDim))
        for dim in range(len(self.cov)):
            np.fill_diagonal(self.cov[dim], 1)

        # prob array
        self.pi = np.ones(nCluster)/nCluster
        # prob mat
        self.pMat = np.zeros((self.nData, nCluster))

    def createDataSet(self, k, nDim=3):
        mean = np.random.randn(nDim)*5
        cov = np.random.randn(nDim, nDim)
        n = np.random.randint(200, 500)
        dataSet = np.random.multivariate_normal(mean, cov, n)
        for _ in range(k-1):
            mean = np.random.randn(nDim)*5
            cov = np.random.randn(nDim, nDim)
            n = np.random.randint(50, 500)
            dataSet = np.concatenate(
                (dataSet, np.random.multivariate_normal(mean, cov, n)), axis=0)
        return dataSet

    def randSample(self, dataSet):
        nData = len(dataSet)
        trainData, testData = [], []
        testInd = np.random.randint(0, nData, int(nData*0.2))
        for i in range(nData):
            if i in testInd:
                testData.append(dataSet[i])
            else:
                trainData.append(dataSet[i])
        trainData = np.mat(trainData)
        testData = np.mat(testData)
        return trainData, testData

    def cal_p_of_x(self, data):
        Px = np.mat(np.zeros((len(data), self.nCluster)))
        CONST = np.power((2*np.pi), (self.nDim/2.0))
        for k in range(self.nCluster):
            delta = np.power((np.linalg.det(self.cov[k, :, :])), 0.5)
            frac_1_2piDOTsigma = 1/(CONST * delta)
            shift = data - self.mu[k, :]
            sigmaInv = np.linalg.inv(self.cov[k, :, :])
            epow = -0.5*(np.multiply(shift.dot(sigmaInv), shift))
            epowsum = np.sum(epow, axis=1)
            Px[:, k] = frac_1_2piDOTsigma * np.exp(epowsum)
        return Px.A

    def train(self):
        max_iter = 30
        for _ in range(max_iter):
            # E
            self.pMat = self.cal_p_of_x(self.trainData)
            # frequence of each dataSample
            totol_N = self.pMat.sum(axis=1)
            # if a dataSample's freq == 0, reset to 1/nCluster
            totol_N[totol_N == 0] = self.nCluster
            self.pMat /= totol_N.reshape(-1, 1)
            # M
            for k in range(self.nCluster):
                N_k = np.sum(self.pMat[:, k], axis=0)
                self.mu[k] = (1/N_k)*np.sum(np.multiply(self.trainData,
                                                        self.pMat[:, k].reshape(-1, 1)), axis=0)
                self.cov[k] = ((1/N_k)*np.dot(np.multiply(self.pMat[:, k].reshape(-1, 1),
                                                          (self.trainData-self.mu[k])).T,
                                              (self.trainData-self.mu[k])))
                self.pi[k] = N_k/self.nData

    def predict(self):
        predMat = self.cal_p_of_x(self.testData)

        totol_N = predMat.sum(axis=1)
        totol_N[totol_N == 0] = self.nCluster
        predMat /= totol_N.reshape(-1, 1)
        yPred = np.argmax(predMat, axis=1)
        # 2D
        # plt.clf()
        # plt.scatter(self.testData[:, 0],
        #             self.testData[:, 1], c=yPred, alpha=0.5)
        # plt.show()
        # 3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter3D(self.testData[:, 0],
                     self.testData[:, 1],
                     self.testData[:, 2], c=yPred, alpha=0.5)
        plt.show()


g = GMM()
g.train()
g.predict()
