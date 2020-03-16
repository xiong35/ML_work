

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image


def pca(dataMat, topNfeat=2):
    mean = dataMat.mean(axis=0)
    meanRemoved = dataMat - mean
    covMat = np.cov(meanRemoved, rowvar=0)
    eigVals, eigVecs = np.linalg.eig(np.mat(covMat))
    eigValInd = np.argsort(eigVals)
    # sort eigvals
    eigValInd = eigValInd[:-(topNfeat+1):-1]
    redEigVecs = eigVecs[:, eigValInd]
    lowDimData = meanRemoved*redEigVecs
    reconMat = (lowDimData*redEigVecs.T)+mean
    return lowDimData, reconMat


# img = Image.open(R"lian_chuang\images\binggongzhu.jpg")
# img_data = np.array(img)
# print(img_data.shape)
# mymat = img_data.mean(axis=2)
# lowDimData0, reconMat0 = pca(mymat, 100)
# plt.imshow(reconMat0, cmap='gist_gray')
# plt.axis('off')
# plt.show()
# lowDimData1, reconMat1 = pca(mymat, 500)
# plt.imshow(reconMat1, cmap='gist_gray')
# plt.axis('off')
# plt.show()
# lowDimData2, reconMat2 = pca(mymat, 2000)
# plt.imshow(reconMat2, cmap='gist_gray')
# plt.axis('off')
# plt.show()


x, y, z = np.random.multivariate_normal(
    [0, 0, 0], [[1, 5, 4], [0, 2, 5], [1, 1, 1]], 300).T
dataMat = np.mat([x, y, z]).T
lowDimData, reconMat = pca(dataMat)
print(lowDimData)
print(reconMat)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(dataMat[:, 0].flatten().A[0],
              dataMat[:, 1].flatten().A[0],
              dataMat[:, 2].flatten().A[0], c='b', alpha=0.5)
ax.scatter3D(reconMat[:, 0].flatten().A[0],
              reconMat[:, 1].flatten().A[0],
              reconMat[:, 2].flatten().A[0],c='r', alpha=0.5)
plt.show()
