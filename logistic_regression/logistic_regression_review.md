# Python + Numpy实现逻辑回归

## 导入Numpy
 
    import numpy as np

## 导入数据

这里选用和鲸社区上下载的电离层二分类数据  

下载地址：[电离层数据](https://www.kesci.com/home/dataset/5dc28bd6080dc30037200775)  

其实这个数据不太好用。。将就一下吧。。。  

    def loadDataSet(fileName):
        detaMatrix = []
        labelMatrix = []
        with open(fileName) as fr:
            for line in fr.readlines()[:250]:
                lineArray = line.strip().split(',')
                detaMatrix.append([1.0, float(lineArray[9]),
                                float(lineArray[22])])
                if lineArray[-1] == 'b':
                    labelMatrix.append(1)
                else:
                    labelMatrix.append(0)
        return detaMatrix, labelMatrix

    dataMatrix, labelMatrix = loadDataSet(
        '/home/ylxiong/Documents/ionosphere.data')

## 定义sigmoid函数

    def sigmoid(inX):
        return 1.0/(1.0+np.exp(-1.0*inX))

## 实现梯度下降（1.0版本）

    def gradientAscent(dataMatrix, classMatrix):
        dataMatrix = np.mat(dataMatrix)
        labelMatrix = np.mat(classMatrix).transpose()
        m, n = np.shape(dataMatrix)
        alpha = 0.001
        maxCycles = 500
        weights = np.ones((n, 1))
        for k in range(maxCycles):
            # dataMatrix: 250 * 3
            # weight    : 3 * 1
            h = sigmoid(np.dot(dataMatrix, weights))
            error = (labelMatrix - h)
            weights = weights + alpha*np.dot(dataMatrix.transpose(), error)
        return weights

    weights = gradientAscent(dataMatrix, labelMatrix)

## 随机梯度下降（2.0）

用随机梯度下降可以大幅提高学习速度，用以下代码替换上一小节（或者忽略这一节  

    def stocGradAscent(dataMatrix, labelMatrix, numIter=150):
        m, n = np.shape(dataMatrix)
        weights = np.ones(n)
        for j in range(numIter):
            dataIndex = list(range(m))
            for i in range(m):
                alpha = 16/(1.0+j+i)+0.04
                rangeIndex = int(np.random.uniform(0,len(dataIndex)))
                h = sigmoid(sum(dataMatrix[i]*weights))
                error = labelMatrix[rangeIndex] - h
                weights = weights + alpha*np.dot(error, dataMatrix[i])
                del(dataIndex[rangeIndex])
        return weights

    weights = stocGradAscent(dataMatrix, labelMatrix)
    # [weights] is a list instead of a matrix

看一看weights长啥样

    print(weights)

## 用matplotlib画出图像

    def plotBestFit(weights):
        import matplotlib.pyplot as plt
        # weights = weights.getA()
        dataMatrix, labelMatrix = \
            loadDataSet('/home/ylxiong/Documents/ionosphere.data')
        n = len(dataMatrix)
        xcord1 = []
        ycord1 = []
        xcord2 = []
        ycord2 = []
        for i in range(n):
            if int(labelMatrix[i]) == 1:
                xcord1.append(dataMatrix[i][1])
                ycord1.append(dataMatrix[i][2])
            else:
                xcord2.append(dataMatrix[i][1])
                ycord2.append(dataMatrix[i][2])
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
        ax.scatter(xcord2, ycord2, s=30, c='blue')
        x = np.arange(-3.0, 3.0, 0.1)
        y = (-weights[0]-weights[1]*x)/weights[2]
        ax.plot(x, y)
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.show()

    plotBestFit(weights)

## 画出来的结果长这样

![figure](http://q5ioolwed.bkt.clouddn.com/logistic_regression_mpl.png)  

很不幸的选到了一个没啥用的feature（红色那堆），完全没有区分度。。。  

总之就是这样了😂