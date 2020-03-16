

# calculate the entropy of the data

import matplotlib.pyplot as plt
import operator
from math import log


def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVector in dataSet:
        currentLabel = featVector[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob*log(prob, 2)
    return shannonEnt


# creat dataSet

def createDataSet():
    dataSet = [[1, 1, 0,'yes'],
               [1, 1, 0,'yes'],
               [1, 0, 0,'no'],
               [0, 1, 0,'no'],
               [0, 1, 1,'no'],
               [0, 1, 1,'yes'],
               [1, 0, 0,'no'],
               [1, 0, 1,'yes'],
               [0, 0, 0,'no']]
    labels = ['no surfacing', 'flippers','true']
    return dataSet, labels
    
""" 
def createDataSet():
    data = pd.read_csv('/home/ylxiong/Documents/mushrooms.csv', nrows=100)
    dataSet = data.values.tolist()

    labels = ['cap-shape', 'cap-surface', 'cap-color', 'bruises',
              'odor', 'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
              'stalk-shape', 'talk-root', 'stalk-surface-above-ring',
              'stalk-surface-below-ring', 'stalk-color-above-ring',
              'stalk-color-below-ring', 'veil-type', 'veil-color',
              'ring-number', 'ring-type', 'spore-print-color',
              'population	habitat']
    return dataSet, labels
 """


# split the data
# if the value in featVec's given axis equals to the given value
# the new data will exclude the given axis
# else just skip the data

def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


# calculate the shannonEnt
# and find the best way to split the data

def chooseBestFeatureToSplit(dataSet):
    # in this case, the dataSet's last value is the y value
    # no need to be taken into consideration
    numFeatures = len(dataSet[0])-1
    # set original value
    bestEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        # set the feature in i axis
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            # calculate posterior prob
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob*calcShannonEnt(subDataSet)
        infoGain = bestEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


# return the name of the most frequent class

def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(),
                              key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


# recursively split the data,
# according to the decreasement of the entropy
# return when all features are split
# or the subDataSet are all the same

def createTree(dataSet, labels):
    # in this case, the dataSet's last value is the y value
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)

    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet
                                                  (dataSet, bestFeat, value), subLabels)
    return myTree


# createTree

myDat, labels = createDataSet()
myTree = createTree(myDat, labels)
print(myTree)


###############################
#####--- show the tree ---#####
###############################


decisionNode = dict(boxstyle='sawtooth', fc='0.8')
leafNode = dict(boxstyle='round4', fc='0.8')
arrow_args = dict(arrowstyle='<-')


def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    plt.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
                            xytext=centerPt, textcoords='axes fraction',
                            va='center', ha='center', bbox=nodeType, arrowprops=arrow_args)


def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs


def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth


def plotMidText(centerPt, parentPt, txtString):
    xMid = (parentPt[0] - centerPt[0]) / 2.0 + centerPt[0]
    yMid = (parentPt[1] - centerPt[1]) / 2.0 + centerPt[1]
    plt.text(xMid, yMid, txtString)


def plotTree(myTree, parentPt, nodeTxt):
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    firstStr = list(myTree.keys())[0]
    centerPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW,
                plotTree.yOff)
    plotMidText(centerPt, parentPt, nodeTxt)
    plotNode(firstStr, centerPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key], centerPt, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff,
                                       plotTree.yOff), centerPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), centerPt,str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD

def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    plt.axis('off')
    plt.subplots(111, frameon=False)
    plt.subplots()
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5/plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(inTree,(0.5,1.0),'')
    plt.show()

createPlot(myTree) 


########################
#####---classify---#####
########################

def calssify(inputTree,featLabels,testVec):
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ =='dict':
                classLabel = classify(secondDict[key],featLabels,testVec)
            else:
                classLabel = secondDict[key]
    return classLabel

##########################
#####---save model---#####
##########################

def storeTree(inputTree,filename):
    import pickle
    fw = open(filename,'wb')
    pickle.dump(inputTree,fw)
    fw.close()

def loadTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)

storeTree(myTree,'tree.bin')
print(loadTree('tree.bin'))