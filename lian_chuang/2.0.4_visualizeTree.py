



decisionNode = dict(boxstyle='round', fc='0.8')
leafNode = dict(boxstyle='round4', fc='0.8')
arrow_args = dict(arrowstyle='<-')


def getNumLeafs(myTree):
    numLeafs = 0
    if not isTree(myTree):
        return 1
    numLeafs += getNumLeafs(myTree.lChild)+getNumLeafs(myTree.rChild)
    return numLeafs


def getTreeDepth(myTree):
    if not isTree(myTree):
        return 1
    return max(getTreeDepth(myTree.lChild), getTreeDepth(myTree.rChild))+1


def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    plt.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
                 xytext=centerPt, textcoords='axes fraction',
                 va='center', ha='center', size=10, bbox=nodeType, arrowprops=arrow_args)


def plotMidText(centerPt, parentPt, txtString):
    xMid = (parentPt[0] - centerPt[0]) / 2.0 + centerPt[0]
    yMid = (parentPt[1] - centerPt[1]) / 2.0 + centerPt[1]
    plt.text(xMid, yMid, txtString, size=10,)


def plotTree(myTree, parentPt, nodeTxt):
    numLeafs = getNumLeafs(myTree)
    # depth = getTreeDepth(myTree)
    firstStr = 'id: ' + str(myTree.spInd)
    centerPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW,
                plotTree.yOff)
    plotMidText(centerPt, parentPt, nodeTxt)
    plotNode(firstStr, centerPt, parentPt, decisionNode)
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD

    if isTree(myTree.lChild):
        plotTree(myTree.lChild, centerPt, '<%.2f' % myTree.spVal)
    else:
        plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
        plotNode('0' if myTree.lChild < 0.5 else '1', (plotTree.xOff,
                                                       plotTree.yOff), centerPt, leafNode)
        plotMidText((plotTree.xOff, plotTree.yOff),
                    centerPt, '<%.2f' % myTree.spVal)

    if isTree(myTree.rChild):
        plotTree(myTree.rChild, centerPt, '>%.2f' % myTree.spVal)
    else:
        plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
        plotNode('0' if myTree.rChild < 0.5 else '1', (plotTree.xOff,
                                                       plotTree.yOff), centerPt, leafNode)
        plotMidText((plotTree.xOff, plotTree.yOff),
                    centerPt, '>%.2f' % myTree.spVal)
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
    plotTree(inTree, (0.5, 1.0), '')
    plt.axis('off')
    plt.show()


createPlot(c.Tree)
