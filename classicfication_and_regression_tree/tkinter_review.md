
# 用Tkinter展示CART的结果

## 引入相关组件

导入[这篇文章](http://101.133.217.104/pythonnumpy%e6%9e%84%e5%bb%ba%e5%88%86%e7%b1%bb%e5%9b%9e%e5%bd%92%e6%a0%91/)里写好的CART

    import classification_and_regression_trees as CART

导入其他依赖项（需要pip inatall tkinter）

    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    from numpy import *
    from tkinter import *
    import matplotlib
    matplotlib.use('TkAgg')
    random.seed(7)

## 定义ReDraw按钮

    def reDraw(tolS, tolN):
        reDraw.f.clf()
        reDraw.a = reDraw.f.add_subplot(111)
        if chkBtnVar.get():
            if tolN < 2:
                tolN = 2
            myTree = CART.createTree(
                CART.myData, CART.modelLeaf, CART.modelErr, (tolS, tolN))
            yHat = CART.createFore(myTree, CART.testData, CART.modelTreeEval)
        else:
            myTree = CART.createTree(CART.myData, ops=(tolS, tolN))
            yHat = CART.createFore(myTree, CART.testData, CART.regTreeEval)
        reDraw.a.scatter(CART.myData[:, 0].T.tolist()[
                        0], CART.myData[:, 1].T.tolist()[0], s=5)
        reDraw.a.plot(CART.testData[:, 0].T.tolist()[0], yHat)
        reDraw.canvas.draw()

## 得到输入框的数据

    def getInputs():
        try: tolN = int(tolNentry.get())
        except:
            tolN = 10
            print("enter Integer for tolN")
            tolNentry.delete(0,END)
            tolNentry.insert(0,'10')
        try: tolS = float(tolSentry.get())
        except:
            tolS = 1.0
            print("enter Float for tolS")
            tolSentry.delete(0,END)
            tolSentry.insert(0,'1.0')
        return tolN, tolS

## 定义画图的主函数

    def drawNewTree():
        tolN,tolS =getInputs()
        reDraw(tolS, tolN)

## 安排配件

    root = Tk()

    reDraw.f = Figure(figsize=(5, 4), dpi=100)
    reDraw.canvas = FigureCanvasTkAgg(reDraw.f, master=root)
    reDraw.canvas.draw()
    reDraw.canvas.get_tk_widget().grid(row=0, columnspan=3)

    Label(root, text='tolN').grid(row=1, column=0)
    tolNentry = Entry(root)
    tolNentry.grid(row=1, column=1)
    tolNentry.insert(0, '10')
    Label(root, text='tolS').grid(row=2, column=0)
    tolSentry = Entry(root)
    tolSentry.grid(row=2, column=1)
    tolSentry.insert(0, '1.0')
    Button(root, text='ReDraw', command=drawNewTree).grid(
        row=1, column=2, rowspan=3)

    chkBtnVar = IntVar()
    chkBtn = Checkbutton(root, text='Model Tree', variable=chkBtnVar)
    chkBtn.grid(row=3, column=0, columnspan=2)

    reDraw.rawDat = mat(CART.testData)
    reDraw.testDat = arange(
        min(reDraw.rawDat[:, 0]), max(reDraw.rawDat[:, 0]), 0.01)

    reDraw(1.0, 10)

## 开始画图

    root.mainloop()

## 结果分析

绘图结果如下：

![tkinter_CART](http://q5ioolwed.bkt.clouddn.com/tkinter_CART.jpg)

可见线性模型在数据点较规整时效果十分不错  
但是如果数据分布不规整，线性模型比较容易出现过拟合的情况  
而普通模型在两种情况下都有稳定的表现  
