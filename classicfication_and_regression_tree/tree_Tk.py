import classification_and_regression_trees as CART
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from numpy import *
from tkinter import *
import matplotlib
matplotlib.use('TkAgg')

random.seed(7)

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


def drawNewTree():
    tolN,tolS =getInputs()
    reDraw(tolS, tolN)


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

root.mainloop()

root.destroy()
