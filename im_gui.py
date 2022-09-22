#import numpy

#import GUI
import fin_knn
import sys
from GUI import Ui_Dialog
from PyQt5 import QtCore, QtGui, QtWidgets
#from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
#from fin_knn import K_cla
#from tkinter import filedialog

#REURL = "/Users/meizhangyu/Desktop/KNN_soft/testdata123.csv"
#ARRY = [[1911, 15, 2, 1, 3]]

class mywindow(QtWidgets.QMainWindow,Ui_Dialog):
    def __init__(self):
        super(mywindow,self).__init__()

        self.setupUi(self)
        self.pushButton.clicked.connect(self.openFile)
        self.pushButton_4.clicked.connect(self.returnarry)
        self.pushButton_5.clicked.connect(self.k_main)







#        self.pushButton_5.clicked.connect(K_cla(str(clic1),clic2))


    '''
    def openFile(self):
        # 选择且获取图片文件的地址
#        fileName, filetype = QFileDialog.getOpenFileName(self)
        filurl=QFileDialog.getOpenFileUrl(self)
        URL = filurl[0].url()
        REURL=URL
        return URL






    def file(self):
    #    root.withdraw()
        c=filedialog.askopenfilename()
        print('Filepath:',c)
        return c


    def returnarry(self):
        a1 = self.getnumber2()
        a2 = self.getnumber3()
        a3 = self.getnumber4()
        a4 = self.getnumber5()
        a5 = self.getnumber6()
        ARRY = numpy.array([a1, a2, a3, a4, a5]).reshape(1, 5)
    '''
if __name__ == '__main__':

    '''
    app = QApplication(sys.argv)
    MainWindow=QMainWindow()
    ui = Ui_Dialog()
    ui.setupUi(mywindow())
    MainWindow.show()
    sys.exit(app.exec())
    button=Ui_Dialog.importfile()
    print(button)
    '''

    app = QtWidgets.QApplication(sys.argv)
    ui = mywindow()
    ui.show()
    sys.exit(app.exec())