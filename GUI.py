# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'GUI.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.
import sys
from tkinter import filedialog

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
#from theme import file
import numpy
import GUI
from tkinter import messagebox
import pictute_rc

from matplotlib import pyplot as plt
from numpy.distutils.fcompiler import pg

import fin_knn
from fin_knn import K_cla
global URL
global ARRY

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(696, 569)
        self.pushButton_4 = QtWidgets.QPushButton(Dialog)
        self.pushButton_4.setGeometry(QtCore.QRect(160, 140, 113, 32))
        self.pushButton_4.setObjectName("pushButton_4")
        self.pushButton_5 = QtWidgets.QPushButton(Dialog)
        self.pushButton_5.setGeometry(QtCore.QRect(380, 140, 113, 32))
        self.pushButton_5.setObjectName("pushButton_5")
        self.widget = QtWidgets.QWidget(Dialog)
        self.widget.setGeometry(QtCore.QRect(100, 20, 491, 103))
        self.widget.setObjectName("widget")
        self.gridLayout = QtWidgets.QGridLayout(self.widget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.pushButton = QtWidgets.QPushButton(self.widget)
        self.pushButton.setObjectName("pushButton")
        self.gridLayout.addWidget(self.pushButton, 0, 0, 1, 1)
#        self.lineEdit = QtWidgets.QLineEdit(self.widget)
#        self.lineEdit.setObjectName("lineEdit")
#        self.gridLayout.addWidget(self.lineEdit, 0, 1, 1, 2)
#        self.pushButton_2 = QtWidgets.QPushButton(self.widget)
#        self.pushButton_2.setObjectName("pushButton_2")
#        self.gridLayout.addWidget(self.pushButton_2, 1, 0, 1, 1)
#        self.lineEdit_2 = QtWidgets.QLineEdit(self.widget)
#        self.lineEdit_2.setObjectName("lineEdit_2")
#        self.gridLayout.addWidget(self.lineEdit_2, 1, 1, 1, 1)
        self.pushButton_3 = QtWidgets.QPushButton(self.widget)
        self.pushButton_3.setObjectName("pushButton_3")
        self.gridLayout.addWidget(self.pushButton_3, 2, 0, 1, 1)
        self.lineEdit_3 = QtWidgets.QLineEdit(self.widget)
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.gridLayout.addWidget(self.lineEdit_3, 2, 1, 1, 1)
        self.lineEdit_4 = QtWidgets.QLineEdit(self.widget)
        self.lineEdit_4.setObjectName("lineEdit_4")
        self.gridLayout.addWidget(self.lineEdit_4, 2, 2, 1, 1)
        self.lineEdit_5 = QtWidgets.QLineEdit(self.widget)
        self.lineEdit_5.setObjectName("lineEdit_5")
        self.gridLayout.addWidget(self.lineEdit_5, 2, 3, 1, 1)
        self.lineEdit_6 = QtWidgets.QLineEdit(self.widget)
        self.lineEdit_6.setObjectName("lineEdit_6")
        self.gridLayout.addWidget(self.lineEdit_6, 2, 4, 1, 1)
        self.lineEdit_7 = QtWidgets.QLineEdit(self.widget)
        self.lineEdit_7.setObjectName("lineEdit_7")
        self.gridLayout.addWidget(self.lineEdit_7, 2, 5, 1, 1)
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(70, 210, 581, 171))
        self.label.setStyleSheet("border-image: url(:/instac/instance.png);")
#        self.label.setStyleSheet("instance.png")
        self.label.setText("")
        self.label.setObjectName("label")




        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)




    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.pushButton_4.setText(_translate("Dialog", "提交预测参数"))
        self.pushButton_5.setText(_translate("Dialog", "运行程序"))
        self.pushButton.setText(_translate("Dialog", "选择文件"))
#        self.pushButton_2.setText(_translate("Dialog", "预测个数"))
        self.pushButton_3.setText(_translate("Dialog", "预测参数"))






    def getnumber(self):
        text=self.lineEdit_2.text()
        return int(text)
    def getnumber2(self):
        text=self.lineEdit_3.text()
        return int(text)
    def getnumber3(self):
        text=self.lineEdit_4.text()
        return int(text)
    def getnumber4(self):
        text=self.lineEdit_5.text()
        return int(text)
    def getnumber5(self):
        text=self.lineEdit_6.text()
        return int(text)
    def getnumber6(self):
        text=self.lineEdit_7.text()
        return int(text)
    def submit(self):
        self.pushButton_4
    def k_main(self):
        global URL,ARRY
        fin_knn.K_cla(URL,ARRY)

    def openFile(self):
            # 选择且获取图片文件的地址
            #        fileName, filetype = QFileDialog.getOpenFileName(self)
        filurl = QFileDialog.getOpenFileUrl(self)
        global URL
        URL = filurl[0].url()
#        messagebox.showinfo('文件路径',URL)
        print(URL)
        return URL


    def returnarry(self):
        a1 = self.getnumber2()
        a2 = self.getnumber3()
        a3 = self.getnumber4()
        a4 = self.getnumber5()
        a5 = self.getnumber6()
        global ARRY
        ARRY = numpy.array([a1,a2,a3,a4,a5]).reshape(1,5)
        print(ARRY)

#        K_cla(URL, ARRY)




'''
if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow=QMainWindow()
    ui = Ui_Dialog()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec())
'''