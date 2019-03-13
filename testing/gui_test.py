# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 01:48:09 2019

@author: oisin
"""
from PyQt5 import QtWidgets, QtCore, QtGui
import sys


class Window(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super(Window, self).__init__(parent)
        self.browseButton = self.createButton("&Browse...")
        self.browseButton.clicked.connect(self.getfiles)
        
        
        self.setWindowTitle("Find Files")
        self.resize(700, 300)

    def getfiles(self):
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Single File', QtCore.QDir.rootPath() , '*.xlsm')
        self.ui.lineEdit.setText(fileName)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())