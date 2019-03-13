from PyQt5 import QtWidgets, QtCore
import sys

class Window(QtWidgets.QWidget):
    
    def __init__(self):
        super().__init__()
        self.initUI()
        
        
    def initUI(self):      

        self.btn = QtWidgets.QPushButton('Browse', self)
        self.btn.move(20, 20)
        self.btn.clicked.connect(self.getfiles)
        
        self.start_btn = QtWidgets.QPushButton('Begin Conversion', self)
        self.start_btn.move(250, 100)
        self.start_btn.clicked.connect(self.begin)
        
        self.le = QtWidgets.QLineEdit(self)
        self.le.move(130, 22)
        self.le.resize(280, 20)
        self.le.setDisabled(True)
        
        self.setGeometry(300, 300, 500, 150)
        self.setWindowTitle('Wav to Midi Converter')
        self.show()
        
    def getfiles(self):
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Single File', QtCore.QDir.rootPath() , '*.wav')
        self.le.setText(fileName)
        
    def begin(self):
        path = self.le.text()
        filename = path[-path[::-1].index("/"):]
        QtWidgets.QMessageBox.question(self, 'Alert', f"Coversion has begun on {filename}, this may take a long time", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
    
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    ex = Window()
    sys.exit(app.exec_())