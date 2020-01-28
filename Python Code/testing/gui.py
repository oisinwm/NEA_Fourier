import json
import os
import pickle
import sys

from classes import Matrix, Fourier, Wave, Midi
from PyQt5 import QtWidgets, QtCore


class Window(QtWidgets.QWidget):
    def __init__(self):
        QtWidgets.QWidget.__init__(self)
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
        filesize = os.path.getsize(path)
        print(filesize)
        time_est = filesize * 1
        QtWidgets.QMessageBox.question(self, 'Alert', f"Coversion has begun on {filename}, this may take a long time.\nEstimate of s", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
    
    def main(self, path):
        filename = path[-path[::-1].index("/"):]

        try:
            with open(path[:-4] + ".pickle", "rb") as file:
                print("Cached file verison found!\n")
                a = pickle.load(file)
        except FileNotFoundError:
            print("No cache found.\n")
            a = Wave(filename)
            with open(path[:-4] + ".pickle", "wb") as file:
                pickle.dump(a, file, protocol=pickle.HIGHEST_PROTOCOL)
        
        FOURIER_INCREMENT = 512
        FOURIER_SIZE = 4096
        
        results_dict = {}
        #for offset in range(130):
        for offset in range((int(a.get_data()[0].get_dim()[0]) - (FOURIER_SIZE-FOURIER_INCREMENT)) // FOURIER_INCREMENT):
            b = Fourier(a.get_data()[0].section(offset*FOURIER_INCREMENT, (offset*FOURIER_INCREMENT+FOURIER_SIZE)-1, "h"), pad=True)
            #Shortest note appears to be 0.012 seconds long, bin number of 512
            
            final = Fourier.FFT(b) # Once transform is complete the values must be converted to hz
            conversion_vector = a.convert_hertz(final) # HO BOI, use this to look up from a conversion table to get hz
            
            results = Matrix([[abs(final[i][0])] for i in range(final.get_dim()[0]//2)])
            peak_pos = [i[0] for i in Fourier.find_peaks(results, 30, 6, 0.1)._contents]
            raw_peak_values = []
            for i in range(0, len(peak_pos)):
                if peak_pos[i]:
                    raw_peak_values += [i]
            
            filtered_peaks = Fourier.filter_peaks(raw_peak_values)
            hz_values = [conversion_vector[i][0] for i in filtered_peaks]
            filtereds_hz_values = [h for h in Fourier.filter_peaks(hz_values) if h not in [333, 4026]]
        
            results_dict[offset*FOURIER_INCREMENT] = list(filtereds_hz_values)
    
        with open(path[:-4] + "_data.json", "w") as file:
            file.write(json.dumps(results_dict).replace("], ","],\n"))
        
        with open(path[:-4] + "_data.json", "r") as file:
            results_dict = json.loads(file.read())
        
        midi_file = Midi()
        
        v = 0
        error = 0
        start_t = 0
        end_t = 0
        for key, value in results_dict.items():
            if len(value) > 3:
                if value[0] in list(range(int(v-error),int(v+error))):
                    v = (v+value[0])/2
                else:
                    end_t = key
                    if v != 0:
                        midi_file.add_note(start_t, end_t, v, 40)
                    v = value[0]
                    start_t = key
                error = v/30
                count = 1
                for i, num in enumerate(value):
                    if (i+1) * v in list(range(int(num-(i+1)*error), int(num+(i+1)*error))):
                        count +=1
                print(f"strength {count}")
        
        midi_file.write(path[:-4] + ".mid")
    
    
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    ex = Window()
    sys.exit(app.exec_())