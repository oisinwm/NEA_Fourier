import os
import pickle
import sys
import time
import math
import threading

from classes import Matrix, Fourier, Wave, Midi
from PyQt5 import QtWidgets, QtCore


class Window(QtWidgets.QWidget):
    def __init__(self):
        QtWidgets.QWidget.__init__(self)
        self.btn = QtWidgets.QPushButton('Browse', self)
        self.btn.move(20, 20)
        self.btn.clicked.connect(self.get_files)

        self.start_btn = QtWidgets.QPushButton('Begin Conversion', self)
        self.start_btn.move(250, 100)
        self.start_btn.clicked.connect(self.begin)

        self.le = QtWidgets.QLineEdit(self)
        self.le.move(130, 22)
        self.le.resize(280, 20)
        self.le.setDisabled(True)
        
        self.progress = QtWidgets.QProgressBar(self)
        self.progress.setGeometry(110, 60, 300, 25)
        self.progress.setMaximum(100)

        self.setGeometry(300, 300, 500, 150)
        self.setWindowTitle('Wav to Midi Converter')
        self.show()

    def get_files(self):
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Single File', QtCore.QDir.rootPath() , '*.wav')
        self.le.setText(fileName)
        
    def begin(self):
        path = self.le.text()
        filename = path[-path[::-1].index("/"):]
        filesize = os.path.getsize(path)
        time_est = int(filesize * 0.0004161731354229695)
        QtWidgets.QMessageBox.question(self, 'Alert', f"Coversion has begun on {filename}, this may take a long time.\nEstimate of {time_est}s", QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
        thread = threading.Thread(target=self.main, args=(path,))
        thread.start()

    def main(self, path):
        filename = path[-path[::-1].index("/"):]

        FOURIER_SIZE = 2048
        FOURIER_INCREMENT = 256

        filename = "3_notes.wav"
        print(f"\nProcessing begun on file '{filename}', this will take a while.\n")

        loadStartTime = time.time()
        try:
            with open(filename[:-4] + ".pickle", "rb") as file:
                print("Cached file version found!\n")
                wave_file = pickle.load(file)
        except FileNotFoundError:
            print("No cache found.\n")
            wave_file = Wave(path)
            with open(filename[:-4] + ".pickle", "wb") as file:
                pickle.dump(wave_file, file, protocol=pickle.HIGHEST_PROTOCOL)
        loadEndTime = time.time()
        print(f"* Wave load complete. Elapsed time {loadEndTime - loadStartTime} seconds.")

        wave_channel = wave_file.get_channel(0)

        results_lst = []
        for offset in range((int(wave_channel.get_dim()[0]) - (
                FOURIER_SIZE - FOURIER_INCREMENT)) // FOURIER_INCREMENT):
            signal = Fourier(wave_channel.section(offset * FOURIER_INCREMENT,
                                                  (offset * FOURIER_INCREMENT + FOURIER_SIZE) - 1,
                                                  "h"), pad=True)
            results_lst.append(Fourier.rms(signal))

        v = Matrix([[i] for i in results_lst])
        x = [i[0] for i in Fourier.find_peaks(v, 10, 3, 0.1)]
        dividers = []
        prev = 0
        for i in range(1, len(x)):
            if x[i] == 1 and x[i - 1] == 0:
                if i - prev > 25:
                    prev = i
                    dividers.append(i)
        dividers.append(len(x))
        
        self.progress.setValue(5)
        noteEndTime = time.time()
        print(f"* Note partitioning complete. Elapsed time {noteEndTime - loadEndTime} seconds.")

        midi_file = Midi()
        
        
        if len(dividers) > 0:
            start = 0
            total = len(dividers)
            for j in dividers:
                current = dividers.index(j)
                self.progress.setValue(int((current*95)/total) + 5)
                end = j * FOURIER_INCREMENT
                # print(f"length - {start}, {end}")
                if start != end:
                    signal = Fourier(wave_channel.section(start, (end) - 1, "h"), pad=True)
                    signal = Fourier.blackman_harris(signal)
                    corr = abs(Fourier.FFT(signal))
                    post = Fourier.median_filter(corr, 15).section(0, corr.get_dim()[0] // 2, "h")

                    value = max([i[0] for i in post])
                    pos = post._contents.index([value])
                    hz_post = wave_file.convert_hertz(post)
                    # print(hz_post[pos][0])
                    if hz_post[pos][0] > 0:
                        midi_file.add_note(start, end, hz_post[pos][0], 40)
                start = end

        else:
            length = 2 ** int(math.log(wave_file.get_data()[0].get_dim()[0] - 1, 2))
            # print(f"length - {length}")
            signal = Fourier(wave_channel.section(0, length - 1, "h"), pad=True)
            corr = abs(Fourier.autocorrelation(signal))
            post = Fourier.median_filter(corr, 15).section(0, corr.get_dim()[0] // 2, "h")

        fourierEndTime = time.time()
        print(
            f"* Fourier transforms complete. Elapsed time {fourierEndTime - noteEndTime} seconds.")
        
        self.progress.setValue(100)
        midi_file.write(filename[:-4] + ".mid")
        endEndTime = time.time()
        print(f"* Midi file write complete. Elapsed time {endEndTime - fourierEndTime} seconds.")
        print(f"Total elapsed time {endEndTime - loadStartTime} seconds.")
    
    
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    ex = Window()
    sys.exit(app.exec_())
