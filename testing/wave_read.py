# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 21:49:20 2019

@author: oisin
"""

import matplotlib.pyplot as plt
import numpy as np
import pickle
import math


filename = "24nocturnea.wav"
with open(filename + ".pickle", "rb") as file:
    a = pickle.load(file)


def sample(amp, freq, x):
        return amp * math.sin(freq*x*2*math.pi) #+ amp/2 * math.sin(freq*x*math.pi)

def generate_test_matrix(frequency, samplerate, length):
    """Given a frequency(Hz), samplerate(Hz) and length(s) returns a matrix of the samples that
    sound would create, if it were a recording of sound from a wave file."""
    amplitude = 4500

    sample_list = []
    for y in range(int(length*samplerate)):
        # x is the integer part of time, y is the decimal
        x = y // samplerate
        t = x + (y/samplerate)

        samp = sample(amplitude, frequency, t)
        sample_list.append(samp)
        

    return sample_list


x = [i[0] for i in a.get_data()[0].section(33075, (55125)-1, "h")._contents]

y = generate_test_matrix(1000, 44100, 0.125)

plt.plot(x)
plt.show()

res = np.fft.fft(x)
#res = np.ma.masked_where(abs(res) > 9000, res)
res = abs(res)
res = res[:res.size//2]
np.savetxt("foo.csv", res, delimiter=",")

plt.plot(res)
plt.show()

vector_res = Matrix([[i] for i in res])
peaks = [i[0] for i in Fourier.find_peaks(vector_res, 30, 5, 0.1)._contents]
for i in range(len(peaks)):
    if peaks[i] != 0:
        print(i)

plt.plot(peaks)
plt.show()
