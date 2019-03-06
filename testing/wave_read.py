# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 21:49:20 2019

@author: oisin
"""

import matplotlib.pyplot as plt
import numpy as np
import wave
import sys


spf = wave.open('24nocturnea.wav','r')

#Extract Raw Audio from Wav File
signal = spf.readframes(-1)
signal = np.fromstring(signal, 'Int16')


#If Stereo
if spf.getnchannels() == 2:
    left = signal[::2]
    right = signal[1::2]

    print(left.size, spf.getnframes())
    plt.figure(1)
    plt.title('Signal Wave Right...')
    plt.plot(right[:2**5])
    plt.show()
    
    plt.figure(2)
    plt.title('Signal Wave Left...')
    plt.plot(left[:2**5])
    plt.show()
else:
    plt.figure(1)
    plt.title('Signal Wave Full...')
    plt.plot(signal[:2**5])
    plt.show()
# Data begins at offset 3C