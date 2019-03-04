# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 21:49:20 2019

@author: oisin
"""

import matplotlib.pyplot as plt
import numpy as np
import wave
import sys


spf = wave.open('24nocturnea_mono.wav','r')

#Extract Raw Audio from Wav File
signal = spf.readframes(-1)
signal = np.fromstring(signal, 'Int16')


#If Stereo
if spf.getnchannels() == 2:
    print('Just mono files')
    sys.exit(0)

plt.figure(1)
plt.title('Signal Wave...')
plt.plot(signal[:2**16])
plt.show()