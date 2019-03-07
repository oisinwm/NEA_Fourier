# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 21:25:33 2019

@author: oisin
"""
import numpy as np
import matplotlib.pyplot as plt

# create data
N = 4097
T = 100.0
t = np.linspace(-T/2,T/2,N)
f = np.exp(-np.pi*t**2)

# perform FT and multiply by dt
dt = t[1]-t[0]
ft = np.fft.fft(f) * dt      
freq = np.fft.fftfreq(N, dt)
freq = freq[:N//2+1]

# plot results
plt.plot(freq, np.abs(ft[:N//2+1]),'o')
plt.plot(freq, np.exp(-np.pi * freq**2),'r')
plt.legend(('numpy fft * dt', 'exact solution'), loc='upper right')
plt.xlabel('f')
plt.ylabel('amplitude')
plt.xlim([0, 1.4])
plt.show()