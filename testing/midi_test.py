# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 17:51:45 2019

@author: oisin
"""
import json
import math
from mido import MidiFile

mid = MidiFile('station6_real.mid')
for i, track in enumerate(mid.tracks):
    print('Track {}: {}'.format(i, track.name))
    for msg in track:
        print(msg)