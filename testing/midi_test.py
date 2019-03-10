# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 17:51:45 2019

@author: oisin
"""

class Midi:
    """A representation of a midi file,
     can be written to an actual file through use of .write(filename)"""
    def __init__(self):
        self.format = 0
        self.tracks = 1
        self.division = 96
    
    def add_note(self, delta, note, velocity, channel=1):
        pass
    
    def write(self, filename):
        pass
    
    def vlq(self, value):
        pass

