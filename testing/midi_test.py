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
        bits = list(bin(value)[2:])
        while len(bits) % 7 != 0:
            bits = ["0"] + bits
        rev_bits = bits[::-1]
        result = []
        for i, value in enumerate(rev_bits):
            result.append(value)
            if (i+1) == 7:
                result.append("0")
            elif (i+1) % 7 == 0:
                result.append("1")
        binary_str = "".join(result)[::-1]
        hex_result = [hex(int(binary_str[i:i + 4],2))[2:] for i in range(0, len(binary_str), 4)]
        return "".join(hex_result)
        
if __name__ == "__main__":
    a = Midi()