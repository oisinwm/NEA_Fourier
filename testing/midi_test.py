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
        self.events = [(0,0)]
        
    def hz_to_key(self, hz):
        return "3c"
    
    def velocity_to_hex(self, v):
        return "40"
    
    def sample_to_tick(self, sample):
        return int(sample // (44100 / (2*self.division))) # Fix hardcoding
    
    def add_note(self, start_sample, end_sample, note, velocity, channel=0):
        # At 120 BPM, 1s = 2b
        # 96 ticks per 1/4 note
        # 230 samples per tick
        note_on = "9" + hex(channel)[2:] + self.hz_to_key(note) + self.velocity_to_hex(velocity)
        note_off = "8" + hex(channel)[2:] + self.hz_to_key(note) + "40"
        self.events.append((self.sample_to_tick(start_sample), note_on))
        self.events.append((self.sample_to_tick(end_sample), note_off))
        
    def write(self, filename):
        # Prepare file header
        header = "4d54686400000006" 
        header += hex(self.format)[2:].zfill(4)
        header += hex(self.tracks)[2:].zfill(4)
        header += hex(self.division)[2:].zfill(4)
        
        # Prepare track data
        track_data = ""
        ordered_events = list(sorted(self.events, key=lambda tup: tup[0]))
        delta_times = [ordered_events[i][0] - ordered_events[i-1][0] for i in range(1, len(ordered_events))]
        delta_vlq = [self.vlq(i) for i in delta_times]
        for index, event in enumerate(ordered_events):
            if index != 0: # Empty event to begin 
                track_data += delta_vlq[index-1] + event[1]
        
        #track_data += "00FF510307A12000FF7F0A53616D706C697475646500FF03084D49444920524543".lower()
        track_data += "19ff2f0000ff2f00" # End of track event
        
        #prepare track header
        track_header = "4d54726b"
        track_header += hex(len(track_data)//2)[2:].zfill(8)
        
        # Write file
        final_hex_string = header + track_header + track_data
        with open(filename, "wb") as midi_file:
            midi_file.write(bytearray.fromhex(final_hex_string))
    
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
    a.add_note(750,44100,0,0)
    a.add_note(30000,88000,0,0)
    a.write("first.mid")
