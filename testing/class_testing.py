import wave


class Wave:
    """A representation of a Wave file, must be created from a string containing the location of a
     wave file on disk"""
    def __init__(self, path):
        with open(path, "rb") as raw_wave_file:
            contents = raw_wave_file.read()

        # Check the header chunk for correctly set values
        # https://blogs.msdn.microsoft.com/dawate/2009/06/23/intro-to-audio-programming-part-2-demystifying-the-wav-format/
        if contents[0:4] != b"RIFF" or contents[8:12] != b"WAVE":
            raise TypeError("Specified file is not in wave format")

        filesize = contents[4:8]
        self.filesize = int(self.little_bin(filesize), 2)  # This correctly calculates filesize
        headerChunk = contents[:12]

        fmtSizeRaw = contents[16:20]
        fmtSize = int(self.little_bin(fmtSizeRaw), 2)

        formatChunk = contents[12:20+fmtSize]
        # bytes 12:16 'fmt '

        sample = contents[24:26]
        self.samplerate = int(self.little_bin(sample), 2)

        channels = contents[22:24]
        self.channels = int(self.little_bin(channels), 2)

        framesize = contents[32:34]
        self.framesize = int(self.little_bin(framesize), 2)

        bitdepth = contents[34:38]
        self.bitdepth = int(self.little_bin(bitdepth), 2)

        datalen = contents[42:46]
        self.datalen = int(self.little_bin(datalen), 2)

        # Read in data from array




    def little_bin(self, rawbytes):
        """Returns the integer representation of an unsigned 32 bit integer,
            stored as bytes in little endian"""
        # print(rawbytes)
        bytez = []
        for i in rawbytes:
            bytez.append(hex(i)[2:].zfill(2))
        hexstr = "".join(bytez[::-1])
        # at this point need a string of raw hex digits only
        result = ""
        for x in hexstr:
            digits = bin(int(x, 16))[2:].zfill(4)
            result += digits

        return result

    def signed_int(self, rawbytes):
        """Returns the integer representation of an unsigned 32 bit integer,
            stored as bytes in little endian"""
        if self.framesize == 8:
            pass
        if self.framesize == 16:
            pass
        if self.framesize == 32:
            pass


class Midi:
    """A representation of a midi file,
     can be written to an actual file through use of .write(filename)"""
    pass


class Fourier:
    """Performs a fourier transform on one Matrix of time domain values and returns a Matrix of
    frequency domain values"""
    pass


if __name__ == "__main__":
    jim = Wave("24nocturnea.wav")
    # with wave.open("24nocturnea.wav", 'rb') as f:
    #     framerate = f.getframerate()
    # print(framerate)