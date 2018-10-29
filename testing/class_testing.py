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
        print(self.filesize)

        fmtSizeRaw = contents[16:20]
        fmtSize = int(self.little_bin(fmtSizeRaw), 2)
        print(fmtSize)

        formatChunk = contents[12:50]
        # bytes 12:16 'fmt '
        print(formatChunk)

        sample = contents[24:26]
        self.samplerate = int(self.little_bin(sample), 2)
        print(self.samplerate)

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