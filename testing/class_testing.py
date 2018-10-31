class Wave:
    """A representation of a Wave file, must be created from a string containing the location of a
     wave file on disk"""
    def __init__(self, path):
        with open(path, "rb") as raw_wave_file:
            contents = raw_wave_file.read()

        # Check the header chunk for correctly set values
        # https://blogs.msdn.microsoft.com/dawate/2009/06/23/intro-to-audio-programming-part-2-demystifying-the-wav-format/
        # The above article is nicely formatted but has a bunch of mistakes
        if contents[0:4] != b"RIFF" or contents[8:12] != b"WAVE":
            raise TypeError("Specified file is not in wave format")

        fileSize = contents[4:8]
        self.fileSize = int(self.little_bin(fileSize), 2)  # This correctly calculates filesize
        headerChunk = contents[:12]

        fmtSizeRaw = contents[16:20]
        fmtSize = int(self.little_bin(fmtSizeRaw), 2)
        # formatChunk = contents[12:20+fmtSize]
        # bytes 12:16 'fmt '
        sampleRate = contents[24:26]
        self.sampleRate = int(self.little_bin(sampleRate), 2)

        channels = contents[22:24]
        self.channels = int(self.little_bin(channels), 2)

        frameSize = contents[32:34]
        self.frameSize = int(self.little_bin(frameSize), 2)

        bitDepth = contents[34:38]
        self.bitDepth = int(self.little_bin(bitDepth), 2)
        # bytes 38:42 'data'
        dataLen = contents[42:46]
        self.dataLen = int(self.little_bin(dataLen), 2)

        # Read in data from array
        self.frameStartIndex = 46  # Not 100% sure if should be hardcoded or dynamic

        framesNum = self.dataLen / 8 / self.frameSize
        if framesNum.is_integer():
            framesNum = int(framesNum)
        else:
            raise ValueError("Non integer frame number")

        # print(framesNum)
        self.frameDataLists = [[] for i in range(self.channels)]
        for frame in range(framesNum-1):
            start = self.frameStartIndex + frame * self.frameSize
            end = self.frameStartIndex + (frame+1) * self.frameSize
            data = contents[start:end]
            for x in range(self.channels):
                s = x * self.bitDepth // 8
                e = (x + 1) * self.bitDepth // 8
                channelData = data[s:e]
                a = self.little_bin(channelData)
                b = self.signed_int(a)
                self.frameDataLists[x].append([b])

        self.dataMatrices = [Matrix(sampleList) for sampleList in self.frameDataLists]

    def little_bin(self, rawbytes):
        """Returns the binary representation of an unsigned 32 bit integer,
            from little endian hex"""
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
        """Returns the integer representation of a signed integer,
            from binary"""
        if self.bitDepth == 8:
            # Data is unsigned 8 bit integer (-128 to 127)
            return int(rawbytes, 2)
        elif self.bitDepth == 16:
            # Data is signed 16 bit integer (-32768 to 32768)
            return -32768 + int(rawbytes[1:], 2)
        elif self.bitDepth == 32:
            # Data is a float (-1.0f ro 1f)
            raise NotImplementedError("Cannot read 32 bit wave file yet")


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
