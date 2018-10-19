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

        filesize = contents[4:8].hex()
        self.filesize = int(self.little_endian(filesize), 2)  # This correctly calculates filesize
        headerChunk = contents[:12]
        print(self.filesize)

        fmtSizeHex = contents[16:20].hex()
        fmtSize = int(self.little_endian(fmtSizeHex), 2)  # This does not correctly calculate
        #                                                        chunksize, seems it isn't actually little endian
        print(fmtSize)

        formatChunk = contents[12:50]
        # bytes 12:16 'fmt '
        print(formatChunk)

        sample = contents[20:21].hex()
        self.samplerate = int(self.little_endian(sample), 2)
        print(self.samplerate)

    def little_endian(self, rawbytes):
        digits = "0123456789abcdef"
        binaryDigits = []
        for i in rawbytes:
            if str(i) in digits:
                digi = bin(digits.index(str(i)))[2:].zfill(4)[::-1]
                binaryDigits.append(digi)
        return "".join(binaryDigits)[::-1]


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