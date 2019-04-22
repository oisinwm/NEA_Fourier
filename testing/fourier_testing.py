import cmath
import math
import matplotlib.pyplot as plt


class Matrix:
    """A  n*m matrix class, can be constructed from a list of objects or a 2d list of objects
        e.g.
        a = Matrix([[1,2], [3,4]])
        a = Matrix(m=10, n=10)
        Methods:
            __init__
            __rmul__
            __add__
            __sub__
    """

    def __init__(self, *args, **kwargs):
        self._contents = []
        self._dimensions = [0, 0]
        self.number_types = (int, float, complex)
        #self.matrix_types = (__name__.Matrix, __name__.Fourier)
        if len(args) == 0 and len(kwargs) == 2:
            # construct from a and b, if validation passes
            if kwargs.keys() == {"m": 0, "n": 0}.keys():
                if isinstance(kwargs["m"], int) and isinstance(kwargs["n"], int):
                    if kwargs["m"] > 0 and kwargs["n"] > 0:
                        # construct
                        self._dimensions = [kwargs["m"], kwargs["n"]]
                        self._contents = [[0 for x in range(kwargs["n"])] for y in
                                          range(kwargs["m"])]
                    else:
                        raise TypeError("Matrix dimension cannot be less than 0")

                else:
                    raise TypeError(f"""Invalid type for dimensions: 
                                    [{type(kwargs['m'])}, {type(kwargs['n'])}]""")

            else:
                raise TypeError(f"""Invalid kwargs {kwargs} must be 'm' and 'n'""")

        elif len(args) == 1 and len(kwargs) == 0:
            # construct from values
            if isinstance(args[0], list):
                if len(args[0]) > 0:
                    # Can construct from list of objects OR
                    # List of lists of objects
                    if isinstance(args[0][0], list):
                        n = len(args[0][0])
                        # print(args, args[0], type(args), type(args[0]))
                        for x in range(len(args[0])):
                            if not isinstance(args[0][x], list):
                                raise TypeError(
                                    "Invalid values for Matrix, must be only of type list")
                            for y in args[0][x]:
                                if isinstance(y, list):
                                    raise TypeError(
                                        "Invalid values for 2D Matrix, must not be list")
                                if len(args[0][x]) != n:
                                    raise TypeError(
                                        "Invalid values for 2D Matrix, must not of equal width")
                        # At this point, valid list of lists
                        self._contents = args[0]
                        self._dimensions = [len(args[0]), len(args[0][0])]

                    else:
                        # At this point a 1D list is detected
                        for x in args[0]:
                            if isinstance(x, list):
                                raise TypeError(
                                    "Invalid values for Matrix, must be only of type: list")

                        self._dimensions = [1, len(args[0])]
                        self._contents = [list(args[0])]
                        if all(issubclass(type(x), type(self)) or issubclass(type(self), type(x)) for x in args[0]):
                            self._contents = self._contents[0]

            elif issubclass(type(args[0]), type(self)) or issubclass(type(self), type(args[0])):
                self._contents = args[0]._contents
                self._dimensions = args[0]._dimensions
            else:
                raise TypeError(f"Invalid type for Matrix: {type(args[0])}, must be list or matrix")

        else:
            # don't construct, incorrect information given
            raise TypeError(f"""Invalid input length for Matrix: {type(args)}, 
                                must be exactly 1 list""")

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._contents[key]
        elif isinstance(key, slice):
            if self.get_dim()[1] == 1:
                data = self._contents[key]
                return Matrix(data)
            else:
                raise KeyError("Slice not supported for matrices, only vectors")
        else:
            raise KeyError

    def __setitem__(self, key, value):
        # print(self._contents, key)
        if isinstance(key, int):
            if self._dimensions[0] >= key:
                self._contents[key] = value
            else:
                raise KeyError
        else:
            raise KeyError

    def __repr__(self):
        return str(self._contents)

    def __rmul__(self, other):
        if isinstance(other, list):
            if len(other) == 1:
                other = other[0]
        if isinstance(other, self.number_types):
            result_matrix = Matrix(m=self._dimensions[0], n=self._dimensions[1])
            for y in range(len(self._contents)):
                if isinstance(self[0], list):
                    for x in range(len(self[y])):
                        result_matrix[y][x] = self[y][x] * other
                else:
                    # in this case a 1d matrix
                    result_matrix[y] = self[y] * other

        elif issubclass(type(other), type(self)) or issubclass(type(self), type(other)):
            # Matrix multiplication should be handled by mul not rmul,
            #  if being found here then an error has occurred
            raise NotImplementedError("Matrix multiplication should be handled by mul")
        else:
            raise TypeError
        return result_matrix

    def __mul__(self, other):
        if issubclass(type(other), type(self)) or issubclass(type(self), type(other)):
            if self._dimensions[1] != other.get_dim()[0]:
                raise ValueError(f"Cannot multiply matrices of incorrect dimensions, "
                                 f"self n ({self._dimensions[1]}) != other "
                                 f"m ({other.get_dim()[0]})")
            else:
                # Multiply two matrices with the correct dimensions
                x = self.get_dim()[0]
                y = other.get_dim()[1]
                result_matrix = Matrix(m=x, n=y)

                for i in range(self._dimensions[0]):
                    for c in range(other.get_dim()[1]):
                        num = 0  # This is an issue when adding two matrices
                        for j in range(other.get_dim()[0]):
                            a = self[i][j] * other[j][c]
                            if num == 0:
                                num = a
                            else:
                                num += a
                        result_matrix[i][c] = num

        elif isinstance(other, self.number_types):
            result_matrix = self.__rmul__(other)
        else:
            raise TypeError
        return result_matrix

    def __add__(self, other):
        if issubclass(type(other), type(self)) or issubclass(type(self), type(other)):
            if self.get_dim() != other.get_dim():
                raise ValueError(
                    f"Cannot add matrices of different dimensions, self ({self.get_dim()}) != other ({other.get_dim()})")
            else:
                x = self.get_dim()[0]
                y = self.get_dim()[1]
                result_matrix = Matrix(m=x, n=y)
                for i in range(x):
                    for j in range(y):
                        result_matrix[i][j] = self[i][j] + other[i][j]
                return result_matrix

        else:
            raise NotImplementedError(
                f"unsupported operand type(s) for +: '{type(self)}' and '{type(other)}'")

    def __sub__(self, other):
        if issubclass(type(other), type(self)) or issubclass(type(self), type(other)):
            if self.get_dim() != other.get_dim():
                raise ValueError(
                    f"Cannot multiply matrices of different dimensions, self ({self.get_dim()}) != other ({other.get_dim()})")
            else:
                x = self.get_dim()[0]
                y = self.get_dim()[1]
                result_matrix = Matrix(m=x, n=y)
                for i in range(x):
                    for j in range(y):
                        result_matrix[i][j] = self[i][j] - other[i][j]
                return result_matrix
        else:
            raise NotImplementedError(
                f"unsupported operand type(s) for +: '{type(self)}' and '{type(other)}'")

    def __abs__(self):
        x = self.get_dim()[0]
        y = self.get_dim()[1]
        result_matrix = Matrix(m=x, n=y)
        for i in range(x):
            for j in range(y):
                result_matrix[i][j] = abs(self[i][j])
        return result_matrix

    def get_dim(self):
        return self._dimensions
    
    @staticmethod
    def exp(x, lst):
        result = []
        for i in lst:
            result += [cmath.exp(x * i)]
        return Matrix([result])
    
    def concatenate(self, other, direction):
        other_m, other_n = zip(other.get_dim())
        other_m, other_n = other_m[0], other_n[0]
        if not (issubclass(type(other), type(self)) or issubclass(type(self), type(other))):
            raise ValueError(f"unsupported type for concatanate: {type(other)}")
        if direction in ["vertical", "v"]:
            if not self.get_dim()[1] == other.get_dim()[1]:
                raise ValueError

            result = Matrix(m=self.get_dim()[0] + other.get_dim()[0],
                            n=self.get_dim()[1])

            for y in range(self.get_dim()[0]):
                for x in range(result.get_dim()[1]):
                    result[y][x] = self[y][x]

            for y in range(self.get_dim()[0], result.get_dim()[0]):
                for x in range(result.get_dim()[1]):
                    result[y][x] = other[y - self.get_dim()[0]][x]

        elif direction in ["horizontal", "h"]:
            if not self.get_dim()[0] == other.get_dim()[0]:
                raise ValueError

            result = Matrix(m=self.get_dim()[0],
                            n=self.get_dim()[1] + other.get_dim()[1])

            for x in range(self.get_dim()[1]):
                for y in range(result.get_dim()[0]):
                    result[y][x] = self[y][x]

            for x in range(self.get_dim()[1], result.get_dim()[1]):
                for y in range(result.get_dim()[0]):
                    result[y][x] = other[y][x - self.get_dim()[1]]

        else:
            raise ValueError

        return result
        
    def section(self, mini, maxi, direction):
        # Takes either a horizontal or vertical slice of the matrix between
        # mini and maxi inclusive
        if direction in ["vertical", "v"]:
            result = Matrix(m=self.get_dim()[0], n=maxi-mini+1)
            for i in range(result.get_dim()[0]):
                result[i] = self[i][mini:(maxi+1)]
        
        elif direction in ["horizontal", "h"]:
            result = Matrix(m=maxi-mini+1, n=self.get_dim()[1])
            for i in range(result.get_dim()[0]):
                result[i] = self[i+mini]
        else:
            raise ValueError(f"Incorrect direction for sectioning: {direction}")
        
        return result


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
        self.fileSize = int(Wave.little_bin(fileSize), 2)  # This correctly calculates filesize
        headerChunk = contents[:12]

        fmtSizeRaw = contents[16:20]
        fmtSize = int(Wave.little_bin(fmtSizeRaw), 2)
        # formatChunk = contents[12:20+fmtSize]
        # bytes 12:16 'fmt '
        sampleRate = contents[24:26]
        self.sampleRate = int(Wave.little_bin(sampleRate), 2)

        channels = contents[22:24]
        self.channels = int(Wave.little_bin(channels), 2)

        frameSize = contents[32:34]
        self.frameSize = int(Wave.little_bin(frameSize), 2)

        bitDepth = contents[34:36]
        self.bitDepth = int(Wave.little_bin(bitDepth), 2)

        # bytes 36:40 = "data"
        dataLen = contents[40:44]
        self.dataLen = int(Wave.little_bin(dataLen), 2)

        # Read in data from array
        self.frameStartIndex = 44

        framesNum = self.dataLen / self.frameSize
        if framesNum.is_integer():
            framesNum = int(framesNum)
        else:
            raise ValueError("Non integer frame number")

        self.frameDataLists = [[] for i in range(self.channels)]
        for frame in range(framesNum):
            start = self.frameStartIndex + frame * self.frameSize
            end = self.frameStartIndex + (frame + 1) * self.frameSize
            data = contents[start:end]
            if not len(data) == self.channels * self.bitDepth // 8:
                raise ValueError("Invalid bit depth")
            n = self.bitDepth // 8
            samples = [data[i:i + n] for i in range(0, len(data), n)]
            for i in range(self.channels):
                self.frameDataLists[i].append([self.signed_int(samples[i])])

        self.dataMatrix = Matrix(self.frameDataLists[0])
        for channel in range(len(self.frameDataLists)):
            self.dataMatrix.concatenate(Matrix(self.frameDataLists[channel]), "h")

    @staticmethod
    def little_bin(rawbytes):
        """Returns the binary representation of an unsigned 32 bit integer,
            from little endian hex"""
        bytez = []
        for i in rawbytes:
            bytez.append(hex(i)[2:].zfill(2))
        hexstr = "".join(bytez[::-1])
        result = ""
        for x in hexstr:
            digits = bin(int(x, 16))[2:].zfill(4)
            result += digits
        return result

    def signed_int(self, rawbytes):
        """Returns the integer representation of a signed integer,
            from binary"""
        if self.bitDepth == 8 or self.bitDepth == 16:
            binary = Wave.little_bin(rawbytes)
            if binary[0] == "1":
                res = -32768 + int(Wave.little_bin(rawbytes)[1:], 2)
            else:
                res = int(Wave.little_bin(rawbytes), 2)
            return res

        elif self.bitDepth == 32:
            # Data is a float (-1.0f ro 1f)
            raise NotImplementedError("Cannot read 32 bit wave file yet")

    def get_channel(self, chan):
        return self.dataMatrix.section(chan, chan, "v")

    def convert_hertz(self, vector):
        """Converts a fourier transform output index to its value in Hz"""
        N = vector.get_dim()[0]
        T = N / self.sampleRate
        df = 1 / T
        result = Matrix(m=N, n=1)
        for n in range(N):
            if n < N / 2:
                result[n][0] = df * n // 2
            else:
                result[n][0] = df * (n - N) // 2
        return result


class Fourier(Matrix):
    """Performs a fourier transform on one Matrix of time domain values and returns a Matrix of
    frequency domain values"""

    def __init__(self, matrix, pad=False):
        super().__init__(matrix)
        self._p = math.ceil(math.log(matrix.get_dim()[0], 2))
        
        if pad:
            length = 2**self._p - matrix.get_dim()[0]
            if length > 0:
                left = math.ceil(length/2)
                right = length//2
                self._contents = Matrix(m=left, n=1).concatenate(self, "v").concatenate(Matrix(m=right, n=1), "v")._contents
            self._dimensions[0] = 2**self._p
        
        self._omega_N = cmath.exp(-2j * math.pi / self.get_dim()[0])

    def DFT(self):
        N = self.get_dim()[0]
        operator = Matrix(m=N, n=N)
        for x in range(N):
            for y in range(N):
                operator[y][x] = self._omega_N ** (x * y)
        return operator * self
    
    @staticmethod
    def find_peaks(vector, lag, threshold, influence):
        """Algorithim from https://stackoverflow.com/questions/22583391/peak-signal-detection-in-realtime-timeseries-data/22640362#22640362"""
        signals = Matrix(m=vector.get_dim()[0], n=1)
        filteredY = vector
        avgFilter = Matrix(m=vector.get_dim()[0], n=1)
        stdFilter = Matrix(m=vector.get_dim()[0], n=1)
        avgFilter[lag-1] = Fourier.avg(vector.section(0, lag-1, "h"))
        stdFilter[lag-1] = Fourier.std(vector.section(0, lag-1, "h"))
        
        for i in range(lag+1, vector.get_dim()[0]):
            if abs(vector[i][0] - avgFilter[i-1][0]) > threshold * stdFilter[i-1][0]:
                if vector[i][0] > avgFilter[i-1][0]:
                    signals[i][0] = 1
                else:
                    signals[i][0] = -1
                filteredY[i][0] = influence*vector[i][0] + (1-influence)*filteredY[i-1][0]
                
            else:
                signals[i][0] = 0
                filteredY[i][0] = vector[i][0]
            
            avgFilter[i][0] = Fourier.avg(filteredY.section(i-lag, i, "h"))
            stdFilter[i][0] = Fourier.std(filteredY.section(i-lag, i, "h"))
        
        return signals
    
    @staticmethod
    def filter_peaks(lst):
        lst = list(lst)
        result_list = []
        while len(lst) > 0:
            temp = [[lst.pop(0)]]
            while len(lst) > 0 and abs(Fourier.avg(Matrix(temp)) - lst[0]) < 6*len(temp):
                temp.append([lst.pop(0)])
            result_list.append(int(Fourier.avg(Matrix(temp))))
        return result_list
        
    @staticmethod
    def avg(vector):
        n = vector.get_dim()[0]
        total = sum([i[0] for i in vector._contents])
        return total / n
    
    @staticmethod
    def std(vector):
        n = vector.get_dim()[0]
        total = sum([i[0]**2 for i in vector])
        return math.sqrt(total / n - Fourier.avg(vector)**2)

    def FFT(self):
        N = self.get_dim()[0]
        if N <= 2:
            return self.DFT()
        else:
            even, odd = self[::2], self[1::2]
            even, odd = Fourier(Matrix(even)), Fourier(Matrix(odd))
            even, odd = even.FFT(), odd.FFT()
            
            factor = Fourier.exp(-2j * math.pi / N, list(range(N)))
            
            first = Matrix(m=even.get_dim()[0], n=even.get_dim()[1])
            second = Matrix(m=even.get_dim()[0], n=even.get_dim()[1])
            
            for i in range(even.get_dim()[0]):
                first[i][0] = even[i][0] + factor[0][:N // 2][i] * odd[i][0]
                second[i][0] = even[i][0] + factor[0][N // 2:][i] * odd[i][0]
            
            return Fourier(first.concatenate(second, "v"))

    @staticmethod
    def rms(vector):
        return math.sqrt(sum([i[0] ** 2 for i in vector]) / vector.get_dim()[0])

    @staticmethod
    def blackman_harris(vector):
        a = [0.35875, 0.48829, 0.14128, 0.01168]
        N = vector.get_dim()[0]
        result = Fourier(Matrix(m=N, n=1))
        for i in range(N):
            window = a[0] - a[1] * math.cos((2 * math.pi * i) / (N - 1)) + a[2] * math.cos(
                (4 * math.pi * i) / (N - 1)) - a[3] * math.cos((6 * math.pi * i) / (N - 1))
            result[i][0] = window * vector[i][0]
        return result

    @staticmethod
    def med(vector):
        values = sorted([i[0] for i in vector])
        if len(values) % 2 == 0:
            x = values[len(values) // 2 - 1:len(values) // 2 + 1]
            return sum(x) / 2
        else:
            return values[len(values) // 2]

    @staticmethod
    def median_filter(vector, size):
        y = Matrix(m=vector.get_dim()[0], n=1)
        y[0][0] = vector[0][0]
        y[vector.get_dim()[0] - 1][0] = vector[vector.get_dim()[0] - 1][0]
        end = size / 2
        for i in range(int(vector.get_dim()[0] - end) - size):
            # print(vector.get_dim()[0], i+size-1, i, int(vector.get_dim()[0]-end))
            window = vector.section(i, i + size - 1, "h")
            y[int(end + i)][0] = Fourier.med(window)
        return y


class Midi:
    """A representation of a midi file,
     can be written to an actual file through use of .write(filename)"""

    def __init__(self):
        self.format = 0
        self.tracks = 1
        self.division = 96
        self.events = [(0, 0)]

    def hz_to_key(self, hz):
        x = int(69 + 12 * math.log(hz / 440, 2))
        if x not in list(range(128)):
            print(f"broken {x}")
        return hex(x)[2:]

    def velocity_to_hex(self, v):
        return "40"

    def sample_to_tick(self, sample):
        return int(int(sample) // (44100 / (2 * self.division)))

    def add_note(self, start_sample, end_sample, note, velocity, channel=0):
        # At 120 BPM, 1s = 2b
        # 96 ticks per 1/4 note
        # 230 samples per tick
        note_on = "9" + hex(channel)[2:] + self.hz_to_key(note) + self.velocity_to_hex(velocity)
        note_off = "8" + hex(channel)[2:] + self.hz_to_key(note) + "40"
        if int(end_sample) - int(start_sample) > 1000:
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
        delta_times = [ordered_events[i][0] - ordered_events[i - 1][0] for i in
                       range(1, len(ordered_events))]
        delta_vlq = [Midi.vlq(i) for i in delta_times]
        for index, event in enumerate(ordered_events):
            if index != 0:  # Empty event to begin
                track_data += delta_vlq[index - 1] + event[1]

        track_data += "19ff2f0000ff2f00"  # End of track event

        # Prepare track header
        track_header = "4d54726b"
        track_header += hex(len(track_data) // 2)[2:].zfill(8)

        # Write file
        final_hex_string = header + track_header + track_data
        with open(filename, "wb") as midi_file:
            midi_file.write(bytearray.fromhex(final_hex_string))

    @staticmethod
    def vlq(value):
        bits = list(bin(value)[2:])
        while len(bits) % 7 != 0:
            bits = ["0"] + bits
        rev_bits = bits[::-1]
        result = []
        for i, value in enumerate(rev_bits):
            result.append(value)
            if (i + 1) == 7:
                result.append("0")
            elif (i + 1) % 7 == 0:
                result.append("1")
        binary_str = "".join(result)[::-1]
        hex_result = [hex(int(binary_str[i:i + 4], 2))[2:] for i in range(0, len(binary_str), 4)]
        return "".join(hex_result)


if __name__ == "__main__":
    def sample(amp, freq, x):
        return amp * math.sin(freq*x*2*math.pi)


    def generate_test_matrix(frequency, samplerate, length):
        """Given a frequency(Hz), samplerate(Hz) and length(s) returns a matrix of the samples that
        sound would create, if it were a recording of sound from a wave file."""
        amplitude = 10
    
        sample_list = []
        for y in range(int(length*samplerate)):
            # x is the integer part of time, y is the decimal
            x = y // samplerate
            t = x + (y/samplerate)
    
            samp = sample(amplitude, frequency, t)
            sample_list.append([samp])
    
        return sample_list
    
    def convert_hertz(sampleRate, vector):
        """Converts a fourier transform output index to its value in Hz"""
        N = vector.get_dim()[0]
        T = N / sampleRate
        df = 1 / T
        result = Matrix(m=N, n=1)
        for n in range(N):
            if n < N / 2:
                result[n][0] = df * n // 2
            else:
                result[n][0] = df * (n - N) // 2
        return result
    
    wave_1 = Matrix(generate_test_matrix(200, 44100, 0.011609977324263039))
    wave_2 = Matrix(generate_test_matrix(500, 44100, 0.011609977324263039))
    wave_combined = wave_1+wave_2
    plt.plot([i for i in wave_1])
    plt.plot([i for i in wave_2])
    plt.plot([i for i in wave_combined])
    plt.legend(["300Hz Sound Wave", "400Hz Sound Wave", "Combined Sound Wave"])
    plt.xlabel("Time (Sample Number)")
    plt.ylabel("Amplitude")
    plt.title("Fourier Input Example with 44100Hz Sample Rate")
    plt.show()
    
    result_1 = abs(Fourier.FFT(wave_combined))
    plt.plot([i for i in result_1][:result_1.get_dim()[0]//2])
    plt.xlabel("FFT Bin")
    plt.ylabel("Probability")
    plt.title("Raw Fourier Output")
    plt.legend(["FFT of Combined Sound Wave", "First Peak in Bin 3, ~258Hz", "First Peak in Bin 5, ~430Hz"])
    plt.show()
    
    result_2 = abs(Fourier.FFT(Fourier.blackman_harris(wave_combined)))
    plt.plot([i for i in result_2][:result_2.get_dim()[0]//2])
    plt.xlabel("FFT Bin")
    plt.ylabel("Probability")
    plt.title("Blackman-Harris Fourier Output")
    plt.legend(["FFT of Combined Sound Wave", "First Peak in Bin 3, ~258Hz", "First Peak in Bin 5, ~430Hz"])
    plt.show()
    
    result_2 = Fourier.blackman_harris(Matrix([[1] for i in range(100)]))
    plt.plot([i for i in result_2])
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.title("Blackman-Harris Window")
    plt.fill([i for i in result_2], color=(31/256,119/256,180/256,0.7))
    plt.show()
    
