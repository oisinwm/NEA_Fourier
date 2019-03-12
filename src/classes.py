import cmath
import math


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
        
        return result_matrix

    def __mul__(self, other):
        if issubclass(type(other), type(self)) or issubclass(type(self), type(other)):
            if self._dimensions[1] != other._dimensions[0]:
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

        elif isinstance(other, self.number_types) or isinstance(self.number_types, other):
            result_matrix = self.__rmul__(other)
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

    def get_dim(self):
        return self._dimensions
    
    @staticmethod
    def exp(x, lst):
        result = []
        for i in lst:
            result += [cmath.exp(x * i)]
        return Matrix([result])
    
    def concatanate(self, other, direction):
        other_m, other_n = zip(other.get_dim())
        other_m, other_n = other_m[0], other_n[0]
        if not (issubclass(type(other), type(self)) or issubclass(type(self), type(other))):
            raise ValueError(f"unsupported type for concatanate: {type(other)}")
        if direction in ["vertial", "v"]:
            if not self.get_dim()[1] == other.get_dim()[1]:
                raise ValueError
            
            result = Matrix(m=self.get_dim()[0] + other.get_dim()[0], 
                            n=self.get_dim()[1])
            
            for y in range(self.get_dim()[0]):
                for x in range(result.get_dim()[1]):
                    result[y][x] = self[y][x]
            
            for y in range(self.get_dim()[0], result.get_dim()[0]):
                for x in range(result.get_dim()[1]):
                    result[y][x] = other[y-self.get_dim()[0]][x]
            
            return result
        
        elif direction in ["horizontal", "h"]:
            if not self.get_dim()[0] == other.get_dim()[0]:
                raise ValueError
            
            result = Matrix(m=self.get_dim()[0], 
                            n=self.get_dim()[1]+ other.get_dim()[1])
            
            for x in range(self.get_dim()[1]):
                for y in range(result.get_dim()[0]):
                    result[y][x] = self[y][x]
            
            for x in range(self.get_dim()[1], result.get_dim()[1]):
                for y in range(result.get_dim()[0]):
                    result[y][x] = other[y][x-self.get_dim()[1]]
            
            return result
        else:
            raise ValueError
            return 0
        
    def section(self, mini, maxi, direction):
        # Takes either a horizontal or vertical slice of the matrix between
        # mini and maxi inclusive
        if direction in ["vertial", "v"]:
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
        self.frameStartIndex = 44  # Not 100% sure if should be hardcoded or dynamic, mostly sure its constant
        print(self.dataLen, self.frameSize)
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
            samples = [data[i:i+n] for i in range(0, len(data), n)]
            for i in range(self.channels):
                self.frameDataLists[i].append([self.signed_int(samples[i])])

        self.dataMatrices = [Matrix(sampleList) for sampleList in self.frameDataLists]
    
    @staticmethod
    def little_bin(rawbytes):
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

    def get_data(self):
        return self.dataMatrices
    
    def convert_hertz(self, vector):
        """Converts a fourier transform output index to its value in Hz"""
        N = vector.get_dim()[0]
        T = N /self.sampleRate
        df = 1/T
        result = Matrix(m=N, n=1)
        for n in range(N):
            if n < N/2:
                result[n][0] = df*n
            else:
                result[n][0] = df*(n-N)
        return result
    
class Fourier(Matrix):
    """Performs a fourier transform on one Matrix of time domain values and returns a Matrix of
    frequency domain values"""

    def __init__(self, matrix, pad=False):
        Matrix.__init__(self, matrix)
        self._p = math.ceil(math.log(matrix.get_dim()[0], 2))
        
        if pad:
            length = 2**self._p - matrix.get_dim()[0]
            if length > 0:
                left = math.ceil(length/2)
                right = length//2
                self._contents = Matrix(m=left, n=1).concatanate(self, "v").concatanate(Matrix(m=right, n=1), "v")._contents
            self._dimensions[0] = 2**self._p
        
        self._omega_N = cmath.exp(-2j * math.pi / self.get_dim()[0])
        
    def get_p(self):
        return int(self._p)
        
    def get_omega(self):
        return self._omega_N
    
    def DFT(self):
        N = self.get_dim()[0]
        DFT_result_list = []
        for x in range(N):
            factor = []
            for y in range(N):
                factor.append(self._omega_N ** (x*y))
            factorVector = Matrix(factor)
            #factorVector = (1 / N) * factorVector
            answer = factorVector * self
            DFT_result_list.append(answer[0][0])
            
        return Fourier(Matrix([[i] for i in DFT_result_list]))
    
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
        total = sum([i[0]**2 for i in vector._contents])
        return math.sqrt(total / n - Fourier.avg(vector)**2)
    
    @staticmethod
    def FFT(vector):
        N = vector.get_dim()[0]
        if N <= 2:
            return vector.DFT()
        else:
            even, odd = vector._contents[::2], vector._contents[1::2]
            even, odd = Fourier(Matrix(even)), Fourier(Matrix(odd))
            even, odd = Fourier.FFT(even), Fourier.FFT(odd)
            
            factor = Fourier.exp(-2j * math.pi / N, list(range(N)))
            
            first = Matrix(m=even.get_dim()[0], n=even.get_dim()[1])
            second = Matrix(m=even.get_dim()[0], n=even.get_dim()[1])
            
            for i in range(even.get_dim()[0]):
                first[i][0] = even[i][0] + factor[0][:N // 2][i] * odd[i][0]
                second[i][0] = even[i][0] + factor[0][N // 2:][i] * odd[i][0]
            
            return Fourier(first.concatanate(second, "v"))
        
    @staticmethod
    def IFFT(v):
        """Not 100% correct, need to divide all final values by 1/N"""        
        reverse = Matrix(m=v.get_dim()[0], n=v.get_dim()[1])
        for i in range(v.get_dim()[0]):
            if i == 0:
                reverse[i][0] = v[i][0]
            else:
                reverse[i][0] = v[-i][0]
        result = Fourier.FFT(reverse)
        return (1/v.get_dim()[0]) * result
    
    @staticmethod
    def autocorrelation(vector):
        # Wiener–Khinchin theorem
        FR = Fourier.FFT(vector)
        S = FR
        for i in range(FR.get_dim()[0]):
            S[i][0] = FR[i][0] * FR[i][0].conjugate()
        R = Fourier.IFFT(S)
        return R
    
    @staticmethod
    def from_combine(mat, test):
        temp = Fourier(Matrix(m=mat.get_dim()[0], n=mat.get_dim()[1]))
        temp._contents = mat._contents
        temp._p = int(test._p)
        temp._omega_N = test._omega_N
        return temp


class Midi:
    """A representation of a midi file,
     can be written to an actual file through use of .write(filename)"""
    def __init__(self):
        self.format = 0
        self.tracks = 1
        self.division = 96
        self.events = [(0,0)]
        
    def hz_to_key(self, hz):
        return hex(int(69 + 12 * math.log(int(hz)/440, 2)))[2:]
    
    def velocity_to_hex(self, v):
        return "40"
    
    def sample_to_tick(self, sample):
        return int(int(sample) // (44100 / (2*self.division))) # Fix hardcoding
    
    def add_note(self, start_sample, end_sample, note, velocity, channel=0):
        # At 120 BPM, 1s = 2b
        # 96 ticks per 1/4 note
        # 230 samples per tick
        note_on = "9" + hex(channel)[2:] + self.hz_to_key(note) + self.velocity_to_hex(velocity)
        note_off = "8" + hex(channel)[2:] + self.hz_to_key(note) + "40"
        if int(end_sample) - int(start_sample) > 2800:
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
        for i, v in enumerate(rev_bits):
            result.append(v)
            if (i+1) == 7:
                result.append("0")
            elif (i+1) % 7 == 0:
                result.append("1")
        binary_str = "".join(result)[::-1]
        hex_result = [hex(int(binary_str[i:i + 4],2))[2:] for i in range(0, len(binary_str), 4)]
        return "".join(hex_result)
