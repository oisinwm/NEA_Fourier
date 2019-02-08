import math
import cmath


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
        self.matrix_types = (Matrix, SquareMatrix)
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
                        if all(isinstance(x, self.matrix_types) for x in args[0]):
                            self._contents = self._contents[0]

            elif isinstance(args[0], self.matrix_types):
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
        else:
            raise KeyError

    def __setitem__(self, key, value):
        # print(self._contents, key)
        if isinstance(key, int):
            if self._dimensions[1] >= key:
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

        elif isinstance(other, self.matrix_types):
            # Matrix multiplication should be handled by mul not rmul,
            #  if being found here then an error has occurred
            raise NotImplementedError("Matrix multiplication should be handled by mul")
        
        return result_matrix

    def __mul__(self, other):
        if isinstance(other, Identity):
            return Matrix(self)
        elif isinstance(other, self.matrix_types):
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

        elif isinstance(other, self.number_types):
            # Scalar multiplication should be handled by rmul not mul,
            # if here then an error has occurred
            raise NotImplementedError("Scalar multiplication should be handled by rmul")

        return result_matrix

    def __add__(self, other):
        if isinstance(other, self.matrix_types):
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
        if isinstance(other, self.matrix_types):
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
        for frame in range(framesNum - 1):
            start = self.frameStartIndex + frame * self.frameSize
            end = self.frameStartIndex + (frame + 1) * self.frameSize
            data = contents[start:end]
            for x in range(self.channels):
                s = x * self.bitDepth // 8
                e = (x + 1) * self.bitDepth // 8
                channelData = data[s:e]
                a = self.little_bin(channelData)
                b = self.signed_int(a)
                self.frameDataLists[x].append([b])

        # Prepare lists for FFT
        self.dataMatrices = []
        y = len(self.frameDataLists[0])
        x = int(2 ** math.ceil(math.log(y, 2))) - y

        for sampleList in self.frameDataLists:
            for i in range(x):
                sampleList.append([0])
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

    def get_data(self):
        return self.dataMatrices


class SquareMatrix(Matrix):
    """A n*n matrix class, a special instance of a Matrix that is square"""

    def __init__(self):
        Matrix.__init__(self)


class Midi:
    """A representation of a midi file,
     can be written to an actual file through use of .write(filename)"""
    pass


class Identity(Matrix):
    def __init__(self, x):
        Matrix.__init__(self, m=x, n=x)
        for i in range(x):
            self[i][i] = 1
    
    
class Fourier(Matrix):
    """Performs a fourier transform on one Matrix of time domain values and returns a Matrix of
    frequency domain values"""

    def __init__(self, matrix):
        Matrix.__init__(self, matrix)
        self._p = math.log(matrix.get_dim()[0], 2) - 1
        self._omega_N = cmath.exp(-2 * math.pi * 1j / max(self._dimensions))
        self.layer = self.get_dim()[0]

    def decompose(self):
        vector = self
        # This function wraps the vector in an unneeded pair of brackets
        if math.log(vector.get_dim()[0], 2) > 1:
            even, odd = vector._contents[::2], vector._contents[1::2]
            even, odd = Matrix(even), Matrix(odd)
            #print(even, odd)
            even, odd = Fourier.decompose(even), Fourier.decompose(odd)
            return Matrix([even, odd])
        else:
            return vector
        
    def get_p(self):
        return self._p
        
    def get_omega(self):
        return self._omega_N
    
    @staticmethod
    def from_combine(mat, test):
        temp = Fourier(Matrix(m=mat.get_dim()[0], n=mat.get_dim()[1]))
        temp._contents = mat._contents
        temp._p = test._p
        temp._omega_N = test._omega_N
        
        return temp
        
        
        

if __name__ == "__main__":
#    a = Wave("24nocturnea.wav")
#    print(a.get_data()[0].get_dim(), a.get_data()[1].get_dim())
#    b = Fourier(a.get_data()[0])
#    print(b._omega_N)
#    print(a.get_data()[0].get_dim())
#    final = b.decompose()
    
    test = Matrix([[i] for i in range(1, 33)])
    test = Fourier(test)
    
    A = test.decompose()
    A = Fourier.from_combine(A, test)
    print(A.get_p())
    
    #transform = Fourier.make_transform(A.get_omega(), A.get_p())
    D = Identity(int(2**(A.get_p()-1)))
    
    for i in range(int(2**(A.get_p()-1))):
        D[i][i] = A.get_omega() ** i
        
    a = Matrix([[Identity(2), -1 * D],[Identity(2), -1 * D]])
    