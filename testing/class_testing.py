class Matrix(object=[]):
    """A  n*m matrix class, can be constructed from a list of objects or a 2d list of objects"""
    pass


class SquareMatrix(Matrix):
    """A n*n matrix class, a special instance of a Matrix that is square"""
    pass


class Midi:
    """A representation of a midi file,
     can be written to an actual file through use of .write(filename)"""
    pass


class Wave(object):
    """A representation of a Wave file, must be created from a _io.BufferedReader of a wave file """
    pass


class Fourier:
    """Performs a fourier transform on one Matrix of time domain values and returns a Matrix of
    frequency domain values"""
    pass
