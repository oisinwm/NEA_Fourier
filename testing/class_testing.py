class Matrix:
    """A  n*m matrix class, can be constructed from a list of objects or a 2d list of objects"""
    def __init__(self, *args, **kwargs):
        self._contents = []
        self._dimensions = [0,0]
        if len(args) == 0 and len(kwargs) == 2:
            # construct from a and b, if validation passes
            if kwargs.keys() == {"m":0, "n":0}.keys():
                if isinstance(kwargs["m"], int) and isinstance(kwargs["n"], int):
                    if kwargs["m"] > 0 and kwargs["n"] > 0:
                        # construct
                        self._dimensions = [kwargs["m"], kwargs["n"]]
                        self._contents = [[0 for x in kwargs["n"]] for y in kwargs["m"]]
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
                        print(args, args[0], type(args), type(args[0]))
                        for x in range(len(args[0])):
                            if not isinstance(args[x], list):
                                raise TypeError(""""Invalid values for Matrix, 
                                                must be only of type list""")
                            for y in args[0][x]:
                                if isinstance(y, list):
                                    raise TypeError(""""Invalid values for 2D Matrix, 
                                                    must not be list""")
                                if len(args[0][x]) != n:
                                    raise TypeError(""""Invalid values for 2D Matrix, 
                                                        must not of equal width""")
                        # At this point, valid list of lists
                        self._contents = args[0]
                        self._dimensions = [len(args[0]), len(args[0][0])]

                    else:
                        for x in args[0]:
                            if isinstance(x, list):
                                raise TypeError(""""Invalid values for Matrix, 
                                                must be only of type: list""")
                        self._dimensions = [1, len(args[0])]
                        self._contents = list(args[0])
            else:
                raise TypeError(f"""Invalid type for Matrix: {type(args[0])}, must be list""")

        else:
            # don't construct, incorrect information given
            raise TypeError(f"""Invalid input length for Matrix: {type(args)}, 
                                must be exactly 1 list""")


class SquareMatrix(Matrix):
    """A n*n matrix class, a special instance of a Matrix that is square"""
    pass


class Midi:
    """A representation of a midi file,
     can be written to an actual file through use of .write(filename)"""
    pass


class Wave:
    """A representation of a Wave file, must be created from a _io.BufferedReader of a wave file """
    pass


class Fourier:
    """Performs a fourier transform on one Matrix of time domain values and returns a Matrix of
    frequency domain values"""
    pass


if __name__ == "__main__":
    a = Matrix([[1, 2], [3, 4]])