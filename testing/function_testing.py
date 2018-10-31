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


def signed_int(rawbytes):
    """Returns the integer representation of a signed integer,
        from binary"""
    frameSize = 16
    if frameSize == 8:
        # Data is unsigned 8 bit integer (-128 to 127)
        return int(rawbytes, 2)
    if frameSize == 16:
        # Data is signed 16 bit integer (-32768 to 32768)
        return -32768 + int(rawbytes[1:], 2)
    if frameSize == 32:
        # Data is a float (-1.0f ro 1f)
        raise NotImplementedError("Cannot read 32 bit wave file")


a = little_bin(b'\xff\xff')
print(a)
print(signed_int(a))
