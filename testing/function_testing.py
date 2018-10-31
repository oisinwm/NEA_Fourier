def little_bin(rawbytes):
    """Returns the integer representation of an unsigned 32 bit integer,
        stored as bytes in little endian"""
    bytez = []
    for i in rawbytes:
        bytez.append(hex(i)[2:].zfill(2))
    hexstr = "".join(bytez[::-1])
    # at this point need a string of raw hex digits only
    print(hexstr)
    result = ""
    for x in hexstr:
        digits = bin(int(x, 16))[2:].zfill(4)
        result += digits

    return result


a = little_bin(b"\x12\x00\x00\x00")
print(a)
print(int(a, 2))