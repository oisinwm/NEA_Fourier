def little_endian(rawbytes):
    digits = "0123456789abcdef"
    binaryDigits = []
    for i in str(rawbytes):
        if str(i) in digits:
            digi = bin(digits.index(str(i)))
            print(digi)
            binaryDigits.append(digi)
    return "".join(binaryDigits)[::-1]


a = little_endian(b"\x12\x00\x00\x00")
print(a)
print(int(a, 2))
