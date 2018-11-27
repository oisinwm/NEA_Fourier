# Got initial matrix of samples, 1*no_of_samples
# Need to loop through he sample matrix, taking slices through time
# For each slice, perform a fourier transform to produce likely frequencies
# Turn likely frequencies into likely notes
# When likelyhood of note is above threshold, set the note to active at that time
# Transcribe note activity to midi file
# Done
#
#
# Now add volume of note at specific time (may have to be relative as opposed to absolute)
import math, cmath, pickle
import class_testing
import matplotlib.pyplot
import time
import pickle


def omega(n):
    return cmath.exp(-2 * math.pi * 1j / n)


def sample(amp, freq, x):
        return amp * math.sin(freq*x)


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


# This is bad
sampleList = generate_test_matrix(10, 44100, 1)
sampleVector = class_testing.Matrix(sampleList)
N = sampleVector.get_dim()[0]

matplotlib.pyplot.plot(sampleList)
matplotlib.pyplot.show()

omega_N = omega(N)
print(N, omega_N)

DFT_list = []

time_start = time.clock()
for x in range(N):
    factor = []
    if x % 100 == 0:
        print(x)
    for y in range(N):
        factor.append(omega_N ** (x*y))
    factorVector = class_testing.Matrix(factor)
    answer = factorVector * sampleVector
    DFT_list.append(answer[0][0])

magnitude_list = [cmath.polar(i)[0] for i in DFT_list]
time_end = time.clock()
print(time_end - time_start)

with open("out.pickle", "wb") as file:
    pickle.dump(magnitude_list, file, protocol=pickle.HIGHEST_PROTOCOL)

with open("out.pickle", "rb") as file:
    stuff = pickle.load(file)

matplotlib.pyplot.plot(stuff)
matplotlib.pyplot.show()

print("complete")
