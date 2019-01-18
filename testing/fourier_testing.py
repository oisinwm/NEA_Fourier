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


def generate_test_matrix():
    return [[0],[1],[0],[0],[0],[0],[0],[0]]


sampleList = generate_test_matrix()
sampleVector = class_testing.Matrix(sampleList)
N = sampleVector.get_dim()[0]

# Plots teh function over time domain to verify it is correct
matplotlib.pyplot.plot(sampleList)
matplotlib.pyplot.show()

omega_N = omega(N)
print(N, omega_N)
DFT_list = []

time_start = time.process_time()
for x in range(N):
    factor = []
    for y in range(N):
        factor.append(omega_N ** (x*y))
    factorVector = class_testing.Matrix(factor)
    factorVector = (1 / N) * factorVector
    answer = factorVector * sampleVector
    DFT_list.append(answer[0][0])

# I'm still not sure exactly how to interpret the output, it's very information dense
# These are the most meaningful metrics that can be extracted
magnitude_list = [cmath.polar(i)[0] for i in DFT_list]
angle_list = [cmath.polar(i)[1] for i in DFT_list]
real_list = [i.real for i in DFT_list]
imag_list = [i.imag for i in DFT_list]

time_end = time.process_time()
print(time_end - time_start)

#with open("combined_out.pickle", "wb") as file:
#    pickle.dump(DFT_list, file, protocol=pickle.HIGHEST_PROTOCOL)

#with open("out.pickle", "rb") as file:
#    stuff = pickle.load(file)

matplotlib.pyplot.plot(magnitude_list)
matplotlib.pyplot.show()
matplotlib.pyplot.plot(angle_list)
matplotlib.pyplot.show()
matplotlib.pyplot.plot(real_list)
matplotlib.pyplot.show()
matplotlib.pyplot.plot(imag_list)
matplotlib.pyplot.show()

print("complete")
print(DFT_list)
