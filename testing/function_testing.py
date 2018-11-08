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


def omega(n):
    return cmath.exp(-2 * math.pi * 1j / n)


def sample(amp, freq, x):
        return amp * math.sin(freq*x)


def generate_test_matrix(frequency, samplerate, length):
    """Given a frequency(Hz), samplerate(Hz) and length(s) returns a matrix of the samples that
    sound would create, if it were a recording of sound from a wave file."""
    amplitude = 10

    sample_list = []
    for x in range(length):
        for y in range(samplerate):
            # x is the integer part of time, y is the decimal
            t = x + (y/samplerate)

            samp = sample(amplitude, frequency, t)
            sample_list.append([samp])

    return sample_list


sampleList = generate_test_matrix(220, 44100, 2)
sampleVector = class_testing.Matrix(sampleList)
N = sampleVector.get_dim()[0]

omega_N = omega(N)
print(N, omega_N)

DFT_list = []

omega_dict = {}
with open('omega_dict.pickle', 'wb') as handle:
    pickle.dump(omega_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

for i in range(N**2):
    with open('omega_dict.pickle', 'rb') as handle:
        omega_dict = pickle.load(handle)
    omega_dict[i] = omega_N ** i
    with open('omega_dict.pickle', 'wb') as handle:
        pickle.dump(omega_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    a = len(omega_dict)
    if a % 100 == 0:
        print(a)

# for y in range(N):
#     row = []
#     if y%100==0:
#         print(y)
#     for x in range(N):
#         pw = x*y
#         row.append(omega_dict[pw])
#     DFT_list.append(row)
#
# print(DFT_list[0][0], DFT_list[0][1], DFT_list[1][0], DFT_list[1][1])
