import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Function to generate Collatz sequence for a number
def generate_collatz_sequence(n):
    sequence = [n]
    while n != 1:
        if n % 2 == 0:
            n = n // 2
        else:
            n = 3 * n + 1
        sequence.append(n)
    return sequence

# Function to reduce numbers to a single-digit using modulo 9
def reduce_to_single_digit(value):
    return (value - 1) % 9 + 1

# Function to apply Fourier Transform and visualize the frequency spectrum
def apply_fourier_transform(sequence, number):
    sequence_array = np.array(sequence)
    fft_result = np.fft.fft(sequence_array)
    frequencies = np.fft.fftfreq(len(sequence_array))
    
    return frequencies, np.abs(fft_result)

# Apply Fourier Transform to multiple numbers
numbers_to_analyze = range(1, 21)  # Change this range for more numbers
fft_data = {}
for num in numbers_to_analyze:
    collatz_sequence = generate_collatz_sequence(num)
    fft_data[num] = apply_fourier_transform(collatz_sequence, num)

# Plot all Fourier Transforms together
plt.figure(figsize=(12, 6))
for num, (freqs, magnitudes) in fft_data.items():
    plt.plot(freqs, magnitudes, linestyle='-', alpha=0.7, label=f"{num}")

plt.title("Fourier Transform of Collatz Sequences (1-20)")
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.legend(loc='upper right', fontsize='small', ncol=2)
plt.grid()
plt.show()
