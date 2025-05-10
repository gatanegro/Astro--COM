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
def apply_fourier_transform(sequence):
    sequence_array = np.array(sequence)
    fft_result = np.fft.fft(sequence_array)
    frequencies = np.fft.fftfreq(len(sequence_array))
    
    # Plot the FFT result
    plt.figure(figsize=(10, 5))
    plt.plot(frequencies, np.abs(fft_result), marker='o', linestyle='-')
    plt.title("Fourier Transform of Collatz Sequence")
    plt.xlabel("Frequency")
    plt.ylabel("Magnitude")
    plt.grid()
    plt.show()

# Test the Fourier Transform on Collatz sequence of a chosen number
number_to_analyze = 12
collatz_sequence = generate_collatz_sequence(number_to_analyze)
apply_fourier_transform(collatz_sequence)