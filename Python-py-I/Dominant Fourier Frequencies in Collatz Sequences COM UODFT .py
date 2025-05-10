import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

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

# Function to apply Fourier Transform and analyze dominant frequencies
def apply_fourier_transform(sequence, number):
    sequence_array = np.array(sequence)
    fft_result = np.fft.fft(sequence_array)
    frequencies = np.fft.fftfreq(len(sequence_array))
    magnitudes = np.abs(fft_result)
    
    # Find peaks in the Fourier spectrum
    peaks, _ = find_peaks(magnitudes, height=0.1 * max(magnitudes))
    peak_frequencies = frequencies[peaks]
    peak_magnitudes = magnitudes[peaks]
    
    return peak_frequencies, peak_magnitudes

# Apply Fourier Transform to multiple numbers
numbers_to_analyze = range(1, 21)
scaling_factor = 1.23498  # Known recursive scaling factor

plt.figure(figsize=(12, 6))
for num in numbers_to_analyze:
    collatz_sequence = generate_collatz_sequence(num)
    peak_freqs, peak_mags = apply_fourier_transform(collatz_sequence, num)
    
    # Plot peaks
    plt.scatter(peak_freqs, peak_mags, label=f"{num}", alpha=0.6)
    
    # Check for presence of Ψ∞ in frequency peaks
    if any(np.isclose(peak_freqs, scaling_factor, atol=0.05)):
        print(f"Ψ∞ ≈ {scaling_factor} found in Collatz {num}")

plt.axvline(x=scaling_factor, color='r', linestyle='--', label=f"Ψ∞ = {scaling_factor}")
plt.title("Dominant Fourier Frequencies in Collatz Sequences")
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.legend(loc='upper right', fontsize='small', ncol=2)
plt.grid()
plt.show()