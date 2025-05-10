import matplotlib.pyplot as plt
import numpy as np

# Function to generate Collatz sequence
def collatz_sequence(n):
    sequence = [n]
    while n != 1:
        if n % 2 == 0:
            n = n // 2
        else:
            n = 3 * n + 1
        sequence.append(n)
    return sequence

# Map numbers to "frequencies" (logarithmic scale to simulate energy levels)
def map_to_frequencies(sequence):
    return [np.log2(num + 1) for num in sequence]

# Generate sequences and frequencies for visualization
start_values = range(2, 20)  # Starting numbers for Collatz sequences
sequences = [collatz_sequence(n) for n in start_values]
frequencies = [map_to_frequencies(seq) for seq in sequences]

# Plot frequencies as oscillatory patterns
plt.figure(figsize=(12, 8))
for i, freq in enumerate(frequencies):
    plt.plot(freq, label=f"Start: {start_values[i]}", alpha=0.7)

plt.title("Collatz Sequences as Oscillatory Frequencies in the FIELD", fontsize=16)
plt.xlabel("Iteration Steps", fontsize=14)
plt.ylabel("Frequency (Log2 Scale)", fontsize=14)
plt.grid(alpha=0.3)
plt.legend(loc="upper right", fontsize=10, ncol=2)
plt.show()