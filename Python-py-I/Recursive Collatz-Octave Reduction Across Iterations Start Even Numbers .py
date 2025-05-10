import numpy as np
import matplotlib.pyplot as plt

# Function to compute the Collatz-Octave reduction (digit sum recurrence)
def collatz_octave_reduction(n):
    """Recursively reduces a number by summing its digits until it reaches a single-digit value."""
    while n >= 10:
        n = sum(int(digit) for digit in str(n))
    return n

# Function to generate a sequence showing how numbers behave under recursive reduction
def generate_collatz_octave_sequence(start, iterations):
    """Generates a sequence based on recursive Collatz-Octave digit sum reduction."""
    sequence = [start]
    for _ in range(iterations):
        next_value = collatz_octave_reduction(sequence[-1] * 3 + 1 if sequence[-1] % 2 else sequence[-1] // 2)
        sequence.append(next_value)
    return sequence

# Generate sequences for different starting numbers
start_values = [2, 4, 6, 8,]
num_iterations = 50

plt.figure(figsize=(12, 6))
for start in start_values:
    seq = generate_collatz_octave_sequence(start, num_iterations)
    plt.plot(seq, label=f"Start {start}", marker='o', linestyle='--')

plt.xlabel("Iteration")
plt.ylabel("Recursive Collatz-Octave Value")
plt.title("Recursive Collatz-Octave Reduction Across Iterations")
plt.legend()
plt.grid(True)
plt.show()