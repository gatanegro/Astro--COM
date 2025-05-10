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

# Function to reduce numbers to a single-digit using modulo 9 (octave reduction)
def reduce_to_single_digit(value):
    return (value - 1) % 9 + 1

# Function to map reduced values to an octave structure
def map_to_octave(value, layer):
    angle = (value / 9) * 2 * np.pi  # Mapping to a circular octave
    x = np.cos(angle) * (layer + 1)
    y = np.sin(angle) * (layer + 1)
    return x, y

# Generate Collatz sequences for numbers 1 to 20
collatz_data = {n: generate_collatz_sequence(n) for n in range(1, 21)}

# Map sequences to the octave model with reduction
octave_positions = {}
num_layers = max(len(seq) for seq in collatz_data.values())
stack_spacing = 1.0  # Space between layers

for number, sequence in collatz_data.items():
    mapped_positions = []
    for layer, value in enumerate(sequence):
        reduced_value = reduce_to_single_digit(value)
        x, y = map_to_octave(reduced_value, layer)
        z = layer * stack_spacing  # Layer height in 3D
        mapped_positions.append((x, y, z))
    octave_positions[number] = mapped_positions

# Plot the 3D visualization
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot each Collatz sequence as a curve
for number, positions in octave_positions.items():
    x_vals = [pos[0] for pos in positions]
    y_vals = [pos[1] for pos in positions]
    z_vals = [pos[2] for pos in positions]
    ax.plot(x_vals, y_vals, z_vals, label=f"Collatz {number}")
    ax.scatter(x_vals, y_vals, z_vals, s=20, zorder=5)  # Points for clarity

# Add labels and adjust the view
ax.set_title("3D Collatz Sequences in Octave Model")
ax.set_xlabel("X (Horizontal Oscillation)")
ax.set_ylabel("Y (Vertical Oscillation)")
ax.set_zlabel("Z (Octave Layer)")
plt.legend(loc='upper right', fontsize='small')
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
start_values = [3, 6, 9,]
num_iterations = 50

plt.figure(figsize=(12, 6))
for start in start_values:
    seq = generate_collatz_octave_sequence(start, num_iterations)
    plt.plot(seq, label=f"Start {start}", marker='o', linestyle='--')

plt.xlabel("Iteration")
plt.ylabel("Recursive Collatz-Octave Value")
plt.title("Recursive Collatz-Octave Reduction Across Iterations 3 6 9")
plt.legend()
plt.grid(True)
plt.show()