import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Function to generate Recamán's sequence
def generate_recaman_sequence(n_terms):
    seen = set()
    sequence = [0]
    for n in range(1, n_terms):
        prev = sequence[-1]
        candidate = prev - n
        if candidate > 0 and candidate not in seen:
            sequence.append(candidate)
        else:
            sequence.append(prev + n)
        seen.add(sequence[-1])
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

# Parameters
n_terms = 100  # Number of Recamán sequence terms to generate
recaman_sequence = generate_recaman_sequence(n_terms)

# Map sequence to the octave model with reduction
mapped_positions = []
stack_spacing = 1.0  # Space between layers

for layer, value in enumerate(recaman_sequence):
    reduced_value = reduce_to_single_digit(value)
    x, y = map_to_octave(reduced_value, layer)
    z = layer * stack_spacing  # Layer height in 3D
    mapped_positions.append((x, y, z))

# Plot the 3D visualization
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

x_vals = [pos[0] for pos in mapped_positions]
y_vals = [pos[1] for pos in mapped_positions]
z_vals = [pos[2] for pos in mapped_positions]

ax.plot(x_vals, y_vals, z_vals, label="Recamán Sequence", color='crimson')
ax.scatter(x_vals, y_vals, z_vals, s=30, zorder=5, color='navy')

# Add labels and adjust the view
ax.set_title("3D Recamán Sequence in Octave Model")
ax.set_xlabel("X (Horizontal Oscillation)")
ax.set_ylabel("Y (Vertical Oscillation)")
ax.set_zlabel("Z (Octave Layer)")
ax.legend(loc='upper right', fontsize='large')

plt.show()
