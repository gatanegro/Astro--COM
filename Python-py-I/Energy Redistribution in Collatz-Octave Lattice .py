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

# Parameters
LZ = 1.23498
HQS = 0.235
layers = 5
nodes = 20
energy = np.zeros((layers, nodes))
energy[-1, :] = 100  # Initialize outer layer (L5)
def collatz_step(phi_current, is_even):
    if is_even:
        phi_next = (phi_current / 2) * LZ * (1 - HQS)
    else:
        phi_next = (3 * phi_current + 1) * LZ * (1 - HQS)
    return phi_next

for layer in range(layers-2, -1, -1):  # From L5 to L1
    for node in range(nodes):
        current_phi = energy[layer+1, node]
        is_even = (node % 2 == 0)
        energy[layer, node] = collatz_step(current_phi, is_even)
        # Redistribute HQS to neighbors
        energy[layer, (node+1) % nodes] += current_phi * HQS / 2
        energy[layer, (node-1) % nodes] += current_phi * HQS / 2
plt.imshow(energy, cmap='viridis', aspect='auto')
plt.colorbar(label='Energy (Φ)')
plt.xlabel('Collatz Nodes')
plt.ylabel('Octave Layers')
plt.title('Energy Redistribution in Collatz-Octave Lattice')
plt.show()