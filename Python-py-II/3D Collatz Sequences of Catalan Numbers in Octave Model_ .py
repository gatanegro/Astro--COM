import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_collatz_sequence(n):
    sequence = [n]
    while n != 1:
        n = n // 2 if n % 2 == 0 else 3 * n + 1
        sequence.append(n)
    return sequence

def reduce_to_single_digit(value):
    return (value - 1) % 9 + 1

def map_to_octave(value, layer):
    angle = (value / 9) * 2 * np.pi
    x = np.cos(angle) * (layer + 1)
    y = np.sin(angle) * (layer + 1)
    return x, y

def catalan(n):
    if n <= 1:
        return 1
    return ((4*n-2) * catalan(n-1)) // (n+1)

# Generate the first 10 Catalan numbers
catalan_numbers = [catalan(i) for i in range(10)]

# Generate Collatz sequences for Catalan numbers
collatz_data = {n: generate_collatz_sequence(n) for n in catalan_numbers}

# Map sequences to the octave model
octave_positions = {}
stack_spacing = 0.5

for number, sequence in collatz_data.items():
    mapped_positions = []
    for layer, value in enumerate(sequence):
        reduced_value = reduce_to_single_digit(value)
        x, y = map_to_octave(reduced_value, layer)
        z = layer * stack_spacing
        mapped_positions.append((x, y, z))
    octave_positions[number] = mapped_positions

# Plot the 3D visualization
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot each Collatz sequence as a curve
for i, (number, positions) in enumerate(octave_positions.items()):
    x_vals, y_vals, z_vals = zip(*positions)
    ax.plot(x_vals, y_vals, z_vals, label=f"C({i})={number}")
    ax.scatter(x_vals[0], y_vals[0], z_vals[0], s=50, c='red', zorder=5)  # Starting point

ax.set_title("3D Collatz Sequences of Catalan Numbers in Octave Model")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z (Sequence Step)")
plt.legend(loc='upper left', fontsize='x-small')

plt.show()
