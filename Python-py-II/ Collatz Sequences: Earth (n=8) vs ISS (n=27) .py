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

# Function to map reduced values to an octave structure (with relativistic scaling)
def map_to_octave(value, layer, scale_factor=0.2):
    angle = (value / 9) * 2 * np.pi  # Mapping to a circular octave
    radius = 1.0 + layer * scale_factor  # Dynamic radius for time dilation analogy
    x = np.cos(angle) * radius
    y = np.sin(angle) * radius
    return x, y

# Generate sequences for Earth (n=8) and ISS (n=27)
earth_seq = generate_collatz_sequence(8)    # 8 → 4 → 2 → 1 (3 steps)
iss_seq = generate_collatz_sequence(27)    # 27 → 82 → ... → 1 (111 steps)

# Plot 3D trajectories
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

for n, seq in [(8, earth_seq), (27, iss_seq)]:
    x, y, z = [], [], []
    for layer, val in enumerate(seq):
        reduced_val = reduce_to_single_digit(val)
        xi, yi = map_to_octave(reduced_val, layer)
        x.append(xi); y.append(yi); z.append(layer)
    ax.plot(x, y, z, label=f"n={n} (steps={len(seq)})", marker='o')

ax.set_title("Collatz Sequences: Earth (n=8) vs ISS (n=27)")
ax.set_xlabel("X (Phase Space)")
ax.set_ylabel("Y (Phase Space)")
ax.set_zlabel("Time (Layer = Step)")
plt.legend()
plt.show()