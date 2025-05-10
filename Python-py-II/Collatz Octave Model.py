import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# =====================
# CORE FUNCTIONS
# =====================
def generate_collatz_sequence(n):
    sequence = [n]
    while n != 1:
        n = 3 * n + 1 if n % 2 else n // 2
        sequence.append(n)
    return sequence

def reduce_to_single_digit(value):
    return (value - 1) % 9 + 1

def map_to_octave(value, layer):
    """Converts number to 3D octave coordinates"""
    reduced = reduce_to_single_digit(value)
    angle = (reduced / 9) * 2 * np.pi
    radius = 1.23489 * (layer + 1)  # LZ scaling
    return (
        radius * np.cos(angle),
        radius * np.sin(angle),
        layer
    )

def compute_energy(sequence):
    return len(sequence) * np.log(len(sequence) + 0.5772)

def compute_eta(sequence):
    if len(sequence) < 2:
        return 0.0
    oscillations = sum(1 if x > y else -1 for x, y in zip(sequence[1:], sequence))
    return np.arctan(oscillations) / (2 * np.pi)

# ======================
# ANALYSIS & VISUALIZATION
# ======================
# Generate Collatz data for numbers 1-100
collatz_data = {n: generate_collatz_sequence(n) for n in range(1, 101)}

# Create 3D plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot all sequences
for number, sequence in collatz_data.items():
    positions = [map_to_octave(v, i) for i, v in enumerate(sequence)]
    x, y, z = zip(*positions)
    ax.plot(x, y, z, alpha=0.6, linewidth=1)
    ax.scatter(x, y, z, s=10)

ax.set_title("Collatz Octave Model")
ax.set_xlabel("X (Harmonic Phase)")
ax.set_ylabel("Y (Radial Scaling)")
ax.set_zlabel("Z (Recursion Depth)")

plt.show()