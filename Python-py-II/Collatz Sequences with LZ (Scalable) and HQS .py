import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Collatz sequence generator (original steps)
def generate_collatz_sequence(n):
    sequence = [n]
    while n != 1:
        n = n // 2 if n % 2 == 0 else 3 * n + 1
        sequence.append(n)
    return sequence

# Reduce to octave (modulo 9)
def reduce_to_octave(value):
    return (value - 1) % 9 + 1

# Map to 3D coordinates (LZ scales radius, HQS fixed)
def map_to_3d(value, layer, LZ=1.23498, HQS=0.235):
    angle = (value / 9) * 2 * np.pi
    radius = (layer + 1) * LZ  # LZ scales radial distance
    x = np.cos(angle) * radius * HQS  # HQS fixed curvature coupling
    y = np.sin(angle) * radius * HQS
    z = layer
    return x, y, z

# Generate and plot sequences
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

for n in range(1, 21):
    seq = generate_collatz_sequence(n)
    x, y, z = [], [], []
    for layer, value in enumerate(seq):
        reduced = reduce_to_octave(value)
        xi, yi, zi = map_to_3d(reduced, layer)
        x.append(xi), y.append(yi), z.append(zi)
    ax.plot(x, y, z, label=f"n={n}")

ax.set_title("Collatz Sequences with LZ (Scalable) and HQS (Fixed)")
ax.set_xlabel("X (LZ * HQS Scaled)")
ax.set_ylabel("Y (Phase Angle)")
ax.set_zlabel("Layer (Z)")
plt.legend()
plt.show()