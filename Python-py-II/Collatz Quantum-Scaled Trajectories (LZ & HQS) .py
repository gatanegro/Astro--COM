import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Constants
LZ = 1.23498  # Scalable amplitude
HQS = 0.235   # Fixed phase curvature
MAX_STEPS = 1000  # Quantum-scale recursion limit

def collatz(n):
    sequence = [n]
    for _ in range(MAX_STEPS):
        if n == 1:
            break
        n = n // 2 if n % 2 == 0 else 3 * n + 1
        sequence.append(n)
    return sequence

def reduce_to_octave(value):
    return (value - 1) % 9 + 1  # Quantum energy quantization

def map_to_3d(value, layer):
    # Scale radius by LZ^(-layer) to enforce quantum downsampling
    angle = (value / 9) * 2 * np.pi * HQS  # HQS fixes curvature
    radius = (layer + 1) * LZ ** (-layer / 10)  # Exponential decay
    x = np.cos(angle) * radius
    y = np.sin(angle) * radius
    z = layer
    return x, y, z

# Generate sequences for n=1 to 50 (including large seeds)
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

for n in range(1, 51):
    seq = collatz(n)
    x, y, z = [], [], []
    for layer, value in enumerate(seq):
        reduced = reduce_to_octave(value)
        xi, yi, zi = map_to_3d(reduced, layer)
        x.append(xi), y.append(yi), z.append(zi)
    ax.plot(x, y, z, linewidth=0.5, label=f'n={n}')

ax.set_title("Collatz Quantum-Scaled Trajectories (LZ & HQS)")
ax.set_xlabel("X (LZ-Scaled Amplitude)")
ax.set_ylabel("Y (HQS-Phase)")
ax.set_zlabel("Quantum Layer (Z)")
plt.legend(loc='upper right', fontsize=6)
plt.show()
