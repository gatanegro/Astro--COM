import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Constants
LZ = 1.23498  # Scalable amplitude
HQS = 0.235   # Fixed curvature
MAX_LAYERS = 100  # Quantum recursion limit

def collatz(n):
    sequence = [n]
    while n != 1 and len(sequence) < MAX_LAYERS:
        n = n // 2 if n % 2 == 0 else 3 * n + 1
        sequence.append(n)
    return sequence

def reduce_to_octave(value):
    return (value - 1) % 9 + 1  # Energy quantization (1-9)

def map_to_3d(value, layer):
    phase = (value / 9) * 2 * np.pi * HQS  # HQS phase alignment
    radius = LZ ** (-layer / 10)  # Quantum amplitude decay
    x = np.cos(phase) * radius
    y = np.sin(phase) * radius
    z = layer
    return x, y, z

# Plot trajectories for n=1-50
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

for n in range(1, 51):
    seq = collatz(n)
    x, y, z = [], [], []
    for layer, value in enumerate(seq):
        octave_value = reduce_to_octave(value)
        xi, yi, zi = map_to_3d(octave_value, layer)
        x.append(xi), y.append(yi), z.append(zi)
    ax.plot(x, y, z, linewidth=0.5, label=f'n={n}')

ax.set_title("Collatz Quantum-Cosmic Trajectories (LZ & HQS)")
ax.set_xlabel("X (LZ-Scaled Amplitude)")
ax.set_ylabel("Y (HQS-Phase)")
ax.set_zlabel("Quantum Layer (Z)")
plt.legend(fontsize=6)
plt.savefig('collatz_trajectories.png', dpi=300)
plt.show()