import numpy as np
import matplotlib.pyplot as plt

# Constants
LZ = 1.23498  # Scalable amplitude
HQS = 0.235   # Fixed curvature
MAX_LAYERS = 100  # Quantum recursion limit

def collatz(n):
    """Generates Collatz sequence for n."""
    sequence = [n]
    while n != 1 and len(sequence) < MAX_LAYERS:
        n = n // 2 if n % 2 == 0 else 3 * n + 1
        sequence.append(n)
    return sequence

def reduce_to_octave(value):
    """Maps value to 9-cycle octave structure."""
    return (value - 1) % 9 + 1

def map_to_3d(value, layer):
    """Projects octave value to 3D coordinates."""
    phase = (value / 9) * 2 * np.pi * HQS
    radius = LZ ** (-layer / 10)
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
        x.append(xi)
        y.append(yi)
        z.append(zi)
    
    # Color by n for distinction, no legend for clarity
    ax.plot(x, y, z, linewidth=0.5, c=plt.cm.viridis(n / 50))

ax.set_title("Collatz Quantum-Cosmic Trajectories (LZ & HQS)")
ax.set_xlabel("X (LZ-Scaled Amplitude)")
ax.set_ylabel("Y (HQS-Phase)")
ax.set_zlabel("Quantum Layer (Z)")
plt.savefig('collatz_trajectories.png', dpi=150)
plt.show()