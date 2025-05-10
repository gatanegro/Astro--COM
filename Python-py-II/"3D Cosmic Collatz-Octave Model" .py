import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# =====================
# CORE COLLATZ FUNCTIONS
# =====================
def generate_collatz(n):
    """Generate sequence with 1-centered cosmology"""
    seq = [n]
    while n != 1:
        n = 3*n + 1 if n%2 else n//2
        seq.append(n)
    return seq

def octave_reduction(value):
    """Map numbers to 1-9 without zeros"""
    return (value - 1) % 9 + 1

def map_to_cosmos(value, layer, LZ=1.23489):
    """3D spiral mapping with 1 at origin"""
    if value == 1:
        return (0, 0, 0)  # Cosmic center
    
    reduced = octave_reduction(value)
    angle = (reduced / 9) * 2 * np.pi  # 40° per step
    radius = LZ * layer
    return (radius * np.cos(angle), radius * np.sin(angle), layer)

# ======================
# PARAMETERS & ANALYSIS
# =====================
NUMBERS_TO_PLOT = range(1, 21)  # 1-20
MAX_LAYERS = 50  # Maximum recursion depth
LZ_SCALE = 1.23489  # Cosmic expansion factor

# Generate cosmic paths
cosmic_paths = {}
for n in NUMBERS_TO_PLOT:
    seq = generate_collatz(n)
    path = [map_to_cosmos(v, i, LZ_SCALE) for i,v in enumerate(seq[:MAX_LAYERS])]
    cosmic_paths[n] = path

# ==============
# 3D VISUALIZATION
# ==============
fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(111, projection='3d')

# Plot celestial paths
for n, path in cosmic_paths.items():
    x, y, z = zip(*path)
    
    # Color by energy signature
    energy = len(path) * np.log(len(path)+0.5772)
    color = plt.cm.plasma(energy/150)  # Normalized to 150=max energy
    
    ax.plot(x, y, z, color=color, alpha=0.7)
    ax.scatter(x, y, z, s=20, color=color)

# Plot cosmic center (1)
ax.scatter(0, 0, 0, s=500, c='gold', marker='*', edgecolors='black')

# Add coordinate guides
ax.plot([0, LZ_SCALE*MAX_LAYERS], [0,0], [0,0], c='gray', ls='--', alpha=0.3)
ax.plot([0,0], [0, LZ_SCALE*MAX_LAYERS], [0,0], c='gray', ls='--', alpha=0.3)
ax.plot([0,0], [0,0], [0, MAX_LAYERS], c='gray', ls='--', alpha=0.3)

ax.set_title("3D Cosmic Collatz-Octave Model", fontsize=14)
ax.set_xlabel("X (Prime Harmony Axis)")
ax.set_ylabel("Y (Fibonacci Phase Axis)")
ax.set_zlabel("Z (Recursion Depth)")
plt.show()