import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Core COM parameters
LZ = 1.23498  # Attractor constant
HQS = 0.618  # Holomorphic Quantum Symmetry (golden ratio)

def generate_collatz_sequence(n):
    """Generates Collatz sequence with COM energy-wave interpretation"""
    sequence = []
    while n != 1:
        sequence.append(n)
        if n % 2 == 0:
            n = n // 2  # Energy dissipation (wave amplitude halving)
        else:
            n = int(3 * n + LZ)  # Energy concentration with LZ attractor
    sequence.append(1)
    return sequence

def map_to_octave(value, layer):
    """Maps numbers to 3D octave space using wave principles"""
    # Phase (θ) determined by value's position in Fibonacci mod 9
    fib_mod = [1, 1, 2, 3, 5, 8, 4, 3, 7, 1, 8, 0, 8, 8][value % 14]
    theta = fib_mod * HQS * np.pi
    
    # Amplitude (r) scales with energy density
    r = np.log(value + 1) / LZ
    
    # 3D coordinates - wave representation
    x = r * np.cos(theta) * (1 + 0.1 * layer)  # Spiral growth per layer
    y = r * np.sin(theta) * (1 + 0.1 * layer)
    z = layer  # Time as vertical dimension (wave frequency stacking)
    
    return x, y, z

# Generate sequences for numbers 1-20 with COM interpretation
collatz_waves = {}
for n in range(1, 21):
    seq = generate_collatz_sequence(n)
    collatz_waves[n] = [map_to_octave(val, i) for i, val in enumerate(seq)]

# 3D Visualization
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot energy wave trajectories
for n, wave in collatz_waves.items():
    x = [p[0] for p in wave]
    y = [p[1] for p in wave]
    z = [p[2] for p in wave]
    
    # Color by stability (convergence rate)
    color = plt.cm.plasma(n/20)
    ax.plot(x, y, z, c=color, label=f'Wave {n}', 
            linewidth=2 - 1.8*(n%2), alpha=0.8)
    
    # Nodes as energy density markers
    ax.scatter(x, y, z, c=color, s=50*np.log(n+1), 
               depthshade=False, edgecolors='w')

# Add fundamental wave grid
theta = np.linspace(0, 2*np.pi, 100)
for layer in range(5):
    r = 0.5 * (layer + 1)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    ax.plot(x, y, layer, 'k--', alpha=0.3)

ax.set_title('COM Framework: Collatz Sequences as Energy Waves\n'
             'Space=Amplitude | Time=Frequency Stacking', pad=20)
ax.set_xlabel('X (Real Wave Component)')
ax.set_ylabel('Y (Imaginary Wave Component)')
ax.set_zlabel('Z (Time/Octave Layers)')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(False)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

plt.tight_layout()
plt.show()