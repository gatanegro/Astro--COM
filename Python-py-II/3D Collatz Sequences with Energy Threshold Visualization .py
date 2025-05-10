import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

LZ_CONSTANT = 1.23498
ENERGY_THRESHOLD = 0.235  # 23.5%

def collatz_sequence_with_energy(n):
    sequence = [n]
    energy = [0]
    while n != 1:
        prev_n = n
        n = 3 * n + 1 if n % 2 else n // 2
        energy_change = abs(n - prev_n) / prev_n
        sequence.append(n)
        energy.append(energy_change)
    return sequence, energy

def map_to_3d_space(sequence, energy):
    x = [np.sin(n * LZ_CONSTANT) for n in sequence]
    y = [np.cos(n * LZ_CONSTANT) for n in sequence]
    z = [np.log(n) / LZ_CONSTANT for n in sequence]
    return x, y, z, energy

fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(111, projection='3d')

start_values = [6, 27, 41, 103]
colors = ['r', 'g', 'b', 'y']

for start, color in zip(start_values, colors):
    seq, energy = collatz_sequence_with_energy(start)
    x, y, z, energy = map_to_3d_space(seq, energy)
    
    # Plot points with color intensity based on energy threshold
    for i in range(len(x)):
        intensity = min(energy[i] / ENERGY_THRESHOLD, 1)  # Normalize to [0,1]
        ax.scatter(x[i], y[i], z[i], c=color, alpha=intensity, s=20)
    
    # Connect points with lines
    ax.plot(x, y, z, color=color, alpha=0.3, label=f'Start: {start}')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.title('3D Collatz Sequences with Energy Threshold Visualization')
plt.show()
