import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parameters
LZ = 1.23489  # Fractal scaling
HQS = 0.235    # Energy transfer ratio

def generate_collatz_sequence(n):
    sequence = [n]
    while n != 1:
        n = 3 * n + 1 if n % 2 else n // 2
        sequence.append(n)
    return sequence

def reduce_to_single_digit(value):
    return (value - 1) % 9 + 1

def map_to_octave(value, layer, LZ=LZ):
    angle = (value / 9) * 2 * np.pi
    radius = LZ * (layer + 1)  # Apply LZ scaling
    x = radius * np.cos(angle)
    y = radius * np.sin(angle)
    z = layer
    return x, y, z

# Generate sequences and map to octave
collatz_data = {n: generate_collatz_sequence(n) for n in range(1, 21)}
octave_positions = {}
for number, sequence in collatz_data.items():
    octave_positions[number] = [map_to_octave(reduce_to_single_digit(v), i) for i, v in enumerate(sequence)]

# Energy redistribution (HQS)
def redistribute_energy(positions):
    energy = {n: len(seq) for n, seq in collatz_data.items()}
    for n in energy.copy():
        neighbors = [n + int(LZ * k) for k in [-1, 1]]  # Neighbors scaled by LZ
        transfer = HQS * energy[n]
        energy[n] -= transfer
        for neighbor in neighbors:
            if neighbor in energy:
                energy[neighbor] += transfer / len(neighbors)
    return energy

energy = redistribute_energy(octave_positions)

# Plot
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

for number, positions in octave_positions.items():
    x, y, z = zip(*positions)
    ax.plot(x, y, z, label=f'N={number}', alpha=0.6)
    ax.scatter(x, y, z, s=20 * energy.get(number, 1), color='blue')  # Size ~ energy

# Highlight attractors (energy > threshold)
attractor_threshold = np.median(list(energy.values()))
for number, e in energy.items():
    if e > attractor_threshold:
        x, y, z = octave_positions[number][-1]  # Last position in sequence
        ax.scatter(x, y, z, s=200, c='red', marker='*', label='Attractor')

ax.set_title(f"Collatz-Octave Field (LZ={LZ}, HQS={HQS*100}%)")
ax.set_xlabel("X (Fractal Scaling)")
ax.set_ylabel("Y (Harmonic Phase)")
ax.set_zlabel("Z (Octave Layer)")
plt.legend()
plt.show()