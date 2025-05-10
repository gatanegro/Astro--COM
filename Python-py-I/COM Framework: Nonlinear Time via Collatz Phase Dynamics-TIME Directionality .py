import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parameters
T_unit = 1.0  # Time unit (e.g., Planck time, cosmic cycle)
BASE_RADIUS = 1.0  # Base radius for geometric scaling

def generate_collatz_sequence(n):
    sequence = [n]
    while n != 1:
        n = n // 2 if n % 2 == 0 else 3 * n + 1
        sequence.append(n)
    return sequence

def reduce_to_single_digit(value):
    return (value - 1) % 9 + 1

def map_to_octave(reduced_value, radius):
    phase = (reduced_value / 9) * 2 * np.pi
    x = np.cos(phase) * radius
    y = np.sin(phase) * radius
    return x, y, phase  # Return phase for time calculation

# Generate Collatz sequences
collatz_data = {n: generate_collatz_sequence(n) for n in range(1, 21)}

# Map to 3D spacetime with recursive time
octave_positions = {}
for number, sequence in collatz_data.items():
    positions = []
    T = 0.0  # Initial time
    prev_phase = None
    for layer, value in enumerate(sequence):
        reduced = reduce_to_single_digit(value)
        x, y, phase = map_to_octave(reduced, BASE_RADIUS * (layer + 1))
        
        # Calculate time differential from phase change
        if prev_phase is not None:
            delta_phase = abs(phase - prev_phase)  # Signed phase difference
            T += delta_phase / (2 * np.pi) * T_unit  # Negative delta_phase = time reversal

        positions.append((x, y, T))
        prev_phase = phase
    octave_positions[number] = positions

# Plot 3D spacetime
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

for number, path in octave_positions.items():
    x = [p[0] for p in path]
    y = [p[1] for p in path]
    z = [p[2] for p in path]
    ax.plot(x, y, z, marker='o', label=f'COM {number}')
    ax.text(x[-1], y[-1], z[-1], str(number), fontsize=8)

ax.set_title("COM Framework: Nonlinear Time via Collatz Phase Dynamics", fontsize=14)
ax.set_xlabel("X (Spatial Oscillation)")
ax.set_ylabel("Y (Spatial Oscillation)")
ax.set_zlabel("Z (Local Time)")
ax.legend()
plt.show()