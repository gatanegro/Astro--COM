import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Collatz sequence generator (no zeros)
def generate_collatz_sequence(n):
    sequence = [n]
    while n != 1:
        n = n // 2 if n % 2 == 0 else 3 * n + 1
        sequence.append(n)
    return sequence

# Energy density mapping (1-9, no zeros)
def reduce_to_single_digit(value):
    return (value - 1) % 9 + 1  # Maps to [1, 9]

# Compute phase changes from energy gradients
def compute_phase_evolution(sequence):
    energies = [reduce_to_single_digit(v) for v in sequence]
    phases = [0.0]
    for i in range(1, len(energies)):
        rho_prev = energies[i-1]
        rho_current = energies[i]
        d_rho = rho_current - rho_prev
        d_phi = 2 * np.pi * (d_rho / rho_prev)  # Signed phase change
        phases.append(phases[-1] + d_phi)
    return phases

# Map to 3D spacetime (X, Y = spatial phase; Z = cumulative time)
def map_to_spacetime(sequence, phases):
    x, y, z = [], [], []
    for step, (value, phi) in enumerate(zip(sequence, phases)):
        # Spatial coordinates from energy wave
        radius = 1.0 + step * 0.2  # Expanding radius (LZ scaling)
        angle = (reduce_to_single_digit(value) / 9) * 2 * np.pi
        xi = radius * np.cos(angle)
        yi = radius * np.sin(angle)
        # Time = cumulative phase (modular to visualize cycles)
        zi = phi % (2 * np.pi)  # Optional: Remove modulo for unbounded time
        x.append(xi)
        y.append(yi)
        z.append(zi)
    return x, y, z

# Generate sequences and compute phases
collatz_data = {n: generate_collatz_sequence(n) for n in range(1, 21)}
phase_data = {n: compute_phase_evolution(seq) for n, seq in collatz_data.items()}

# Plot
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

for number, seq in collatz_data.items():
    phases = phase_data[number]
    x, y, z = map_to_spacetime(seq, phases)
    ax.plot(x, y, z, marker='o', label=f'COM {number}')
    ax.text(x[-1], y[-1], z[-1], str(number), fontsize=8)

ax.set_title("Zero-Free COM Framework: Time as Phase of Energy Density Waves", fontsize=14)
ax.set_xlabel("X (Spatial Phase)")
ax.set_ylabel("Y (Spatial Phase)")
ax.set_zlabel("Local Time (Cumulative Phase ϕ)")
ax.legend()
plt.show()