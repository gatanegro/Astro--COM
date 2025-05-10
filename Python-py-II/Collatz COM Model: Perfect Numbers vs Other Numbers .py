import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- Collatz Sequence Generator ---
def collatz_sequence(n):
    sequence = [n]
    while n != 1:
        n = n // 2 if n % 2 == 0 else 3 * n + 1
        sequence.append(n)
    return sequence

# --- Digital Root Reduction ---
def reduce_to_single_digit(n):
    return (n - 1) % 9 + 1

# --- Map to Circle or Center ---
def map_to_circle_or_center(value, layer, is_center):
    angle = (value / 9) * 2 * np.pi if not is_center else 0
    radius = layer + 1 if not is_center else 0
    x = np.cos(angle) * radius
    y = np.sin(angle) * radius
    z = layer
    return x, y, z

# --- Parameters ---
even_perfects = [6, 28, 496]
other_numbers = [10, 15, 22]  # Example non-perfect numbers for comparison
stack_spacing = 1.0

# --- Generate Data ---
perfect_data = {}
other_data = {}

for number in even_perfects:
    sequence = collatz_sequence(number)
    mapped_positions = []
    for layer, value in enumerate(sequence):
        is_center = reduce_to_single_digit(value) == 1
        x, y, z = map_to_circle_or_center(value, layer, is_center)
        mapped_positions.append((x, y, z))
    perfect_data[number] = mapped_positions

for number in other_numbers:
    sequence = collatz_sequence(number)
    mapped_positions = []
    for layer, value in enumerate(sequence):
        is_center = reduce_to_single_digit(value) == 1
        x, y, z = map_to_circle_or_center(value, layer, is_center)
        mapped_positions.append((x, y, z))
    other_data[number] = mapped_positions

# --- Plotting ---
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot even perfect numbers (center-to-center behavior)
for number, positions in perfect_data.items():
    x_vals = [p[0] for p in positions]
    y_vals = [p[1] for p in positions]
    z_vals = [p[2] for p in positions]
    
    ax.plot(x_vals, y_vals, z_vals, label=f'Perfect {number}', lw=2)
    ax.scatter(x_vals[0], y_vals[0], z_vals[0], color='red', s=80)   # Start point (center)
    ax.scatter(x_vals[-1], y_vals[-1], z_vals[-1], color='blue', s=80) # End point (center)

# Plot other numbers (clock-like circumference behavior)
for number, positions in other_data.items():
    x_vals = [p[0] for p in positions]
    y_vals = [p[1] for p in positions]
    z_vals = [p[2] for p in positions]
    
    ax.plot(x_vals, y_vals, z_vals, label=f'Other {number}', lw=2)
    ax.scatter(x_vals[0], y_vals[0], z_vals[0], color='green', s=50)   # Start point (circumference)
    ax.scatter(x_vals[-1], y_vals[-1], z_vals[-1], color='orange', s=50) # End point (circumference)

ax.set_title("Collatz COM Model: Perfect Numbers vs Other Numbers")
ax.set_xlabel("X (Horizontal Oscillation)")
ax.set_ylabel("Y (Vertical Oscillation)")
ax.set_zlabel("Z (Layer)")
ax.legend()
plt.show()
