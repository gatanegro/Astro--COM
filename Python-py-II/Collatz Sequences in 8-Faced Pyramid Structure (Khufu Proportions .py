import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Function to generate Collatz sequence
def generate_collatz_sequence(n):
    sequence = [n]
    while n != 1:
        n = n // 2 if n % 2 == 0 else 3 * n + 1
        sequence.append(n)
    return sequence

# Reduce numbers to 1-8 (for 8-faced pyramid)
def reduce_to_octa(value):
    return (value - 1) % 8 + 1

# Map values to 8 directions with Khufu's slope (51.8°)
def map_to_pyramid(value, layer, stack_spacing=1.0):
    angle = np.deg2rad(45 * (value - 1))  # 8 directions at 45° increments
    radius = layer * np.tan(np.radians(51.8))  # Slope calculation
    x = np.cos(angle) * radius
    y = np.sin(angle) * radius
    z = layer * stack_spacing
    return x, y, z

# Generate Collatz sequences (n=1 to 20)
collatz_data = {n: generate_collatz_sequence(n) for n in range(1, 21)}

# Map sequences to pyramid coordinates
stack_spacing = 1.0
octave_positions = {}
num_layers = max(len(seq) for seq in collatz_data.values())

for number, sequence in collatz_data.items():
    mapped_positions = []
    for layer, value in enumerate(sequence):
        reduced_value = reduce_to_octa(value)
        x, y, z = map_to_pyramid(reduced_value, layer, stack_spacing)
        mapped_positions.append((x, y, z))
    octave_positions[number] = mapped_positions

# Create 3D plot with pyramid scaffolding
fig = plt.figure(figsize=(14, 12))
ax = fig.add_subplot(111, projection='3d')

# Plot Collatz sequences
colors = plt.cm.viridis(np.linspace(0, 1, len(collatz_data)))
for (number, positions), color in zip(octave_positions.items(), colors):
    x_vals = [pos[0] for pos in positions]
    y_vals = [pos[1] for pos in positions]
    z_vals = [pos[2] for pos in positions]
    ax.plot(x_vals, y_vals, z_vals, color=color, marker='o', markersize=4, 
            linewidth=1, alpha=0.7, label=f'N={number}')

# Add 8-faced pyramid wireframe
theta = np.linspace(0, 2*np.pi, 8, endpoint=False)
for layer in range(num_layers):
    radius = layer * np.tan(np.radians(51.8))
    x = np.cos(theta) * radius
    y = np.sin(theta) * radius
    z = np.full_like(x, layer * stack_spacing)
    ax.plot(np.append(x, x[0]), np.append(y, y[0]), np.append(z, z[0]),
            color='gray', linestyle='--', linewidth=0.7, alpha=0.4)

# Configure plot
ax.set_title("Collatz Sequences in 8-Faced Pyramid Structure (Khufu Proportions)", fontsize=14)
ax.set_xlabel("East-West Axis", fontsize=12)
ax.set_ylabel("North-South Axis", fontsize=12)
ax.set_zlabel("Height (Layers)", fontsize=12)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.grid(True, linestyle=':', alpha=0.6)
ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=8)
plt.tight_layout()

# Adjust aspect ratio to match pyramid slope
ax.set_box_aspect([2*np.tan(np.radians(51.8)), 2*np.tan(np.radians(51.8)), 1])

plt.show()