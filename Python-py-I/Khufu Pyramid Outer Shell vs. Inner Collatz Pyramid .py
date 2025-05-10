import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Collatz sequence generator
def generate_collatz_sequence(n):
    sequence = [n]
    while n != 1:
        n = n // 2 if n % 2 == 0 else 3 * n + 1
        sequence.append(n)
    return sequence

# Reduction for 8-faced pyramid
def reduce_to_octa(value):
    return (value - 1) % 8 + 1

# Inner pyramid mapping (scaled down)
def map_to_inner_pyramid(value, layer, stack_spacing=1.0):
    angle = np.deg2rad(45 * (value - 1))  # 8 directions
    radius = layer * np.tan(np.radians(51.8)) * 0.8  # 80% of outer
    x = np.cos(angle) * radius
    y = np.sin(angle) * radius
    z = layer * stack_spacing
    return x, y, z

# Plot outer pyramid shell
def plot_outer_pyramid(ax, num_layers, stack_spacing):
    for layer in range(num_layers):
        radius = layer * np.tan(np.radians(51.8))
        theta = np.linspace(0, 2*np.pi, 8, endpoint=False)
        x = np.cos(theta) * radius
        y = np.sin(theta) * radius
        z = np.full_like(x, layer * stack_spacing)
        ax.plot(np.append(x, x[0]), np.append(y, y[0]), np.append(z, z[0]),
                color='#1f77b4', alpha=0.15, linewidth=2, linestyle='--')

# Generate Collatz data (n=1 to 30 for denser inner structure)
collatz_data = {n: generate_collatz_sequence(n) for n in range(1, 31)}
num_layers = max(len(seq) for seq in collatz_data.values())
stack_spacing = 1.0

# Map Collatz sequences to inner pyramid
inner_positions = {}
for number, sequence in collatz_data.items():
    mapped = []
    for layer, value in enumerate(sequence):
        reduced = reduce_to_octa(value)
        x, y, z = map_to_inner_pyramid(reduced, layer, stack_spacing)
        mapped.append((x, y, z))
    inner_positions[number] = mapped

# Create plot
fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(111, projection='3d')

# Plot outer pyramid
plot_outer_pyramid(ax, num_layers, stack_spacing)

# Plot inner Collatz pyramid (red/orange for contrast)
colors = plt.cm.autumn(np.linspace(0, 1, len(collatz_data)))
for (number, positions), color in zip(inner_positions.items(), colors):
    x_vals = [pos[0] for pos in positions]
    y_vals = [pos[1] for pos in positions]
    z_vals = [pos[2] for pos in positions]
    ax.plot(x_vals, y_vals, z_vals, color=color, linewidth=1.5, alpha=0.7)

# Configure axes and labels
ax.set_title("Khufu Pyramid Outer Shell vs. Inner Collatz Pyramid", fontsize=14)
ax.set_xlabel("East-West Axis", fontsize=12)
ax.set_ylabel("North-South Axis", fontsize=12)
ax.set_zlabel("Height (Layers)", fontsize=12)
ax.grid(True, alpha=0.3)
ax.set_box_aspect([2*np.tan(np.radians(51.8)), 2*np.tan(np.radians(51.8)), 1])

plt.tight_layout()
plt.show()