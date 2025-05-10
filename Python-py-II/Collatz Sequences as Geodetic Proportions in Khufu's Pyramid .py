import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Earth-Khufu scaling factors
EARTH_EQUATOR = 40_075_000  # meters
KHUFU_SCALE = 1 / 43_200

def collatz_geometric(n):
    sequence = []
    while n != 1:
        if n % 2 == 0:
            n = n // 2  # Subharmonic division
        else:
            n = 3 * n + 1  # Resonant multiplication + unity
        scaled_value = n * KHUFU_SCALE * EARTH_EQUATOR  # Convert to meters
        sequence.append(scaled_value)
    return sequence

# Generate Collatz sequences for key numbers (3, 7, 22)
collatz_data = {seed: collatz_geometric(seed) for seed in [3, 7, 22]}

# Convert to 3D pyramid coordinates
def map_to_pyramid(value, layer):
    angle = (value / EARTH_EQUATOR) * 2 * np.pi  # Ratio as angle
    radius = layer * 146.6  # Khufu's original height per layer
    x = np.cos(angle) * radius
    y = np.sin(angle) * radius
    z = layer * 146.6  # Height proportional to layer
    return x, y, z

# Plot
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

for seed, sequence in collatz_data.items():
    x_vals, y_vals, z_vals = [], [], []
    for layer, value in enumerate(sequence):
        x, y, z = map_to_pyramid(value, layer)
        x_vals.append(x)
        y_vals.append(y)
        z_vals.append(z)
    ax.plot(x_vals, y_vals, z_vals, marker='o', label=f'Seed {seed}')

ax.set_title("Collatz Sequences as Geodetic Proportions in Khufu's Pyramid", fontsize=12)
ax.set_xlabel("East-West (Meters)")
ax.set_ylabel("North-South (Meters)")
ax.set_zlabel("Height (Meters)")
ax.legend()
plt.show()