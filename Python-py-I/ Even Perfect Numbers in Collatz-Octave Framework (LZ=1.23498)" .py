import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- Generate Even Perfect Numbers ---
def generate_even_perfect_numbers():
    return [6, 28, 496, 8128]  # First four even perfect numbers

even_perfects = generate_even_perfect_numbers()

# --- Collatz Sequence Generator ---
def generate_collatz_sequence(n):
    sequence = [n]
    while n != 1:
        n = n // 2 if n % 2 == 0 else 3 * n + 1
        sequence.append(n)
    return sequence

# --- Digit Reduction Function ---
def reduce_to_single_digit(value):
    return (value - 1) % 9 + 1  # Maps to 1-9 (digital root)

# --- Map to 3D Spiral with LZ Scaling ---
LZ_CONSTANT = 1.23498

def map_to_spiral(digit, layer):
    angle = (digit / 9) * 2 * np.pi  # Position on the circle
    radius = (layer + 1) * LZ_CONSTANT  # Radial scaling
    x = np.cos(angle) * radius
    y = np.sin(angle) * radius
    z = layer  # Step index as height
    return (x, y, z)

# --- Generate Data ---
collatz_data = {n: [reduce_to_single_digit(x) for x in generate_collatz_sequence(n)] for n in even_perfects}
spiral_data = {n: [map_to_spiral(d, i) for i, d in enumerate(digits)] for n, digits in collatz_data.items()}

# --- Plotting ---
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

colors = {'expand': '#1f77b4', 'contract': '#ff7f0e', 'neutral': '#888888'}

for n, points in spiral_data.items():
    x_vals = [p[0] for p in points]
    y_vals = [p[1] for p in points]
    z_vals = [p[2] for p in points]
    
    # Classify digits
    categories = []
    for d in collatz_data[n]:
        if d in {1, 3, 7, 9}:
            categories.append('expand')
        elif d in {2, 4, 6, 8}:
            categories.append('contract')
        else:
            categories.append('neutral')
    
    # Plot each segment with color
    for i in range(len(x_vals)-1):
        ax.plot(x_vals[i:i+2], y_vals[i:i+2], z_vals[i:i+2], 
                color=colors[categories[i]], alpha=0.7)
    
    # Mark start (green) and end (red)
    ax.scatter(x_vals[0], y_vals[0], z_vals[0], color='green', s=100, label='Start' if n == 6 else '')
    ax.scatter(x_vals[-1], y_vals[-1], z_vals[-1], color='red', s=100, label='End (1)' if n == 6 else '')

ax.set_title("Even Perfect Numbers in Collatz-Octave Framework (LZ=1.23498)")
ax.set_xlabel("X (LZ-scaled Radius)")
ax.set_ylabel("Y (Phase Angle)")
ax.set_zlabel("Collatz Step (Layer)")
ax.legend()
plt.show()