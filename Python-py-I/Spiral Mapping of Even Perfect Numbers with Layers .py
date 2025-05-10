import numpy as np
import matplotlib.pyplot as plt

# --- Even Perfect Numbers ---
even_perfects = [6, 28, 496, 8128]

# --- Collatz Sequence Generator ---
def generate_collatz_sequence(n):
    sequence = [n]
    while n != 1:
        n = n // 2 if n % 2 == 0 else 3 * n + 1
        sequence.append(n)
    return sequence

# --- Digital Root Reduction ---
def reduce_to_single_digit(n):
    return (n - 1) % 9 + 1

# --- Parameters ---
LZ_BASE = 1.23498
HQS = 0.235  # Adjust scaling factor (23.5%)
LZ_adjusted = LZ_BASE * (1 + HQS)

# --- Map to Spiral ---
def map_to_spiral(digit, layer, LZ):
    angle = (digit / 9.0) * 2 * np.pi
    radius = (layer + 1) * LZ
    x = np.cos(angle) * radius
    y = np.sin(angle) * radius
    z = layer
    return (x, y, z)

# --- Find Layers for Even Perfect Numbers ---
def find_layers_in_collatz(even_perfects):
    layers = {}
    for n in even_perfects:
        collatz_seq = generate_collatz_sequence(n)
        layers[n] = {num: i for i, num in enumerate(collatz_seq)}
    return layers

# --- Generate Spiral Data ---
spiral_data = {}
even_perfect_layers = find_layers_in_collatz(even_perfects)

for n in even_perfects:
    collatz_seq = generate_collatz_sequence(n)
    reduced_digits = [reduce_to_single_digit(x) for x in collatz_seq]
    points = [map_to_spiral(digit, i, LZ_adjusted) for i, digit in enumerate(reduced_digits)]
    spiral_data[n] = points

# --- Plotting Spiral Data ---
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

for n, points in spiral_data.items():
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    zs = [p[2] for p in points]
    
    # Plot spiral path
    ax.plot(xs, ys, zs, label=f'Collatz spiral for {n}', lw=2, alpha=0.8)
    
    # Mark start and end points
    ax.scatter(xs[0], ys[0], zs[0], color='green', s=80)  # Start point
    ax.scatter(xs[-1], ys[-1], zs[-1], color='red', s=80)  # Endpoint
    
    # Highlight specific layers corresponding to even perfect numbers
    layer_indices = list(even_perfect_layers[n].values())
    for idx in layer_indices:
        ax.scatter(xs[idx], ys[idx], zs[idx], color='blue', s=100, label=f'{n} Layer' if idx == layer_indices[0] else '')

ax.set_title("Spiral Mapping of Even Perfect Numbers with Layers")
ax.set_xlabel("X (LZ-scaled Radius)")
ax.set_ylabel("Y (Phase Angle)")
ax.set_zlabel("Collatz Step (Layer)")
ax.legend()
plt.show()

# --- Print Layer Information ---
print("Layer Information for Even Perfect Numbers:")
for n in even_perfects:
    collatz_seq = generate_collatz_sequence(n)
    print(f"\nEven Perfect Number: {n}")
    print(f"Collatz Sequence: {collatz_seq}")
    
    print(f"Layers: {even_perfect_layers[n]}")