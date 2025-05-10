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

# --- Generate Spiral Data ---
spiral_data = {}
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

ax.set_title("Spiral Mapping of Even Perfect Numbers")
ax.set_xlabel("X (LZ-scaled Radius)")
ax.set_ylabel("Y (Phase Angle)")
ax.set_zlabel("Collatz Step (Layer)")
ax.legend()
plt.show()

# --- Compute Euclidean Distances Between Spiral Endpoints ---
def euclidean_distance(p1, p2):
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)

endpoints = {n: spiral_data[n][-1] for n in even_perfects}
sorted_numbers = sorted(even_perfects)

print("Euclidean Distances Between Spiral Endpoints:")
for i in range(1, len(sorted_numbers)):
    n1 = sorted_numbers[i-1]
    n2 = sorted_numbers[i]
    dist = euclidean_distance(endpoints[n1], endpoints[n2])
    print(f"Distance between {n1} and {n2}: {dist:.2f}")
