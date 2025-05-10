import numpy as np
import matplotlib.pyplot as plt

# --- Even Perfect Numbers ---
even_perfects = [6, 28, 496, 8128,2305843008139952128]

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
HQS = 0.235  # 23.5%
LZ_adjusted = LZ_BASE * (1 + HQS)

# --- Map to 3D Spiral using the adjusted LZ ---
def map_to_spiral(digit, layer, LZ):
    angle = (digit / 9.0) * 2 * np.pi
    radius = (layer + 1) * LZ
    x = np.cos(angle) * radius
    y = np.sin(angle) * radius
    z = layer
    return (x, y, z)

# --- Theoretical Radial Distance Function ---
def theoretical_radial_distance(n, C=1):
    return C * np.log2(n) * LZ_adjusted

# --- Generate Spiral and Theoretical Data ---
spiral_data = {}
theoretical_data = {}
for n in even_perfects:
    collatz_seq = generate_collatz_sequence(n)
    reduced_digits = [reduce_to_single_digit(x) for x in collatz_seq]
    points = [map_to_spiral(d, i, LZ_adjusted) for i, d in enumerate(reduced_digits)]
    spiral_data[n] = points
    theoretical_data[n] = theoretical_radial_distance(n, C=3)

# --- Plotting the Spiral Data ---
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

for n, points in spiral_data.items():
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    zs = [p[2] for p in points]
    ax.plot(xs, ys, zs, label=f'Collatz spiral for {n}', lw=2, alpha=0.8)
    ax.scatter(xs[0], ys[0], zs[0], color='green', s=80)  # start point
    ax.scatter(xs[-1], ys[-1], zs[-1], color='red', s=80)  # endpoint
    rad_theo = theoretical_data[n]
    ax.text(xs[-1], ys[-1], zs[-1], f' R_th={rad_theo:.2f}', color='black')

ax.set_title("Spiral Mapping with Theoretical Radial Distances")
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

print("\nTheoretical Radial Distance Differences (using C=1):")
for i in range(1, len(sorted_numbers)):
    n1 = sorted_numbers[i-1]
    n2 = sorted_numbers[i]
    diff_theo = theoretical_data[n2] - theoretical_data[n1]
    print(f"Theoretical difference between {n1} and {n2}: {diff_theo:.2f}")