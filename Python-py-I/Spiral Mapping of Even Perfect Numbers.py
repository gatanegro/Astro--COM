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
    # Digital root maps n to a value between 1 and 9
    return (n - 1) % 9 + 1

# --- Parameters ---
LZ_BASE = 1.23498
HQS = 0.235  # 23.5%
LZ_adjusted = LZ_BASE * (1 + HQS)  # Adjusted LZ constant

# --- Map to 3D Spiral using the adjusted LZ ---
def map_to_spiral(digit, layer, LZ):
    angle = (digit / 9.0) * 2 * np.pi  # Map digit to angle [0, 2pi]
    radius = (layer + 1) * LZ           # Radial distance increases per layer
    x = np.cos(angle) * radius
    y = np.sin(angle) * radius
    z = layer                         # Use layer as height
    return (x, y, z)

# --- Theoretical Radial Distance Function ---
def theoretical_radial_distance(n, C=1):
    # Even perfect numbers follow N = 2^(p-1)*(2^p - 1)
    # Use log2(n) to linearize exponential growth
    return C * np.log2(n) * LZ_adjusted

# --- Generate Spiral and Theoretical Data ---
spiral_data = {}
theoretical_data = {}
for n in even_perfects:
    collatz_seq = generate_collatz_sequence(n)
    # Apply digital reduction to each element of the sequence
    reduced_digits = [reduce_to_single_digit(x) for x in collatz_seq]
    # Map each reduced digit to a 3D point using our adjusted constant
    points = [map_to_spiral(d, i, LZ_adjusted) for i, d in enumerate(reduced_digits)]
    spiral_data[n] = points
    # Calculate the theoretical radial distance for the number (using default C = 1)
    theoretical_data[n] = theoretical_radial_distance(n, C=1)

# --- Plotting the Spiral Data ---
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

for n, points in spiral_data.items():
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    zs = [p[2] for p in points]
    ax.plot(xs, ys, zs, label=f'Collatz spiral for {n}', lw=2, alpha=0.8)
    # Mark the start (green) and end (red) points of each spiral
    ax.scatter(xs[0], ys[0], zs[0], color='green', s=80)
    ax.scatter(xs[-1], ys[-1], zs[-1], color='red', s=80)
    # Annotate the final point with the theoretical radial distance
    rad_theo = theoretical_data[n]
    ax.text(xs[-1], ys[-1], zs[-1], f' R_th={rad_theo:.2f}', color='black')

ax.set_title("Spiral Mapping of Even Perfect Numbers\nwith HQS-adjusted LZ Constant and Theoretical Radial Distances")
ax.set_xlabel("X (LZ-scaled Radius)")
ax.set_ylabel("Y (Phase Angle)")
ax.set_zlabel("Collatz Step (Layer)")
ax.legend()
plt.show()

# --- Experimenting with different C values ---
print("Theoretical Radial Distances for different C values:")
Cs = np.linspace(0.5, 2.0, 10)
for C in Cs:
    print(f"\nFor C = {C:.2f}:")
    for n in even_perfects:
        R_theo = theoretical_radial_distance(n, C=C)
        print(f"  n = {n:5d} -> R_theo = {R_theo:.2f}")
