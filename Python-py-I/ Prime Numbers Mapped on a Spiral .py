import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Function to generate Collatz sequence for a number
def generate_collatz_sequence(n):
    sequence = [n]
    while n != 1:
        if n % 2 == 0:
            n = n // 2
        else:
            n = 3 * n + 1
        sequence.append(n)
    return sequence

# Function to reduce numbers to a single-digit using modulo 9 (octave reduction)
def reduce_to_single_digit(value):
    return (value - 1) % 9 + 1

# Function to map reduced values to an octave structure
def map_to_octave(value, layer):
    angle = (value / 9) * 2 * np.pi  # Mapping to a circular octave
    x = np.cos(angle) * (layer + 1)
    y = np.sin(angle) * (layer + 1)
    return x, y

# Generate Collatz sequences for numbers 1 to 20
collatz_data = {n: generate_collatz_sequence(n) for n in range(1, 21)}

# Map sequences to the octave model with reduction
octave_positions = {}
num_layers = max(len(seq) for seq in collatz_data.values())
stack_spacing = 1.0  # Space between layers

for number, sequence in collatz_data.items():
    mapped_positions = []
    for layer, value in enumerate(sequence):
        reduced_value = reduce_to_single_digit(value)
        x, y = map_to_octave(reduced_value, layer)
        z = layer * stack_spacing  # Layer height in 3D
        mapped_positions.append((x, y, z))
    octave_positions[number] = mapped_positions

# Plot the 3D visualization
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot each Collatz sequence as a curve
for number, positions in octave_positions.items():
    x_vals = [pos[0] for pos in positions]
    y_vals = [pos[1] for pos in positions]
    z_vals = [pos[2] for pos in positions]
    ax.plot(x_vals, y_vals, z_vals, label=f"Collatz {number}")
    ax.scatter(x_vals, y_vals, z_vals, s=20, zorder=5)  # Points for clarity

# Add labels and adjust the view
ax.set_title("3D Collatz Sequences in Octave Model")
ax.set_xlabel("X (Horizontal Oscillation)")
ax.set_ylabel("Y (Vertical Oscillation)")
ax.set_zlabel("Z (Octave Layer)")
plt.legend(loc='upper right', fontsize='small')

# Generate prime numbers using the Sieve of Eratosthenes
def generate_primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False  # 0 and 1 are not primes
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    return [x for x in range(limit + 1) if sieve[x]]

# Generate primes up to a limit
prime_limit = 500
primes = generate_primes(prime_limit)

# Map primes onto a spiral for symmetry visualization
def prime_spiral(primes):
    theta = [i * 2 * np.pi / 10 for i in range(len(primes))]  # Angular progression
    radius = [np.log2(p + 1) for p in primes]  # Radius based on prime magnitude
    x = [r * np.cos(t) for r, t in zip(radius, theta)]
    y = [r * np.sin(t) for r, t in zip(radius, theta)]
    return x, y

# Generate spiral coordinates for primes
x_spiral, y_spiral = prime_spiral(primes)

# Plot the prime spiral
plt.figure(figsize=(10, 8))
plt.scatter(x_spiral, y_spiral, c="blue", label="Primes", s=30)
plt.title("Prime Numbers Mapped on a Spiral", fontsize=16)
plt.xlabel("X Coordinate", fontsize=14)
plt.ylabel("Y Coordinate", fontsize=14)
plt.grid(alpha=0.3)
plt.legend()
plt.show()