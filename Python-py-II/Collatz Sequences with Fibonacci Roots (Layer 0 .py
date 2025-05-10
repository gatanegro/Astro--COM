import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# === YOUR ORIGINAL FUNCTIONS (UNCHANGED) ===
def generate_collatz(n):
    sequence = [n]
    while n != 1:
        n = n // 2 if n % 2 == 0 else 3 * n + 1
        sequence.append(n)
    return sequence

def reduce_to_single_digit(value):
    return (value - 1) % 9 + 1  # Now returns integers

def map_to_octave(value, layer):
    angle = (int(value) / 9) * 2 * np.pi  # Explicit integer conversion
    x = np.cos(angle) * (layer + 1)
    y = np.sin(angle) * (layer + 1)
    return (x, y, layer)  # Now returns 3-tuple directly

# === FIBONACCI GENERATION ===
def generate_fibonacci(length):
    fib = [1, 1]
    while len(fib) < length:
        fib.append(fib[-1] + fib[-2])
    return fib

# === DATA GENERATION ===
# Collatz data (your original)
collatz_data = {n: generate_collatz(n) for n in range(1, 21)}
octave_positions = {}
for number, sequence in collatz_data.items():
    octave_positions[number] = [map_to_octave(reduce_to_single_digit(val), layer) 
                               for layer, val in enumerate(sequence)]

# Fibonacci data (new)
fib_sequence = generate_fibonacci(24)  # First 24 Fibonacci numbers
fib_reduced = [reduce_to_single_digit(f) for f in fib_sequence]
fib_mapped = [map_to_octave(f, 0) for f in fib_reduced]  # Layer=0

# === PLOTTING ===
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot Collatz paths (your original)
for number, positions in octave_positions.items():
    x, y, z = zip(*positions)
    ax.plot(x, y, z, label=f"Collatz {number}")
    ax.scatter(x, y, z, s=20)

# Plot Fibonacci roots (new)
fib_x, fib_y, fib_z = zip(*fib_mapped)
ax.scatter(fib_x, fib_y, fib_z, s=200, c='red', marker='*', label='Fibonacci Roots')

ax.set_title("Collatz Sequences with Fibonacci Roots (Layer 0)", fontsize=14)
ax.set_xlabel("X: Real(Energy Phase)")
ax.set_ylabel("Y: Imag(Energy Phase)")
ax.set_zlabel("Z: Octave Layer")
plt.legend()
plt.tight_layout()
plt.show()