import matplotlib.pyplot as plt
import numpy as np

# Function to generate Collatz sequence
def collatz_sequence(n):
    sequence = [n]
    while n != 1:
        if n % 2 == 0:
            n = n // 2
        else:
            n = 3 * n + 1
        sequence.append(n)
    return sequence

# Map Collatz sequences onto a 2D plane to explore FIELD-like patterns
def map_to_plane(sequence):
    x, y = [0], [0]  # Start at the origin
    for i, num in enumerate(sequence):
        angle = (i % 360) * np.pi / 180  # Map step to angle (circle-like)
        radius = np.log(num + 1)  # Use log scale for spacing
        x.append(radius * np.cos(angle))
        y.append(radius * np.sin(angle))
    return x, y

# Generate and visualize patterns for multiple starting values
plt.figure(figsize=(10, 10))
for start in range(2, 50):  # Starting numbers from 2 to 50
    sequence = collatz_sequence(start)
    x, y = map_to_plane(sequence)
    plt.plot(x, y, alpha=0.7, label=f"Start: {start}" if start <= 5 else None)

# Customize plot
plt.title("Collatz Sequences as FIELD-Like Patterns", fontsize=16)
plt.xlabel("X Coordinate (Oscillation)", fontsize=14)
plt.ylabel("Y Coordinate (Oscillation)", fontsize=14)
plt.grid(alpha=0.3)
plt.legend(loc="upper left", fontsize=10, ncol=1, bbox_to_anchor=(1, 1))
plt.show()