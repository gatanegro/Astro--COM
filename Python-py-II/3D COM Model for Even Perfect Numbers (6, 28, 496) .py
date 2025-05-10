import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def collatz_sequence(n):
    sequence = [n]
    while n != 1:
        n = 3 * n + 1 if n % 2 else n // 2
        sequence.append(n)
    return sequence

def digital_root(n):
    return (n - 1) % 9 + 1 if n else 0

def spiral_coordinates(n, step, scale=0.1):
    angle = 2 * np.pi * digital_root(n) / 9
    radius = np.log(n) * scale
    return (radius * np.cos(angle), radius * np.sin(angle), step)

even_perfects = [6, 28, 496]
colors = ['r', 'g', 'b']

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

for number, color in zip(even_perfects, colors):
    sequence = collatz_sequence(number)
    coords = [spiral_coordinates(n, i) for i, n in enumerate(sequence)]
    x, y, z = zip(*coords)
    
    ax.plot(x, y, z, color=color, label=f'Sequence for {number}')
    ax.scatter(x[0], y[0], z[0], color=color, s=100, marker='o')
    ax.text(x[0], y[0], z[0], f'{number}', fontsize=10)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Step in sequence')
ax.set_title('3D COM Model for Even Perfect Numbers (6, 28, 496)')
ax.legend()

plt.show()

# Print additional information
for number in even_perfects:
    sequence = collatz_sequence(number)
    print(f"\nNumber: {number}")
    print(f"Sequence length: {len(sequence)}")
    print(f"Digital root: {digital_root(number)}")
    print(f"Sequence: {sequence}")
