import numpy as np
import matplotlib.pyplot as plt

# Even Perfect Numbers
even_perfects = [6, 28, 496, 8128]

def collatz_sequence(n):
    seq = [n]
    while n != 1:
        n = n // 2 if n % 2 == 0 else 3 * n + 1
        seq.append(n)
    return seq

def digital_root(n):
    return (n - 1) % 9 + 1

def spiral_coordinates(n, step, scale=0.1):
    angle = 2 * np.pi * digital_root(n) / 9
    radius = np.log(n) * scale
    return (radius * np.cos(angle), radius * np.sin(angle), step)

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

for perfect in even_perfects:
    sequence = collatz_sequence(perfect)
    x, y, z = zip(*[spiral_coordinates(n, i) for i, n in enumerate(sequence)])
    
    # Plot the sequence
    ax.plot(x, y, z, label=f'Sequence for {perfect}')
    
    # Highlight the perfect number
    ax.scatter(x[0], y[0], z[0], color='red', s=100, label='Perfect number' if perfect == even_perfects[0] else '')
    
    # Highlight the collapse to 1
    ax.scatter(x[-1], y[-1], z[-1], color='green', s=100, label='Collapse to 1' if perfect == even_perfects[0] else '')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Step in sequence')
ax.set_title('Collatz Sequences for Even Perfect Numbers')
ax.legend()

plt.show()

# Print digital roots
for perfect in even_perfects:
    print(f"Digital root of {perfect}: {digital_root(perfect)}")
