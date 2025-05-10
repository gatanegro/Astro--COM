import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Generate Collatz sequence
def generate_collatz(n):
    sequence = [n]
    while n != 1:
        n = n // 2 if n % 2 == 0 else 3 * n + 1
        sequence.append(n)
    return sequence

# Reduce to energy state (1-9)
def octave_reduce(value):
    return (value - 1) % 9 + 1

# Plot 3D spacetime
def plot_sequences():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot sequences 7 and 9 (short, illustrative)
    for number in [7, 9]:
        seq = generate_collatz(number)
        x, y, z = [], [], []
        time = 0
        for step, value in enumerate(seq):
            energy = octave_reduce(value)
            angle = (energy / 9) * 2 * np.pi
            x_val = np.cos(angle) * (step + 1)
            y_val = np.sin(angle) * (step + 1)
            time += energy * 0.1  # Simple time scaling
            x.append(x_val)
            y.append(y_val)
            z.append(time)
        ax.plot(x, y, z, label=f'Collatz {number}', marker='o')

    ax.set_title("Collatz Energy-Time")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Time (Energy Phase)")
    ax.legend()
    plt.show()

# Run directly without __name__ check
plot_sequences()
