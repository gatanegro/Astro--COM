import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_collatz(n):
    sequence = [n]
    while n != 1:
        n = n // 2 if n % 2 == 0 else 3 * n + 1
        sequence.append(n)
    return sequence

def octave_reduce(value):
    return (value - 1) % 9 + 1

def plot_sequences():
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    for number in [7, 9, 15]:  # More sequences
        seq = generate_collatz(number)
        x, y, z, energies = [], [], [], []
        time = 0
        for step, value in enumerate(seq):
            energy = octave_reduce(value)
            angle = (energy / 9) * 2 * np.pi
            x_val = np.cos(angle) * (step + 1)
            y_val = np.sin(angle) * (step + 1)
            time += energy * 0.1  # Adjust this for time scaling
            x.append(x_val)
            y.append(y_val)
            z.append(time)
            energies.append(energy)
        
        ax.plot(x, y, z, label=f'Collatz {number}', alpha=0.5)
        ax.scatter(x, y, z, c=energies, cmap='viridis', s=30)  # Color by energy

    ax.set_title("Collatz Energy-Time with Color-Coded States")
    ax.set_xlabel("X (Spatial Oscillation)")
    ax.set_ylabel("Y (Spatial Oscillation)")
    ax.set_zlabel("Time (Cumulative Phase)")
    ax.legend()
    plt.show()

plot_sequences()