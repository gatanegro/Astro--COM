import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Function to generate the Collatz sequence for a given starting number
def generate_collatz_sequence(n):
    sequence = [n]
    while n != 1:
        if n % 2 == 0:
            n = n // 2
        else:
            n = 3 * n + 1
        sequence.append(n)
    return sequence

# Mapping function: assign a number n to a circle based on its magnitude.
# The circle index k is given by floor((n-1)/9). The center of that circle is 1+9*k.
# If n equals the center, it is plotted at (0,0). Otherwise, it is placed on the circumference,
# evenly spaced in angle.
def map_to_circle(n, layer, circle_spacing=1.0, circle_radius=1.0):
    k = (n - 1) // 9  # determine which circle this number belongs to
    center_value = 1 + 9 * k
    if n == center_value:
        # Place the center value at the center of the circle
        x = 0
        y = 0
    else:
        # Numbers other than the center are placed on the circumference.
        # There are 8 positions available (center+1 to center+8).
        pos_index = n - center_value - 1  # index from 0 to 7
        angle = 2 * np.pi * pos_index / 8  # evenly spaced
        x = np.cos(angle) * circle_radius
        y = np.sin(angle) * circle_radius
    # The z-coordinate indicates the layer (i.e. the iteration step in the Collatz sequence)
    z = layer * circle_spacing
    return x, y, z

# Function to generate and plot Collatz sequences for a range of starting numbers
def plot_collatz_sequences(start, end, circle_spacing=1.0, circle_radius=1.0):
    # Colors for odd and even starting numbers (can be adjusted)
    odd_color = 'blue'
    even_color = 'red'
    
    # Generate Collatz sequences for the desired range
    collatz_data = {n: generate_collatz_sequence(n) for n in range(start, end + 1)}
    
    # For each sequence, map each number using our circle mapping
    circle_positions = {}
    for number, sequence in collatz_data.items():
        positions = []
        for layer, value in enumerate(sequence):
            pos = map_to_circle(value, layer, circle_spacing=circle_spacing, circle_radius=circle_radius)
            positions.append(pos)
        circle_positions[number] = positions

    # Create a 3D plot of the sequences
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot each sequence with color depending on the parity of the starting number
    for number, positions in circle_positions.items():
        x_vals = [pos[0] for pos in positions]
        y_vals = [pos[1] for pos in positions]
        z_vals = [pos[2] for pos in positions]
        color = odd_color if number % 2 != 0 else even_color
        ax.plot(x_vals, y_vals, z_vals, label=f"Start {number}", color=color)
        ax.scatter(x_vals, y_vals, z_vals, s=20, color=color)
    
    ax.set_title("3D Collatz Sequences with Circle Mapping")
    ax.set_xlabel("X (Circle Coordinate)")
    ax.set_ylabel("Y (Circle Coordinate)")
    ax.set_zlabel("Layer (Iteration)")
    ax.legend(loc='upper right', fontsize='small')
    plt.show()

# Example usage: Plot Collatz sequences for starting numbers 1 to 20
plot_collatz_sequences(1, 20, circle_spacing=1.0, circle_radius=1.0)
