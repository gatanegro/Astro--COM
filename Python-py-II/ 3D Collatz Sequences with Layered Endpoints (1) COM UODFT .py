import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Function to generate the Collatz sequence for a given starting number
def generate_collatz_sequence(n):
    sequence = [n]
    while n != 1:
        if n % 2 == 0:
            n //= 2
        else:
            n = 3 * n + 1
        sequence.append(n)
    return sequence

# Mapping function: assign a number to a circle based on its magnitude.
# The circle index k is floor((n-1)/9). The center of that circle is 1+9*k.
# If n equals the center, it is plotted at (0,0). Otherwise, it is placed on the circumference.
def map_to_circle(n, layer, circle_spacing=1.0, circle_radius=1.0):
    k = (n - 1) // 9  # determine which circle this number belongs to
    center_value = 1 + 9 * k
    if n == center_value:
        x = 0
        y = 0
    else:
        # positions from center+1 to center+8
        pos_index = n - center_value - 1  # index 0 to 7
        angle = 2 * np.pi * pos_index / 8
        x = np.cos(angle) * circle_radius
        y = np.sin(angle) * circle_radius
    z = layer * circle_spacing
    return x, y, z

# Main plotting function
def plot_collatz_sequences(start, end, circle_spacing=1.0, circle_radius=1.0, endpoint_radius=1.0):
    # Colors for odd and even starting numbers (customize as desired)
    odd_color = 'blue'
    even_color = 'red'
    
    # Generate Collatz sequences for the desired range.
    collatz_data = {n: generate_collatz_sequence(n) for n in range(start, end + 1)}
    
    # Map each number in the sequences to (x, y, z) positions using our circle mapping.
    # We initially map all points the same way—even if the number is 1.
    positions_data = {}
    for number, sequence in collatz_data.items():
        positions = []
        for layer, value in enumerate(sequence):
            pos = map_to_circle(value, layer, circle_spacing=circle_spacing, circle_radius=circle_radius)
            positions.append(pos)
        positions_data[number] = positions

    # Now, for the endpoints (the value 1) we want to avoid overlap.
    # Gather endpoints by layer. (Usually the final point is 1.)
    endpoints_by_layer = {}
    for number, sequence in collatz_data.items():
        last_layer = len(sequence) - 1
        # The sequence should always end in 1.
        endpoints_by_layer.setdefault(last_layer, []).append(number)
    
    # For each layer where endpoints exist, reassign the (x, y) positions to lie
    # evenly on a circle of radius endpoint_radius.
    for layer, numbers in endpoints_by_layer.items():
        count = len(numbers)
        for idx, number in enumerate(numbers):
            angle = 2 * np.pi * idx / count  # evenly distribute
            x = np.cos(angle) * endpoint_radius
            y = np.sin(angle) * endpoint_radius
            z = layer * circle_spacing
            # Replace the endpoint position for this sequence.
            positions_data[number][-1] = (x, y, z)
    
    # Create the 3D plot.
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    for number, pos_list in positions_data.items():
        x_vals = [p[0] for p in pos_list]
        y_vals = [p[1] for p in pos_list]
        z_vals = [p[2] for p in pos_list]
        # Color by starting number parity
        color = odd_color if number % 2 != 0 else even_color
        ax.plot(x_vals, y_vals, z_vals, label=f"Start {number}", color=color)
        ax.scatter(x_vals, y_vals, z_vals, s=20, color=color)
    
    ax.set_title("3D Collatz Sequences with Layered Endpoints (1)")
    ax.set_xlabel("X (Circle Coordinate)")
    ax.set_ylabel("Y (Circle Coordinate)")
    ax.set_zlabel("Layer (Iteration)")
    ax.legend(loc='upper right', fontsize='small')
    plt.show()

# Example usage: Plot Collatz sequences for starting numbers 1 to 20
plot_collatz_sequences(1, 20, circle_spacing=1.0, circle_radius=1.0, endpoint_radius=1.5)
