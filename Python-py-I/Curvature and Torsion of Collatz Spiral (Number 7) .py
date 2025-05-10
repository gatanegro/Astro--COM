import numpy as np
import matplotlib.pyplot as plt

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
ax.legend(loc='upper right', fontsize='small')

plt.show()
import numpy as np
import matplotlib.pyplot as plt

def curvature_torsion(points):
    points = np.array(points)
    # First derivative
    d1 = np.gradient(points, axis=0)
    # Second derivative
    d2 = np.gradient(d1, axis=0)
    # Third derivative
    d3 = np.gradient(d2, axis=0)

    cross = np.cross(d1, d2)
    curvature = np.linalg.norm(cross, axis=1) / (np.linalg.norm(d1, axis=1) ** 3 + 1e-8)
    torsion = np.einsum('ij,ij->i', cross, d3) / (np.linalg.norm(cross, axis=1) ** 2 + 1e-8)

    return curvature, torsion

# Example: compute curvature and torsion for Collatz sequence of number 7
example_points = octave_positions[7]  # Change number as needed
curv, tors = curvature_torsion(example_points)

plt.figure(figsize=(10, 5))
plt.plot(curv, label='Curvature')
plt.plot(tors, label='Torsion')
plt.legend()
plt.title('Curvature and Torsion of Collatz Spiral (Number 7)')
plt.xlabel('Layer')
plt.ylabel('Value')
plt.show()
