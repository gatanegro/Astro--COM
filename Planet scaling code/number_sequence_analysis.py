"""
Number Sequence Analysis in 3D COM Framework
Author: Martin Doina
Date: April 25, 2025

This script analyzes the convergence of Fibonacci, Tribonacci, Pi, and Lucas sequences 
in the 3D Continuous Oscillatory Model (COM) framework, with a focus on their convergence
at the number 5 and connections to quasicrystals.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

# Constants from the COM framework
LZ = 1.23498  # LZ constant
HQS = 0.235   # HQS constant

# Function to generate Fibonacci sequence
def fibonacci(n):
    sequence = [1, 1]
    while len(sequence) < n:
        sequence.append(sequence[-1] + sequence[-2])
    return sequence

# Function to generate Tribonacci sequence
def tribonacci(n):
    sequence = [1, 1, 2]
    while len(sequence) < n:
        sequence.append(sequence[-1] + sequence[-2] + sequence[-3])
    return sequence

# Function to generate Lucas sequence
def lucas(n):
    sequence = [2, 1]
    while len(sequence) < n:
        sequence.append(sequence[-1] + sequence[-2])
    return sequence

# Function to generate Pi digits
def pi_digits(n):
    pi_str = str(math.pi).replace('.', '')
    return [int(digit) for digit in pi_str[:n]]

# Function to reduce a number to its root number (digital root)
def reduce_to_root(n):
    if n == 0:
        return 0
    return 1 + ((n - 1) % 9)

# Function to map a number to 3D coordinates in the COM framework
def map_to_com_3d(n, layer=0):
    # Map the number to a position on a circle
    angle = 2 * math.pi * n / 9  # Divide the circle into 9 positions (1-9)
    
    # Apply COM framework scaling with LZ constant
    radius = LZ ** layer
    
    # Apply phase function modulation with HQS constant
    phase = math.sin(4 * math.pi * n)
    radius *= (1 + HQS * phase)
    
    # Calculate 3D coordinates
    x = radius * math.cos(angle)
    y = radius * math.sin(angle)
    z = layer
    
    return (x, y, z)

# Generate the sequences
n_terms = 20
fib_seq = fibonacci(n_terms)
trib_seq = tribonacci(n_terms)
lucas_seq = lucas(n_terms)
pi_seq = pi_digits(n_terms)

# Reduce sequences to root numbers
fib_roots = [reduce_to_root(n) for n in fib_seq]
trib_roots = [reduce_to_root(n) for n in trib_seq]
lucas_roots = [reduce_to_root(n) for n in lucas_seq]
pi_roots = pi_seq  # Pi digits are already single digits

# Map sequences to 3D COM coordinates
fib_coords = [map_to_com_3d(n, i) for i, n in enumerate(fib_roots)]
trib_coords = [map_to_com_3d(n, i) for i, n in enumerate(trib_roots)]
lucas_coords = [map_to_com_3d(n, i) for i, n in enumerate(lucas_roots)]
pi_coords = [map_to_com_3d(n, i) for i, n in enumerate(pi_roots)]

# Create 3D visualization
fig = plt.figure(figsize=(15, 12))
ax = fig.add_subplot(111, projection='3d')

# Plot the sequences
ax.plot([x for x, y, z in fib_coords], [y for x, y, z in fib_coords], [z for x, y, z in fib_coords], 
        'o-', label='Fibonacci', linewidth=2)
ax.plot([x for x, y, z in trib_coords], [y for x, y, z in trib_coords], [z for x, y, z in trib_coords], 
        'o-', label='Tribonacci', linewidth=2)
ax.plot([x for x, y, z in lucas_coords], [y for x, y, z in lucas_coords], [z for x, y, z in lucas_coords], 
        'o-', label='Lucas', linewidth=2)
ax.plot([x for x, y, z in pi_coords], [y for x, y, z in pi_coords], [z for x, y, z in pi_coords], 
        'o-', label='Pi Digits', linewidth=2)

# Highlight the number 5 positions in each sequence
fib_5_indices = [i for i, n in enumerate(fib_roots) if n == 5]
trib_5_indices = [i for i, n in enumerate(trib_roots) if n == 5]
lucas_5_indices = [i for i, n in enumerate(lucas_roots) if n == 5]
pi_5_indices = [i for i, n in enumerate(pi_roots) if n == 5]

# Extract coordinates for number 5 occurrences
fib_5_coords = [fib_coords[i] for i in fib_5_indices]
trib_5_coords = [trib_coords[i] for i in trib_5_indices]
lucas_5_coords = [lucas_coords[i] for i in lucas_5_indices]
pi_5_coords = [pi_coords[i] for i in pi_5_indices]

# Plot number 5 occurrences with larger markers
if fib_5_coords:
    ax.scatter([x for x, y, z in fib_5_coords], [y for x, y, z in fib_5_coords], [z for x, y, z in fib_5_coords], 
               color='red', s=100, label='Fibonacci 5s')
if trib_5_coords:
    ax.scatter([x for x, y, z in trib_5_coords], [y for x, y, z in trib_5_coords], [z for x, y, z in trib_5_coords], 
               color='green', s=100, label='Tribonacci 5s')
if lucas_5_coords:
    ax.scatter([x for x, y, z in lucas_5_coords], [y for x, y, z in lucas_5_coords], [z for x, y, z in lucas_5_coords], 
               color='blue', s=100, label='Lucas 5s')
if pi_5_coords:
    ax.scatter([x for x, y, z in pi_5_coords], [y for x, y, z in pi_5_coords], [z for x, y, z in pi_5_coords], 
               color='purple', s=100, label='Pi 5s')

# Add reference circles for each layer
for layer in range(n_terms):
    theta = np.linspace(0, 2*np.pi, 100)
    radius = LZ ** layer
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    z = np.ones_like(theta) * layer
    ax.plot(x, y, z, color='gray', alpha=0.3)

# Add labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z (Layer)')
ax.set_title('Convergence of Number Sequences in 3D COM Framework\nFocusing on Number 5 Occurrences')
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

# Analyze the convergence at number 5
print("Analysis of Number 5 Occurrences in Sequences:")
print(f"Fibonacci sequence: {fib_5_indices} (positions where root number is 5)")
print(f"Tribonacci sequence: {trib_5_indices} (positions where root number is 5)")
print(f"Lucas sequence: {lucas_5_indices} (positions where root number is 5)")
print(f"Pi digits: {pi_5_indices} (positions where digit is 5)")

# Calculate distances between number 5 occurrences in 3D space
print("\nDistances between Number 5 Occurrences in 3D COM Space:")

# Function to calculate Euclidean distance between two 3D points
def distance_3d(p1, p2):
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)

# Analyze distances between all pairs of number 5 occurrences
all_5_coords = fib_5_coords + trib_5_coords + lucas_5_coords + pi_5_coords
all_5_labels = (["Fibonacci"]*len(fib_5_coords) + ["Tribonacci"]*len(trib_5_coords) + 
                ["Lucas"]*len(lucas_5_coords) + ["Pi"]*len(pi_5_coords))

for i in range(len(all_5_coords)):
    for j in range(i+1, len(all_5_coords)):
        dist = distance_3d(all_5_coords[i], all_5_coords[j])
        print(f"Distance between {all_5_labels[i]} 5 at layer {all_5_coords[i][2]} and " +
              f"{all_5_labels[j]} 5 at layer {all_5_coords[j][2]}: {dist:.4f}")

# Save the figure
plt.tight_layout()
plt.savefig('number_sequence_convergence.png', dpi=300)
plt.close()

# Create a 2D projection focusing on the number 5 positions
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111)

# Plot the 2D projections (x, y) of the sequences
ax.plot([x for x, y, z in fib_coords], [y for x, y, z in fib_coords], 'o-', label='Fibonacci', linewidth=2)
ax.plot([x for x, y, z in trib_coords], [y for x, y, z in trib_coords], 'o-', label='Tribonacci', linewidth=2)
ax.plot([x for x, y, z in lucas_coords], [y for x, y, z in lucas_coords], 'o-', label='Lucas', linewidth=2)
ax.plot([x for x, y, z in pi_coords], [y for x, y, z in pi_coords], 'o-', label='Pi Digits', linewidth=2)

# Highlight the number 5 positions in the 2D projection
if fib_5_coords:
    ax.scatter([x for x, y, z in fib_5_coords], [y for x, y, z in fib_5_coords], 
               color='red', s=100, label='Fibonacci 5s')
if trib_5_coords:
    ax.scatter([x for x, y, z in trib_5_coords], [y for x, y, z in trib_5_coords], 
               color='green', s=100, label='Tribonacci 5s')
if lucas_5_coords:
    ax.scatter([x for x, y, z in lucas_5_coords], [y for x, y, z in lucas_5_coords], 
               color='blue', s=100, label='Lucas 5s')
if pi_5_coords:
    ax.scatter([x for x, y, z in pi_5_coords], [y for x, y, z in pi_5_coords], 
               color='purple', s=100, label='Pi 5s')

# Add reference circles for each layer
for layer in range(5):  # Just show first 5 layers for clarity
    theta = np.linspace(0, 2*np.pi, 100)
    radius = LZ ** layer
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    ax.plot(x, y, color='gray', alpha=0.3)

# Add labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('2D Projection of Number Sequences in COM Framework\nFocusing on Number 5 Occurrences')
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)

# Save the 2D projection
plt.tight_layout()
plt.savefig('number_sequence_2d_projection.png', dpi=300)
plt.close()

# Create a visualization showing the 5-fold symmetry connection to quasicrystals
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Create a pentagonal pattern (5-fold symmetry) at each number 5 occurrence
def pentagon_3d(center, radius, layer):
    points = []
    for i in range(5):
        angle = 2 * np.pi * i / 5
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)
        z = layer
        points.append((x, y, z))
    # Add the first point again to close the pentagon
    points.append(points[0])
    return points

# Draw pentagons at each number 5 occurrence
pentagon_coords = []
for coords, label in zip([fib_5_coords, trib_5_coords, lucas_5_coords, pi_5_coords],
                         ['Fibonacci', 'Tribonacci', 'Lucas', 'Pi']):
    for center in coords:
        pentagon = pentagon_3d(center, 0.2, center[2])
        pentagon_coords.append((pentagon, label))
        x_vals = [p[0] for p in pentagon]
        y_vals = [p[1] for p in pentagon]
        z_vals = [p[2] for p in pentagon]
        ax.plot(x_vals, y_vals, z_vals, '-', linewidth=2)

# Plot connecting lines between pentagons to show quasicrystal-like structure
for i in range(len(pentagon_coords)):
    for j in range(i+1, len(pentagon_coords)):
        # Connect the centers of the pentagons
        center1 = pentagon_coords[i][0][0]  # First point of first pentagon
        center2 = pentagon_coords[j][0][0]  # First point of second pentagon
        ax.plot([center1[0], center2[0]], [center1[1], center2[1]], [center1[2], center2[2]], 
                'k-', alpha=0.3, linewidth=1)

# Add labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z (Layer)')
ax.set_title('5-Fold Symmetry in Number Sequences: Connection to Quasicrystals')

# Save the quasicrystal visualization
plt.tight_layout()
plt.savefig('number_sequence_quasicrystal.png', dpi=300)
plt.close()

print("\nAnalysis complete. Visualizations saved to:")
print("1. number_sequence_convergence.png - 3D visualization of sequence convergence")
print("2. number_sequence_2d_projection.png - 2D projection focusing on number 5 occurrences")
print("3. number_sequence_quasicrystal.png - 5-fold symmetry connection to quasicrystals")

# Additional analysis: Golden ratio connection
golden_ratio = (1 + np.sqrt(5)) / 2  # Approximately 1.618...
lz_ratio = LZ / 1  # LZ constant ratio

print("\nMathematical Relationship Analysis:")
print(f"Golden Ratio (φ): {golden_ratio:.6f}")
print(f"LZ Constant: {LZ:.6f}")
print(f"Ratio of LZ to 1: {lz_ratio:.6f}")
print(f"Ratio of Golden Ratio to LZ: {golden_ratio/LZ:.6f}")
print(f"LZ^2: {LZ**2:.6f}")
print(f"Golden Ratio - LZ: {golden_ratio-LZ:.6f}")

# Check if the number 5 has special properties in these relationships
print("\nNumber 5 Significance:")
print(f"√5 (appears in golden ratio formula): {np.sqrt(5):.6f}")
print(f"5 × LZ: {5*LZ:.6f}")
print(f"5 × HQS: {5*HQS:.6f}")
print(f"LZ^5: {LZ**5:.6f}")
print(f"5-fold symmetry angle (2π/5): {2*np.pi/5:.6f} radians or {360/5:.1f} degrees")
