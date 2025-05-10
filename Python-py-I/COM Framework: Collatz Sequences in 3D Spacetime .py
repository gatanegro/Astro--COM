import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# =================================================================
# Core Functions (No Changes Needed)
# =================================================================

def generate_collatz_sequence(n):
    sequence = [n]
    while n != 1:
        n = n // 2 if n % 2 == 0 else 3 * n + 1
        sequence.append(n)
    return sequence

def octave_reduction(value):
    return (value - 1) % 9 + 1

# =================================================================
# Simplified Visualization (No Input Required)
# =================================================================

def plot_com_spacetime():
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Predefined sequences to plot (no user input)
    numbers_to_plot = [7, 9, 15, 27]  # Interesting Collatz sequences

    for n in numbers_to_plot:
        seq = generate_collatz_sequence(n)
        x, y, z = [], [], []
        cumulative_phase = 0.0
        for layer, value in enumerate(seq):
            # Spatial coordinates
            rho_E = octave_reduction(value)
            radius = 1.0 + layer * 0.2  # Simple scaling
            angle = (rho_E / 9) * 2 * np.pi
            xi = radius * np.cos(angle)
            yi = radius * np.sin(angle)
            
            # Time from phase gradient
            if layer > 0:
                delta_rho = octave_reduction(seq[layer]) - octave_reduction(seq[layer-1])
                delta_phi = 2 * np.pi * delta_rho / (octave_reduction(seq[layer-1]) + 1e-9)
                cumulative_phase += delta_phi * 0.1  # Scaling for visualization
            
            x.append(xi)
            y.append(yi)
            z.append(cumulative_phase)
        
        ax.plot(x, y, z, marker='o', markersize=3, label=f'Collatz {n}')
    
    ax.set_title("COM Framework: Collatz Sequences in 3D Spacetime")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Local Time (Phase)")
    ax.legend()
    plt.show()
