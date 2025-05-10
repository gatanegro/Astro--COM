import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def collatz_3d_energy_step(energy_grid):
    """Applies Collatz-like rules to a 3D energy field."""
    new_grid = np.zeros_like(energy_grid)
    for x, y, z in np.ndindex(energy_grid.shape):
        e = energy_grid[x, y, z]
        if e % 2 == 0:  # Split/decay
            new_grid[x, y, z] = e / 2
        else:  # Merge/amplify
            new_grid[x, y, z] = 3 * e + 1 + np.sum(energy_grid[max(0,x-1):x+2, ...])  # Local feedback
    return new_grid

def run_com_simulation(steps=100, size=50):
    """Runs the energy recursion and visualizes stable loops."""
    grid = np.random.rand(size, size, size)  # Initial chaotic energy
    for _ in range(steps):
        grid = collatz_3d_energy_step(grid)
        # Detect stable loops (particles)
        particles = np.where(grid > threshold)  # Tune threshold
    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(*particles, s=1, alpha=0.5)
    plt.show()