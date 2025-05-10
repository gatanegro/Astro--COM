import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Constants
LZ = 1.23498  # Your scaling constant

# 1. Octave reduction (maps numbers 1-9)
def octave(n):
    return (n - 1) % 9 + 1

# 2. Collatz sequence generator
def collatz(n):
    path = []
    while n != 1:
        path.append(n)
        n = n // 2 if n % 2 == 0 else 3 * n + 1
    path.append(1)
    return path

# 3. Main visualization
def plot_com():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    for n in range(1, 21):  # First 20 energy states
        path = collatz(n)
        for layer, energy in enumerate(path):
            phase = (octave(energy) / 9) * 2 * np.pi  # Map to 0-2π
            r = np.log(energy + 1)  # Amplitude
            
            x = r * np.cos(phase)
            y = r * np.sin(phase)
            z = layer
            
            ax.scatter(x, y, z, c=phase, s=50)
    
    ax.set_title("COM Energy States")
    plt.show()

plot_com()  # Run the visualization