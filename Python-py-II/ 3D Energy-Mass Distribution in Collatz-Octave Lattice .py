import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parameters
LZ = 1.23498
HQS = 0.235
layers = 5
nodes = 20
iterations = 50
energy = np.zeros((layers, nodes))
energy[-1, :] = 100  # Outer layer (L5) initialized

# Track energy flow and spacetime metrics
time_emergent = []
space_emergent = []
mass_node1 = []

for it in range(iterations):
    new_energy = np.zeros_like(energy)
    for layer in range(layers-1, 0, -1):  # From outer to inner layers
        for node in range(nodes):
            current_phi = energy[layer, node]
            if current_phi == 0:
                continue
            # Collatz rule: Even or odd?
            is_even = (current_phi % 2 == 0)
            if is_even:
                phi_next = (current_phi / 2) * LZ * (1 - HQS)
            else:
                phi_next = (3 * current_phi + 1) * LZ * (1 - HQS)
            # Assign energy to next layer
            new_energy[layer-1, node] += phi_next
            # Redistribute HQS to adjacent nodes (directional flow)
            left_node = (node - 1) % nodes
            right_node = (node + 1) % nodes
            new_energy[layer, left_node] += current_phi * HQS * 0.5
            new_energy[layer, right_node] += current_phi * HQS * 0.5
    energy = new_energy.copy()
    
    # Track emergent properties
    mass_node1.append(np.sum(energy[:, 0]))  # Mass at Node 1 (center)
    frequency = it + 1  # Crude proxy for field frequency
    time_emergent.append(frequency / LZ)
    space_emergent.append(np.max(energy) * LZ)  # Amplitude = max(energy)
# Plot 1: 3D Energy Distribution
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
x, y = np.meshgrid(range(nodes), range(layers))
ax.plot_surface(x, y, energy, cmap='viridis')
ax.set_xlabel('Collatz Nodes')
ax.set_ylabel('Octave Layers')
ax.set_zlabel('Energy (Φ)')
plt.title('3D Energy-Mass Distribution in Collatz-Octave Lattice')
plt.show()

# Plot 2: Emergent Spacetime
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(time_emergent, mass_node1, 'b-')
ax1.set_xlabel('Emergent Time (t = f/LZ)')
ax1.set_ylabel('Mass at Node 1')
ax1.set_title('Mass Formation Over Time')

ax2.plot(time_emergent, space_emergent, 'r-')
ax2.set_xlabel('Emergent Time (t = f/LZ)')
ax2.set_ylabel('Emergent Space (x = A*LZ)')
ax2.set_title('Space Expansion Over Time')
plt.tight_layout()
plt.show()