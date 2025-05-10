import numpy as np
import matplotlib.pyplot as plt

# Parameters
LZ = 1.23498
HQS = 0.235
layers = 5
nodes = 20
energy = np.zeros((layers, nodes))
energy[-1, :] = 100  # Initialize outer layer (L5)

def collatz_step(phi_current, is_even):
    if is_even:
        phi_next = (phi_current / 2) * LZ * (1 - HQS)
    else:
        phi_next = (3 * phi_current + 1) * LZ * (1 - HQS)
    return phi_next

for layer in range(layers-2, -1, -1):  # From L5 to L1
    for node in range(nodes):
        current_phi = energy[layer+1, node]
        is_even = (node % 2 == 0)
        energy[layer, node] = collatz_step(current_phi, is_even)
        # Redistribute HQS to neighbors
        energy[layer, (node+1) % nodes] += current_phi * HQS / 2
        energy[layer, (node-1) % nodes] += current_phi * HQS / 2

plt.imshow(energy, cmap='viridis', aspect='auto')
plt.colorbar(label='Energy (Φ)')
plt.xlabel('Collatz Nodes')
plt.ylabel('Octave Layers')
plt.title('Energy Redistribution in Collatz-Octave Lattice')
plt.show()

total_energy = np.sum(energy, axis=(0,1))
plt.plot(total_energy)
plt.xlabel('Iterations')
plt.ylabel('Total Energy')
plt.title('Energy Conservation (Divergence?)')
plt.show()