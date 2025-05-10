import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Constants (Atomic-like energy scaling)
LZ = 1.23498      # Poincaré-derived amplitude (binding energy factor)
HQS = 0.235       # Energy shift rate (quantum state coupling)

def collatz_energy(n):
    sequence = [n]
    energy = [LZ * n]  # Initial energy scaled by LZ
    while n != 1:
        n = n // 2 if n % 2 == 0 else 3 * n + 1
        sequence.append(n)
        energy.append(energy[-1] * HQS + LZ * n)  # HQS energy shift
    return sequence, energy

# Generate data for elements 1-20 (atomic number analogy)
elements = {}
for atomic_number in range(1, 21):
    seq, energy = collatz_energy(atomic_number)
    elements[atomic_number] = {'sequence': seq, 'energy': energy}

# 3D Plot: Energy orbitals of "Collatz atoms"
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot each "element" as an energy orbital
for an, data in elements.items():
    x, y, z = [], [], []
    for step, (n, e) in enumerate(zip(data['sequence'], data['energy'])):
        # Spherical coordinates scaled by energy
        theta = 2 * np.pi * (step / len(data['sequence']))
        phi = np.arccos(1 - 2 * n / max(data['sequence']))
        r = e * 0.1  # Energy radius scaling
        
        x.append(r * np.sin(phi) * np.cos(theta))
        y.append(r * np.sin(phi) * np.sin(theta))
        z.append(r * np.cos(phi))
    
    ax.plot(x, y, z, label=f'Element {an}', alpha=0.7)
    ax.scatter(x, y, z, s=[50*e for e in data['energy']], c=data['energy'], 
               cmap='viridis', depthshade=False)

# Style
ax.set_title("Collatz Atomic Model: Quantum Energy Orbitals (LZ/HQS)")
ax.set_xlabel("X (Electron Cloud Density)")
ax.set_ylabel("Y (Spin Axis)")
ax.set_zlabel("Z (Nuclear Binding Layer)")
plt.colorbar(plt.cm.ScalarMappable(cmap='viridis'), ax=ax, label='Energy State')

plt.legend()
plt.show()