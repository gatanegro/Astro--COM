import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import expm

# ====================== Collatz Lattice Functions ======================
def generate_collatz(n):
    sequence = [n]
    while n != 1:
        n = n//2 if n%2 == 0 else 3*n + 1
        sequence.append(n)
    return sequence

def digital_root(n):
    return (n-1)%9 + 1

def collatz_to_lattice(sequence):
    positions = []
    for layer, value in enumerate(sequence):
        dr = digital_root(value)
        x = (dr-1)%3 + 0.5*(layer%2)
        y = (dr-1)//3 + 0.5*((layer//2)%2)
        z = layer * 2.0
        positions.append((x,y,z))
    return np.array(positions)

# ====================== Hydrogen HCP Lattice ======================
def hcp_lattice(layers):
    a = 1.0  # Base spacing
    c = np.sqrt(8/3)*a  # Height spacing
    positions = []
    for layer in range(layers):
        offset = 0.5*(layer%2)
        for i in range(3):
            for j in range(3):
                x = i*a + offset
                y = j*a*np.sqrt(3)/2 + offset
                z = layer*c
                positions.append((x,y,z))
    return np.array(positions)

# ====================== Quantum Analysis ======================
def create_hamiltonian(points):
    n = len(points)
    H = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dist = np.linalg.norm(points[i]-points[j])
                H[i,j] = np.exp(-dist**2/0.5)  # Gaussian coupling
    return H

def quantum_walk(H, initial_state, time_steps):
    U = expm(-1j*H*time_steps)
    return U @ initial_state

# ====================== Generate Data ======================
collatz_data = {n: collatz_to_lattice(generate_collatz(n)) for n in range(1,21)}
h2_structure = hcp_lattice(5)

# Quantum system setup
collatz_points = np.concatenate(list(collatz_data.values()))
initial_state = np.zeros(len(collatz_points))
initial_state[len(collatz_points)//2] = 1  # Central position
H = create_hamiltonian(collatz_points)
psi = quantum_walk(H, initial_state, 2.0)

# ====================== Visualization ======================
fig = plt.figure(figsize=(20,6))

# Collatz Lattice
ax1 = fig.add_subplot(131, projection='3d')
for path in collatz_data.values():
    ax1.plot(*path.T, alpha=0.4)
ax1.set_title("Collatz Computational Lattice")

# Hydrogen Lattice
ax2 = fig.add_subplot(132, projection='3d')
ax2.scatter(*h2_structure.T, c='blue', s=50)
ax2.set_title("Hydrogen HCP Lattice")

# Quantum Probability
ax3 = fig.add_subplot(133, projection='3d')
sc = ax3.scatter(*collatz_points.T, c=np.abs(psi)**2, cmap='viridis', s=50)
plt.colorbar(sc, label='Probability Density')
ax3.set_title("Quantum Walk on Collatz Lattice")

plt.tight_layout()
plt.show()
