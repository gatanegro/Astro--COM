import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Constants from your theory
LZ = 1.23498        # Poincaré limit cycle (S^3 collapse)
HQS = 0.235         # Ricci curvature energy redistribution rate

# Recursive wave evolution to derive LZ (validating stabilization)
def recursive_wave(n_iter=100):
    psi = 1.0  # Initial condition
    for _ in range(n_iter):
        psi = np.sin(psi) + np.exp(-psi)
    return psi  # Stabilizes to LZ

assert np.isclose(recursive_wave(), LZ, atol=1e-5), "LZ validation failed!"

# Energy of elements (atomic numbers 1-20) using Ricci-HQS coupling
def element_energy(Z, HQS):
    # Ricci curvature term: HQS * Z^2 (energy redistribution)
    return LZ * Z * (1 + HQS * Z)  # LZ scaled, HQS curvature-modulated

# CCOM spiral coordinates (Poincaré S^3 projection)
def ccom_spiral(Z, energy):
    theta = HQS * 2 * np.pi * Z  # Angular phase (Ricci curvature shift)
    radius = LZ * np.log(Z + 1)  # Radial scaling (Poincaré limit)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    z = energy  # Energy axis (LZ+HQS)
    return x, y, z

# Generate data for elements 1-20
atomic_numbers = np.arange(1, 21)
energies = [element_energy(Z, HQS) for Z in atomic_numbers]
positions = [ccom_spiral(Z, energy) for Z, energy in zip(atomic_numbers, energies)]

# Plot
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot elements on CCOM spiral
x, y, z = zip(*positions)
scatter = ax.scatter(x, y, z, 
                    s=100 * np.array(energies)/max(energies),  # Size ~ energy
                    c=energies, 
                    cmap='inferno', 
                    depthshade=False)

# Add Poincaré S^3 reference (limit cycle)
phi = np.linspace(0, 2*np.pi, 100)
theta = np.linspace(0, np.pi, 100)
x_sphere = LZ * np.outer(np.cos(phi), np.sin(theta))
y_sphere = LZ * np.outer(np.sin(phi), np.sin(theta))
z_sphere = LZ * np.outer(np.ones(100), np.cos(theta))
ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.1, color='gray', label='Poincaré S^3')

# Add Ricci curvature vectors (HQS)
for Z, (x_i, y_i, z_i) in zip(atomic_numbers, positions):
    dx = -HQS * y_i  # Ricci flow direction
    dy = HQS * x_i
    ax.quiver(x_i, y_i, z_i, dx, dy, 0, color='green', length=0.5, label='Ricci Flow' if Z == 1 else None)

ax.set_title(f"CCOM Spiral: Poincaré LZ={LZ} + Ricci HQS={HQS}", fontsize=14)
ax.set_xlabel("X (Poincaré Radius)")
ax.set_ylabel("Y (Ricci Phase)")
ax.set_zlabel("Z (Element Energy)")
plt.colorbar(scatter, label='Energy (LZ * Z * (1 + HQS*Z))')
ax.legend()
plt.show()