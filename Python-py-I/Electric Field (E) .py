import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# Parameters
L = 64  # Grid size
kappa = 0.117  # Collatz-Zeta constant

# Initialize Collatz nodes (n) and energy density (ρ_E)
n_grid = np.random.randint(1, 100, (L, L))
rho_E = np.zeros((L, L))

def collatz_steps(n):
    steps = 0
    while n != 1:
        n = n // 2 if n % 2 == 0 else 3 * n + 1
        steps += 1
    return steps

# Compute ρ_E for each node
for i in range(L):
    for j in range(L):
        S = (n_grid[i, j] - 1) % 9 + 1  # Octave reduction
        k = collatz_steps(n_grid[i, j])
        rho_E[i, j] = kappa * k * S

# Smooth ρ_E to simulate field continuity
rho_E = gaussian_filter(rho_E, sigma=2)

# Compute E-field (negative gradient of ρ_E)
Ey, Ex = np.gradient(-rho_E)

# Plot
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.imshow(rho_E, cmap='inferno')
plt.title("Energy Density (ρ_E)")
plt.colorbar()

plt.subplot(122)
plt.streamplot(np.arange(L), np.arange(L), Ex, Ey, color='white')
plt.title("Electric Field (E)")
plt.show()