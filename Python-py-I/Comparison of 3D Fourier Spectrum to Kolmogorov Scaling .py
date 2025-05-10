import numpy as np
import matplotlib.pyplot as plt

# Parameters
nu = 0.1  # Viscosity
k_vectors = np.array([[1,1,1], [2,2,2], [3,3,3], [4,4,4], [5,5,5], [6,6,6], [7,7,7], [8,8,8], [9,9,9], [10,10,10]])  # 3D wavevectors
A = np.ones(len(k_vectors), dtype=float)  # Initial amplitudes
dt = 0.01
steps = 1000

# Compute wave properties
omega_k = np.linalg.norm(k_vectors, axis=1)  # Frequency ω_k = |k|
gamma_k = nu * omega_k**2  # Dissipation factor γ_k

# Evolve amplitudes with dissipation
for _ in range(steps):
    A *= np.exp(-gamma_k * dt)  # Apply dissipation

# Compute energy spectrum
E_k = np.abs(A)**2

# Kolmogorov -5/3 scaling for comparison
k_range = np.linspace(1, 10, 100)
E_k_Kolmogorov = k_range**(-5/3)  # Theoretical scaling

# Plot results
plt.figure(figsize=(8, 6))
plt.loglog(omega_k, E_k, 'o-', label="3D Fourier Energy Spectrum")
plt.loglog(k_range, E_k_Kolmogorov, '--', label="Kolmogorov $k^{-5/3}$", color='r')
plt.xlabel("Wavenumber |k|")
plt.ylabel("Energy Spectrum E(k)")
plt.title("Comparison of 3D Fourier Spectrum to Kolmogorov Scaling")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.show()
