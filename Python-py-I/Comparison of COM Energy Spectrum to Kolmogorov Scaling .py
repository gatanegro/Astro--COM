import numpy as np
import matplotlib.pyplot as plt

# Parameters
nu = 0.1  # Viscosity
k = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])  # Wavenumbers
A = np.ones_like(k, dtype=float)  # Initial amplitudes
dt = 0.01
steps = 1000

# Collatz-like energy cascade simulation
for _ in range(steps):
    for i in range(len(k)):
        if k[i] % 2 == 0:
            A[i] = 0.5 * A[i] + (nu * k[i]**2) / 3
        else:
            A[i] = 3 * A[i] + nu * k[i]**2
    A = np.clip(A, -1e3, 1e3)  # Keep amplitudes bounded

# Compute energy spectrum: E(k) = |A_k|^2
E_k_COM = np.abs(A)**2

# Kolmogorov -5/3 scaling for comparison
k_range = np.linspace(1, 10, 100)
E_k_Kolmogorov = k_range**(-5/3)  # Theoretical scaling

# Plot results
plt.figure(figsize=(8, 6))
plt.loglog(k, E_k_COM, 'o-', label="COM Energy Spectrum")
plt.loglog(k_range, E_k_Kolmogorov, '--', label="Kolmogorov $k^{-5/3}$", color='r')
plt.xlabel("Wavenumber $k$")
plt.ylabel("Energy Spectrum $E(k)$")
plt.title("Comparison of COM Energy Spectrum to Kolmogorov Scaling")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.show()
