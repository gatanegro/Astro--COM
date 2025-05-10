import numpy as np
import matplotlib.pyplot as plt

# Parameters
nu = 0.1  # Viscosity
k = np.array([1, 2, 3, 4, 5])  # Wavenumbers
A = np.ones_like(k, dtype=float)  # Initial amplitudes
dt = 0.01
steps = 1000

# Collatz-like energy cascade
for _ in range(steps):
    for i in range(len(k)):
        if k[i] % 2 == 0:
            A[i] = 0.5 * A[i] + (nu * k[i]**2) / 3
        else:
            A[i] = 3 * A[i] + nu * k[i]**2
    # Ensure boundedness
    A = np.clip(A, -1e3, 1e3)

# Compute energy spectrum
E_k = A**2  # Energy is proportional to |A_k|^2

# Kolmogorov -5/3 reference line
k_ref = np.linspace(1, 5, 100)
E_ref = 10 * k_ref**(-5/3)  # Adjust scaling factor for comparison

# Plot results
plt.figure(figsize=(8, 6))
plt.loglog(k, E_k, 'o-', label='Model Energy $|A_k|^2$')
plt.loglog(k_ref, E_ref, '--', label=r'Kolmogorov $k^{-5/3}$', color='red')

plt.xlabel('Wavenumber $k$')
plt.ylabel('Energy $E(k)$')
plt.title('Energy Spectrum Comparison')
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.show()
