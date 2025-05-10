import numpy as np
import matplotlib.pyplot as plt

# Parameters
nu = 0.1  # Viscosity
k = np.array([1, 2, 3, 4, 5])  # Wavenumbers
A = np.ones_like(k, dtype=float)  # Initial amplitudes
dt = 0.01
steps = 1000

# COM Model Evolution (Collatz-like energy cascade)
for _ in range(steps):
    for i in range(len(k)):
        if k[i] % 2 == 0:
            A[i] = 0.5 * A[i] + (nu * k[i]**2) / 3
        else:
            A[i] = 3 * A[i] + nu * k[i]**2
    # Ensure boundedness
    A = np.clip(A, -1e3, 1e3)

# Compute energy spectrum for COM model
E_com = A**2  # Energy is proportional to |A_k|^2

# Viscous homogeneous turbulence model
k_viscous = np.linspace(1, 5, 100)
E_viscous = (10 * k_viscous**(-5/3)) * np.exp(-nu * k_viscous**2)  # Kolmogorov scaling + viscous decay

# Kolmogorov -5/3 reference line (pure scaling, no viscosity)
E_kolm = 10 * k_viscous**(-5/3)

# Plot results
plt.figure(figsize=(8, 6))
plt.loglog(k, E_com, 'o-', label='COM Model $|A_k|^2$', markersize=6)
plt.loglog(k_viscous, E_viscous, '-', label='Viscous Homogeneous Data', linewidth=2, color='green')
plt.loglog(k_viscous, E_kolm, '--', label=r'Kolmogorov $k^{-5/3}$', color='red')

plt.xlabel('Wavenumber $k$')
plt.ylabel('Energy $E(k)$')
plt.title('Energy Spectrum Comparison: COM vs. Viscous Homogeneous')
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.show()
