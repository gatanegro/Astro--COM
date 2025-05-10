import numpy as np
import matplotlib.pyplot as plt

def collatz_turbulence_spectrum(k_max=100, init_energy=1.0):
    """
    Generates a turbulence energy spectrum using a Collatz-inspired energy transfer model.
    """
    k_values = np.arange(1, k_max + 1)
    E_k = np.zeros_like(k_values, dtype=float)
    E_k[0] = init_energy  # Initial energy at k=1
    
    # Energy transfer based on a Collatz-like process
    for k in range(1, k_max):
        if k % 2 == 0:
            next_k = k // 2
            E_k[next_k] += E_k[k] / 2  # Transfer energy downward
        else:
            next_k = 3 * k + 1
            if next_k < k_max:
                E_k[next_k] += E_k[k] / 2  # Transfer energy upward
        
        E_k[k] *= 0.5  # Decay energy at current scale
    
    return k_values, E_k

# Generate spectrum using Collatz model
k_values, E_k_collatz = collatz_turbulence_spectrum()

# Kolmogorov -5/3 Scaling for comparison
k_range = np.linspace(1, 100, 100)
E_k_Kolmogorov = k_range ** (-5/3)
E_k_Kolmogorov /= E_k_Kolmogorov.max()  # Normalize for comparison

# Plot Comparison
plt.figure(figsize=(8, 6))
plt.loglog(k_values, E_k_collatz, 'o-', label="Collatz Turbulence Spectrum", color='b')
plt.loglog(k_range, E_k_Kolmogorov, '--', label="Kolmogorov $k^{-5/3}$ Scaling", color='r')

plt.xlabel("Wavenumber $|k|$")
plt.ylabel("Energy Spectrum $E(k)$")
plt.title("Comparison of Collatz Turbulence Model and Kolmogorov Scaling")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.show()