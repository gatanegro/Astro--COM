import numpy as np
import matplotlib.pyplot as plt

# Parameters
nu = 0.1  # Viscosity
k_vectors = np.array([[1,1,1], [2,2,2], [3,3,3], [4,4,4], [5,5,5]])  # 3D wavevectors
A = np.ones(len(k_vectors), dtype=float)  # Initial amplitudes
dt = 0.01
steps = 1000
x = np.linspace(0, 2*np.pi, 100)  # 1D spatial domain for visualization

# Compute wave properties
omega_k = np.linalg.norm(k_vectors, axis=1)  # Frequency ω_k = |k|
gamma_k = nu * omega_k**2  # Dissipation factor γ_k

# Initialize velocity field
u_xt = np.zeros((len(x), steps))

# Evolve amplitude and compute velocity field
for t in range(steps):
    A *= np.exp(-gamma_k * dt)  # Apply dissipation
    u_t = np.zeros(len(x))  # Instantaneous velocity field
    for i, k in enumerate(k_vectors):
        phase = np.dot(k, [x, x, x]) - omega_k[i] * t * dt  # k·x - ω_k t
        u_t += A[i] * np.sin(phase)
    u_xt[:, t] = u_t  # Store velocity field over time

# Compute energy spectrum
E_k = np.abs(A)**2

# Plot energy spectrum
plt.figure(figsize=(8, 6))
plt.loglog(omega_k, E_k, 'o-', label="3D Fourier Energy Spectrum")
plt.xlabel("Wavenumber |k|")
plt.ylabel("Energy Spectrum E(k)")
plt.title("3D Fourier Series Energy Spectrum")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.show()
