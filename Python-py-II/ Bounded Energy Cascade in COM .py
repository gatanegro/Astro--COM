import numpy as np
import matplotlib.pyplot as plt

# Parameters
nu = 0.1  # Viscosity
k_values = np.array([1, 2, 3, 4, 5])  # Wavenumbers
A = np.ones(len(k_values))  # Initial amplitudes
dt = 0.01
steps = 1000

# Collatz-Octave Energy Cascade
for _ in range(steps):
    for i, k in enumerate(k_values):
        if k % 2 == 0:
            A[i] = 0.5 * A[i] + (nu * k**2) / 3
        else:
            A[i] = 3 * A[i] + nu * k**2
    # Clip to avoid overflow
    A = np.clip(A, -1e3, 1e3)

# Plot energy spectrum
plt.plot(k_values, A, 'o-', label='COM Amplitudes')
plt.xlabel('Wavenumber $k$')
plt.ylabel('$|A_k|$')
plt.title('Bounded Energy Cascade in COM')
plt.legend()
plt.show()