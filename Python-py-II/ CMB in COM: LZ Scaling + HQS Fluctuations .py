import numpy as np
import matplotlib.pyplot as plt

# Constants
a0 = 2.725           # CMB baseline (K)
LZ = 1.23498         # Scaling constant
HQS = 0.00235        # Fluctuation strength
theta_n = lambda n: 4 * n * np.pi  # Phase term

def com_cmb(n):
    """Compute CMB temperature at step `n`."""
    return a0 * (LZ ** n) * (1 + HQS * np.sin(theta_n(n)))

# Generate CMB sky (512x512 grid)
size = 512
n_values = np.linspace(0, 1, size)  # Steps (normalized)
x, y = np.meshgrid(n_values, n_values)
cmb_sky = com_cmb(x * y)  # 2D temperature map

# Plot
plt.figure(figsize=(10, 8))
plt.imshow(cmb_sky, cmap='viridis', extent=[0, 1, 0, 1])
plt.colorbar(label='Temperature (K)')
plt.title('CMB in COM: LZ Scaling + HQS Fluctuations')
plt.show()