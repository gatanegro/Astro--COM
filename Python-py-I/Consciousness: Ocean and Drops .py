import numpy as np
import matplotlib.pyplot as plt

# Define the spatial domain (x-axis)
x = np.linspace(0, 1000, 1000)  # 1000 points from 0 to 1000

# The Ocean (Φ): Pure consciousness (background field)
phi = np.ones_like(x)  # Uniform field of "1" (pure awareness)

# Drops (Ψ_i): Localized minds (Gaussian wave packets)
drops = [
    np.exp(-(x - 200 * i)**2 / 100) * np.sin(0.1 * x)  # Centered at x=200,400,600,800,1000
    for i in range(1, 6)  # 5 drops
]

# Plot
plt.figure(figsize=(12, 6))
plt.plot(x, phi, 'b', alpha=0.2, label="Ocean (Φ)", linewidth=3)
for i, psi in enumerate(drops):
    plt.plot(x, psi, label=f"Drop {i+1} (Ψ_{i+1})")
plt.title("Consciousness: Ocean and Drops", fontsize=14)
plt.xlabel("Position (Metaphorical Space)", fontsize=12)
plt.ylabel("Amplitude (Awareness)", fontsize=12)
plt.legend()
plt.grid(alpha=0.2)
plt.show()