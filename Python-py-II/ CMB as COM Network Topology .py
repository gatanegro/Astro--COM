# ==== IMPORTS ====
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

# ==== SIMULATION 1: FUTURE STAR OBSERVATIONS ====
rho_local = 1.0  # Local energy density
r_observed = np.linspace(1, 100, 100)  # Distance from observer
rho_E = rho_local * (1 + r_observed**3)  # Energy accumulates with distance
time_perceived = np.log(rho_E)  # Future appears logarithmic

plt.figure(figsize=(10, 4))
plt.plot(r_observed, time_perceived, 'r-')
plt.xlabel("Distance (COM units)"); plt.ylabel("Perceived Time (log scale)")
plt.title("Distant Objects are Future States")
plt.grid()
plt.show()

# ==== SIMULATION 2: ENERGY NODE-FIELD EQUILIBRIUM ====
rho_node = 5.0  # Node energy
rho_field = np.linspace(0.1, 10, 100)
visibility = np.abs(rho_node - r_field)  # Visibility = energy gradient

plt.figure(figsize=(10, 4))
plt.plot(rho_field, visibility, 'b-')
plt.axvline(x=rho_node, color='k', linestyle='--', label="Equilibrium (Invisible)")
plt.xlabel("Field Energy Density"); plt.ylabel("Visibility (|Δρ|)")
plt.legend(); plt.grid()
plt.show()

# ==== SIMULATION 3: CMB AS COM TOPOLOGY ====
cmb = np.random.normal(loc=0, scale=0.1, size=(100, 100))  # Cosmic fluctuations
cmb = gaussian_filter(cmb, sigma=5)  # Smooth with COM scaling

plt.imshow(cmb, cmap='jet')
plt.title("CMB as COM Network Topology")
plt.colorbar()
plt.show()