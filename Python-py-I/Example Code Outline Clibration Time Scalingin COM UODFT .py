import numpy as np
import matplotlib.pyplot as plt

# Define a sample energy density function (e.g., as a function of height in a gravitational field)
def energy_density(h, rho0=1.0, alpha=1e-7):
    # A simple exponential drop-off in energy density with height h
    return rho0 * np.exp(-alpha * h)

# Define height range (e.g., from ground level to satellite altitude)
h = np.linspace(0, 10000, 1000)  # in meters

# Compute energy density and its logarithmic derivative
rho_E = energy_density(h)
log_rho = np.log(rho_E)
dlog_rho = np.gradient(log_rho, h)

# Calculate phase differential dphi = 2π d(log(rho_E))
dphi = 2 * np.pi * dlog_rho
phi = np.cumsum(dphi)  # Integrated phase

# Map phase to time: assuming one full cycle (2π rad) is one unit time (say, 1 second for illustration)
T_unit = 1.0  # 1 second per cycle
T = phi / (2 * np.pi) * T_unit

# Plot results
plt.figure(figsize=(10,5))
plt.subplot(2,1,1)
plt.plot(h, rho_E, label='Energy Density')
plt.xlabel("Height (m)")
plt.ylabel("ρ_E")
plt.legend()

plt.subplot(2,1,2)
plt.plot(h, T, label='Local Time Shift (s)', color='r')
plt.xlabel("Height (m)")
plt.ylabel("Time Shift (s)")
plt.legend()
plt.tight_layout()
plt.show()
