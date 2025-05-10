import numpy as np
import matplotlib.pyplot as plt
# Constants
m = 9.11e-31 # Electron mass (kg)
hbar = 1.05e-34 # Reduced Planck's constant (J·s)
LZ = 1.23498 # Universal scaling factor
d = 2.0e-9 # Barrier width (meters)
alpha = np.sqrt(2 * m / hbar**2) * LZ
# Voltage range (0 to 1 V)
voltages = np.linspace(0, 1, 100) # Volts
V = voltages * 1.6e-19 # Convert to Joules
# Tunneling probability and current
tunneling_prob = np.exp(-alpha * d * np.sqrt(V))
current = tunneling_prob * voltages
# Plot
plt.figure(figsize=(8, 4))
plt.plot(voltages, current, 'b-', linewidth=2)
plt.xlabel('Voltage (V)')
plt.ylabel('Current (arb. units)')
plt.title('Diode Tunneling Current with COM Scaling (LZ = 1.23498)')
plt.grid(True)
plt.show()