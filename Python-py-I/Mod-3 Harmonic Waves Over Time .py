import numpy as np
import matplotlib.pyplot as plt

# Function to generate harmonic scaling based on mod-3 periodicity
def harmonic_mod3_series(start, length):
    sequence = [start]
    for _ in range(length - 1):
        if sequence[-1] % 3 == 0:
            sequence.append(sequence[-1] // 3)  # Collapse inward if multiple of 3
        else:
            sequence.append(3 * sequence[-1] + 1)  # Expand outward if not multiple of 3
    return sequence

# Generate sequences for different starting points
harmonic_series_3 = harmonic_mod3_series(3, 20)
harmonic_series_9 = harmonic_mod3_series(9, 20)
harmonic_series_27 = harmonic_mod3_series(27, 20)

# Plot the harmonic sequences
plt.figure(figsize=(10, 6))
plt.plot(harmonic_series_3, marker='o', label="Start: 3")
plt.plot(harmonic_series_9, marker='s', label="Start: 9")
plt.plot(harmonic_series_27, marker='^', label="Start: 27")

plt.xlabel("Iteration Step")
plt.ylabel("Harmonic Value (Scaling)")
plt.title("Harmonic Mod-3 Scaling in Oscillatory Field")
plt.legend()
plt.grid(True)
plt.show()
# Simulating mod-3 periodic wave interactions

# Define wave parameters
x = np.linspace(0, 10, 400)  # Space variable
t_values = np.linspace(0, 10, 5)  # Time steps

# Create wave functions mod-3
def mod3_wave(x, t, k=1, omega=1):
    return np.sin(k * x - omega * t) + np.sin(3 * k * x - 3 * omega * t)

# Plot the wave evolution
plt.figure(figsize=(10, 6))
for t in t_values:
    y = mod3_wave(x, t)
    plt.plot(x, y, label=f"t={t:.1f}")

plt.xlabel("Space (x)")
plt.ylabel("Wave Amplitude")
plt.title("Mod-3 Harmonic Waves Over Time")
plt.legend()
plt.grid(True)
plt.show()