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