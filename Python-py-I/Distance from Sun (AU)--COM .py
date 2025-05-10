import numpy as np
import matplotlib.pyplot as plt

# Constants
LZ = 1.23498
HQS = 0.235
a0_sun = 0.0  # Sun as attractor (0 AU)

# Revised COM equation (Sun-centered)
def com_orbit_sun(n):
    return (LZ ** n) * (1 + HQS * np.tanh(n/2))  # No a0 term

# Known solar system orbits (AU)
observed_planets = {
    "Mercury": 0.387,
    "Venus": 0.723,
    "Earth": 1.000,
    "Mars": 1.524,
    "Jupiter": 5.203,
    "Saturn": 9.537,
    "Uranus": 19.191,
    "Neptune": 30.069,
    "Pluto": 39.482
}

# Generate predictions for n=1 to n=10
n_values = np.arange(1, 11)
predicted_orbits = [com_orbit_sun(n) for n in n_values]

# Plot
plt.figure(figsize=(12, 6))
plt.title("COM Model: Sun as Attractor (No Mercury Baseline)", fontsize=14)

# Observed planets
for planet, orbit in observed_planets.items():
    plt.scatter(orbit, 0, label=planet, s=100)

# COM predictions (red X's)
plt.scatter(predicted_orbits, [0.5]*len(predicted_orbits), 
            color='red', marker='x', s=100, label='COM Predictions (Sun)')

# Label mismatches
for i, (n, orbit) in enumerate(zip(n_values, predicted_orbits)):
    closest_planet = min(observed_planets.values(), key=lambda x: abs(x - orbit))
    error_pct = 100 * (orbit - closest_planet) / closest_planet
    if abs(error_pct) > 20:  # Highlight large deviations
        plt.annotate(f"n={n}\n{orbit:.2f} AU\n({error_pct:.0f}% error)", 
                     (orbit, 0.5), textcoords="offset points", xytext=(0,10), ha='center')

plt.xlabel("Distance from Sun (AU)", fontsize=12)
plt.yticks([])
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)
plt.show()