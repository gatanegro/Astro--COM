import numpy as np
import matplotlib.pyplot as plt

# Constants
LZ = 1.23498
HQS = 0.235

# Hybrid COM equation
def com_orbit_hybrid(n):
    if n <= 4:  # Inner: Sun as attractor
        return (LZ ** n) * (1 + HQS * np.tanh(n/2))
    else:       # Outer: Mercury as baseline
        return 0.387 * (LZ ** (n-4)) * (1 + HQS * np.tanh((n-4)/2))

# Observed solar system orbits (AU)
observed_planets = {
    "Mercury": 0.387, "Venus": 0.723, "Earth": 1.000, "Mars": 1.524,
    "Jupiter": 5.203, "Saturn": 9.537, "Uranus": 19.191, "Neptune": 30.069,
    "Pluto": 39.482, "Eris": 67.668
}

# Generate predictions for n=1 to n=12
n_values = np.arange(1, 13)
predicted_orbits = [com_orbit_hybrid(n) for n in n_values]

# Plot
plt.figure(figsize=(14, 7))
plt.title("Hybrid COM Model: Sun (n≤4) + Mercury (n>4) as Attractors", fontsize=16)

# Observed planets
for planet, orbit in observed_planets.items():
    plt.scatter(orbit, 0, label=planet, s=100, zorder=5)

# COM predictions
plt.scatter(predicted_orbits, [0.5]*len(predicted_orbits), 
            color='red', marker='x', s=100, label='COM Predictions', zorder=4)

# Annotate mismatches
for n, orbit in zip(n_values, predicted_orbits):
    closest_planet = min(observed_planets.values(), key=lambda x: abs(x - orbit))
    error_pct = 100 * (orbit - closest_planet) / closest_planet
    if abs(error_pct) > 10:
        plt.annotate(f"n={n}\n{orbit:.2f} AU\n({error_pct:.0f}%)", 
                    (orbit, 0.5), textcoords="offset points", xytext=(0, 10), ha='center')

plt.xlabel("Distance from Sun (AU)", fontsize=14)
plt.yticks([])
plt.legend(ncol=3, loc='upper center')
plt.grid(True, alpha=0.3)
plt.show()