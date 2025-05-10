import numpy as np
import matplotlib.pyplot as plt

# Constants
LZ = 1.23498
HQS = 0.235
a0 = 0.387  # Mercury's semi-major axis (AU) as baseline

# COM orbital equation
def com_orbit(n):
    return a0 * (LZ ** n) * (1 + HQS * np.tanh(n/2))

# Known solar system planets (AU)
known_planets = {
    "Mercury": 0.387,
    "Venus": 0.723,
    "Earth": 1.000,
    "Mars": 1.524,
    "Ceres": 2.767,  # Asteroid belt
    "Jupiter": 5.203,
    "Saturn": 9.537,
    "Uranus": 19.191,
    "Neptune": 30.069,
    "Pluto": 39.482
}

# Generate predictions for n=0 to n=15
n_values = np.arange(0, 16)
predicted_orbits = [com_orbit(n) for n in n_values]

# Plot
plt.figure(figsize=(12, 6))
plt.title("COM Model Predictions vs. Observed Solar System Orbits", fontsize=14)

# Plot known planets
for planet, orbit in known_planets.items():
    plt.scatter(orbit, 0, label=planet, s=100)

# Plot COM predictions
plt.scatter(predicted_orbits, [0.5]*len(predicted_orbits), 
            color='red', marker='x', s=100, label='COM Predictions')

# Highlight potential new objects
for i, orbit in enumerate(predicted_orbits):
    if not any(np.isclose(orbit, known_orbits, rtol=0.1) for known_orbits in known_planets.values()):
        plt.annotate(f"n={i}\n{orbit:.2f} AU", (orbit, 0.5), 
                    textcoords="offset points", xytext=(0,10), ha='center')

plt.xlabel("Semi-Major Axis (AU)", fontsize=12)
plt.yticks([])
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)
plt.show()