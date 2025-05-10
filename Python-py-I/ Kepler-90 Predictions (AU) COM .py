import numpy as np
import matplotlib.pyplot as plt

# COM Constants
LZ = 1.23498       # Poincaré 3-sphere scaling
HQS = 0.235        # Energy threshold (23.5%)
HQS_compact = 0.18 # Adjusted for compact systems (TRAPPIST-1)

def com_orbit(n, system, baseline_planet=0):
    """
    Predicts orbits using adaptive sin/tanh modulation.
    system: 'solar', 'trappist1', 'kepler90'
    baseline_planet: Orbital distance of the baseline planet (AU)
    """
    # Select modulation function
    trig_func = np.sin if system in ['solar', 'kepler90'] else np.tanh
    
    # Tidal compensation factor (0 for resonant systems)
    k_tidal = 0.1 if system == 'trappist1' else 0.01 if system == 'solar' else 0.0
    
    # Calculate raw orbit
    a_n = (LZ ** n) * (1 + (HQS_compact if system == 'trappist1' else HQS) * trig_func(n/2))
    
    # Apply baseline planet scaling for outer planets
    if baseline_planet > 0:
        if system == 'solar' and n > 4:    # Solar: Mercury baseline for n>4
            a_n = 0.387 * (LZ ** (n-4)) * (1 + HQS * trig_func((n-4)/2))
        elif system == 'trappist1' and n > 3:  # TRAPPIST-1: Planet-b baseline
            a_n = 0.0115 * (LZ ** (n-3)) * (1 + HQS * trig_func((n-3)/2))
        elif system == 'kepler90' and n > 1:   # Kepler-90: Planet-b baseline
            a_n = 0.074 * (LZ ** (n-1)) * (1 + HQS * trig_func((n-1)/2))
    
    # Tidal suppression
    a_n *= (1 - k_tidal * (a_n ** 1.5))  # a_n^3/2 term
    
    return a_n

# Example: Kepler-90 predictions
n_values = np.arange(1, 10)  # n=1 to n=9 (for planet 90j)
kepler90_pred = [com_orbit(n, 'kepler90', baseline_planet=0.074) for n in n_values]

print("Kepler-90 Predictions (AU):")
for n, a in zip(n_values, kepler90_pred):
    print(f"n={n}: {a:.3f} AU")