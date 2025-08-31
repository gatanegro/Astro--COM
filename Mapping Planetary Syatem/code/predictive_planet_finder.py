import numpy as np
"""
3DCOM UOFT
Author: Martin Doina 
"""

# =============================================================================
# 3DCOM UNIVERSAL CONSTANTS
# =============================================================================
LZ = 1.23498228
HQS = 0.235
PI = np.pi

# =============================================================================
# PREDICTIVE PLANET FINDER
# =============================================================================
def find_predicted_planets(a0, max_n=20, resonance_threshold=0.1):
    """
    Find ALL possible planetary orbits where stable resonances occur
    Returns orbital distances where resonance function peaks
    """
    predicted_orbits = []
    
    # Scan through all possible recursive levels
    for n in np.arange(0, max_n, 0.001):  # Fine-grained scan
        # Calculate base orbital distance
        base_orbit = a0 * (LZ ** n)
        
        # Calculate resonance strength (using 3DCOM resonance function)
        resonance_strength = HQS * np.sin(0.104 * PI * n)  # Example for Trappist-1
        
        # Planets form at resonance peaks (where resonance is strong)
        if abs(resonance_strength) > resonance_threshold:
            predicted_orbits.append(base_orbit)
    
    return sorted(set(round(orbit, 6) for orbit in predicted_orbits))

# =============================================================================
# PLANETARY DATA FOR COMPARISON
# =============================================================================
observed_t1 = [0.0115, 0.0158, 0.0222, 0.0293, 0.0385, 0.0469, 0.0619]  # Trappist-1
observed_ss = [0.387, 0.723, 1.000, 1.524, 5.203, 9.539, 19.18, 30.06]   # Solar System
observed_k90 = [0.074, 0.089, 0.32, 0.42, 0.48, 0.71, 1.01]              # Kepler-90

# =============================================================================
# PREDICT PLANETS FOR DIFFERENT SYSTEMS
# =============================================================================
print("=== PREDICTED PLANETARY ORBITS ===")

# Trappist-1 predictions
trappist_predicted = find_predicted_planets(a0=0.0115, resonance_threshold=0.08)
print(f"Trappist-1 predicted orbits: {trappist_predicted}")
print(f"Trappist-1 actual orbits:    {observed_t1}")

# Solar system predictions  
solar_predicted = find_predicted_planets(a0=0.387, resonance_threshold=0.05)
print(f"\nSolar system predicted orbits: {solar_predicted}")
print(f"Solar system actual orbits:    {observed_ss}")

# Kepler-90 predictions
kepler_predicted = find_predicted_planets(a0=0.074, resonance_threshold=0.07)
print(f"\nKepler-90 predicted orbits: {kepler_predicted}")
print(f"Kepler-90 actual orbits:    {observed_k90}")

# =============================================================================
# ANALYSIS: See how well predictions match reality
# =============================================================================
def check_predictions(predicted, observed, tolerance=0.01):
    """Check how many predicted orbits match actual planets"""
    matches = 0
    for actual in observed:
        for pred in predicted:
            if abs(pred - actual) / actual < tolerance:  # 1% tolerance
                matches += 1
                break
    return matches, len(observed)

print("\n=== PREDICTION ACCURACY ===")
for system, pred, obs in [("Trappist-1", trappist_predicted, observed_t1),
                         ("Solar System", solar_predicted, observed_ss),
                         ("Kepler-90", kepler_predicted, observed_k90)]:
    matches, total = check_predictions(pred, obs)
    print(f"{system}: {matches}/{total} planets predicted correctly")
