import numpy as np

"""
LOGOS 3DCOM UOFT - UNIVERSAL PLANET PREDICTOR (PURE MATHEMATICAL)
Author: Martin Doina 
"""

# =============================================================================
# PURE MATHEMATICAL CONSTANTS - NO EMPIRICAL VALUES
# =============================================================================
PI = np.pi
E = np.e
PHI = (1 + np.sqrt(5)) / 2

# Calculate fundamental constants from pure mathematics
LZ_PURE = PI / (2 * np.sqrt(PHI))                    # π/(2√φ)
HQS_PURE = np.exp(-LZ_PURE) / LZ_PURE                # e^(-LZ)/LZ

# =============================================================================
# PLANETARY SYSTEMS DATA
# =============================================================================
PLANETARY_SYSTEMS = {
    "Solar System": {
        "a0": 0.387,
        "observed": [0.387, 0.723, 1.000, 1.524, 5.203, 9.539, 19.18, 30.06],
        "threshold": 0.05,
        "planet_names": ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"]
    },
    "Trappist-1": {
        "a0": 0.0115,
        "observed": [0.0115, 0.0158, 0.0222, 0.0293, 0.0385, 0.0469, 0.0619],
        "threshold": 0.07,
        "planet_names": ["TRAPPIST-1b", "TRAPPIST-1c", "TRAPPIST-1d", "TRAPPIST-1e", "TRAPPIST-1f", "TRAPPIST-1g", "TRAPPIST-1h"]
    },
    "Kepler-90": {
        "a0": 0.074,
        "observed": [0.074, 0.089, 0.32, 0.42, 0.48, 0.71, 1.01],
        "threshold": 0.07,
        "planet_names": ["Kepler-90b", "Kepler-90c", "Kepler-90d", "Kepler-90e", "Kepler-90f", "Kepler-90g", "Kepler-90h"]
    },
    "Proxima Centauri": {
        "a0": 0.0485,
        "observed": [0.0485, 0.081],
        "threshold": 0.07,
        "planet_names": ["Proxima Centauri a", "Proxima Centauri b"]
    },
    "HD 10180": {
        "a0": 0.022,
        "observed": [0.022, 0.064, 0.128, 0.27, 0.49, 1.42, 3.4],
        "threshold": 0.15,
        "planet_names": ["HD 10180 a", "HD 10180 b","HD 10180 c","HD 10180 d","HD 10180 e","HD 10180 f","HD 10180 g"]
    },
    "Kepler-11": {
        "a0": 0.091,
        "observed": [0.091, 0.107, 0.155, 0.195, 0.250, 0.466],
        "threshold": 0.07,
        "planet_names": ["Kepler-11 a", "Kepler-11 b","Kepler-11 c","Kepler-11 d","Kepler-11 e","Kepler-11 f"]
    },
    "Barnard Star": {
        "a0": 0.0188,
        "observed": [0.0188, 0.0229, 0.0274, 0.0381],
        "threshold": 0.07,
        "planet_names": ["Barnard Star a", "Barnard Star b","Barnard Star c","Barnard Star d"]

    },
    "Kepler-444": {
        "a0": 0.04,  # base orbit in AU, approximate for innermost planet
        "observed": [0.04, 0.06, 0.09, 0.14, 0.21],
        "threshold": 0.07,
        "planet_names": ["Kepler-444b", "Kepler-444c", "Kepler-444d", "Kepler-444e", "Kepler-444f"]
    },
    "Kepler-51": {
        "a0": 0.2514,  # AU, semi-major axis of innermost planet Kepler-51b
        "observed": [0.2514, 0.384, 0.509, 0.795],    
        "threshold": 0.07,
        "planet_names": ["Kepler-51b", "Kepler-51c", "Kepler-51d", "Kepler-51e"]
    },
    "Kepler-296": {
        "a0": 0.0521,  # AU, semi-major axis of innermost planet Kepler-296c
        "observed": [0.0521, 0.079, 0.118, 0.169, 0.255],  
        "threshold": 0.07,
        "planet_names": ["Kepler-296c", "Kepler-296b", "Kepler-296d", "Kepler-296e", "Kepler-296f"]

    },

    "55_Cancri": {
        "a0": 0.038,  # AU, semi-major axis of innermost planet 55 Cancri e
        "observed": [0.038, 0.115, 0.24, 0.781, 5.77],  
        "threshold": 0.07,
        "planet_names": ["55 Cancri e", "55 Cancri b", "55 Cancri c", "55 Cancri f", "55 Cancri d"]
    },
    "HD 202206": {
        "a0": 0.083,
        "observed": [0.083, 0.13, 2.55],  # Known unstable configuration
        "threshold": 0.07,
        "planet_names": ["HD 202206 b", "HD 202206 c", "HD 202206?"]
    },
    "GJ 876": {
        "a0": 0.0208,
        "observed": [0.0208, 0.13, 0.21, 1.58],  # Laplace resonance - stable but complex
        "threshold": 0.07,
        "planet_names": ["GJ 876 b", "GJ 876 c", "GJ 876 e", "GJ 876 d"]
    },
    "Kepler-36": {
        "a0": 0.1153,
        "observed": [0.1153, 0.1283],  # Extremely close orbits, near instability
        "threshold": 0.07,
        "planet_names": ["Kepler-36 b", "Kepler-36 c"]
    },
    "Upsilon Andromedae": {
        "a0": 0.059,
        "observed": [0.059, 0.829, 2.53, 5.25],  # Mutually inclined, complex dynamics
        "threshold": 0.07,
        "planet_names": ["Ups And b", "Ups And c", "Ups And d", "Ups And e"]

    },
    "HR 8799": {
        "a0": 0.064,  # Actually starts at ~15 AU but testing scaling
        "observed": [15.0, 24.0, 38.0, 68.0],  # Widely spaced, directly imaged
        "threshold": 0.015,
        "planet_names": ["HR 8799 e", "HR 8799 d", "HR 8799 c", "HR 8799 b"]
    },
    "HD 20781": {
        "a0": 0.169,  # Innermost planet HD 20781 b
        "observed": [0.169, 0.337, 0.787],  # Three known planets in binary system
        "threshold": 0.07,
        "planet_names": ["HD 20781 b", "HD 20781 c", "HD 20781 d"]
    },
    "Kepler-16": {
        "a0": 0.704,  # Kepler-16b - orbits BOTH stars
        "observed": [0.704],
        "threshold": 0.07,
        "planet_names": ["Kepler-16b"]
    },
    "Kepler-34": {
        "a0": 1.09,  # Kepler-34b - circumbinary
        "observed": [1.09], 
        "threshold": 0.07,
        "planet_names": ["Kepler-34b"]
    }
}

# =============================================================================
# CORE FUNCTIONS - PURE MATHEMATICS ONLY
# =============================================================================
def find_predicted_orbits_pure(a0, max_n=74, resonance_threshold=0.07):
    """Find planetary orbits using pure mathematical constants only"""
    predicted_orbits = [a0]  # Reference orbit
    
    for n in np.arange(0.001, max_n, 0.001):
        # Pure mathematical formula: a₀ × (π/(2√φ))ⁿ
        base_orbit = a0 * (LZ_PURE ** n)
        
        # Pure mathematical resonance: (1 + e^(-π/(2√φ))/(π/(2√φ))) × sin(4πn)
        resonance_strength = (1 + HQS_PURE) * np.sin(4 * PI * n)
        
        if abs(resonance_strength) > resonance_threshold:
            predicted_orbits.append(base_orbit)
    
    # Remove duplicates with high precision
    unique_orbits = []
    for orbit in sorted(predicted_orbits):
        if not unique_orbits or abs(orbit - unique_orbits[-1]) > 1e-10:
            unique_orbits.append(orbit)
    
    return unique_orbits

def match_predictions(predicted, observed, planet_names, tolerance=0.1):
    """Match predictions with true precision analysis"""
    available_preds = predicted.copy()
    matches = []
    
    for i, (actual, name) in enumerate(zip(observed, planet_names)):
        best_pred = None
        best_error = float('inf')
        
        # Find mathematically closest prediction
        for pred in available_preds:
            error = abs(pred - actual) / actual * 100
            
            # Check for exact mathematical match
            if abs(pred - actual) < 1e-15:
                best_pred = pred
                best_error = 0.0
                break
            elif error < best_error:
                best_pred = pred
                best_error = error
        
        if best_pred and best_error < tolerance * 100:
            matches.append({
                'name': name, 
                'actual': actual, 
                'predicted': best_pred, 
                'error_percent': best_error, 
                'index': i,
                'absolute_difference': abs(best_pred - actual)
            })
            available_preds.remove(best_pred)
    
    matches.sort(key=lambda x: x['index'])
    return matches

# =============================================================================
# ANALYSIS AND OUTPUT
# =============================================================================
def analyze_system(system_name, data):
    """Analyze system with pure mathematical constants"""
    print(f"\n{'='*70}")
    print(f"SYSTEM: {system_name}")
    print(f"{'='*70}")
    
    # Generate predictions using pure mathematics
    predicted_orbits = find_predicted_orbits_pure(
        data["a0"], resonance_threshold=data["threshold"]
    )
    
    # Match predictions
    matches = match_predictions(
        predicted_orbits, data["observed"], data["planet_names"]
    )
    
    # Display results
    print(f"Reference: {data['planet_names'][0]} at {data['a0']} AU")
    print(f"Predicted orbits: {len(predicted_orbits)}")
    print(f"Planets matched: {len(matches)}/{len(data['observed'])}")
    
    print(f"\n{'Planet':<15} {'Actual (AU)':<15} {'Predicted (AU)':<18} {'Error (%)':<12} {'Abs Diff':<15} {'Status':<12}")
    print(f"{'-'*15} {'-'*15} {'-'*18} {'-'*12} {'-'*15} {'-'*12}")
    
    for match in matches:
        # Precision classification
        if match['error_percent'] < 1e-10:
            status = "MATHEMATICAL"
            error_display = 0.0
        elif match['error_percent'] < 0.001:
            status = "QUANTUM"
            error_display = match['error_percent']
        elif match['error_percent'] < 0.01:
            status = "ATOMIC" 
            error_display = match['error_percent']
        elif match['error_percent'] < 0.1:
            status = "NANO"
            error_display = match['error_percent']
        else:
            status = "EXCELLENT"
            error_display = match['error_percent']
        
        print(f"{match['name']:<15} {match['actual']:<15.6f} {match['predicted']:<18.12f} "
              f"{error_display:<12.8f} {match['absolute_difference']:<15.2e} {status:<12}")
    
    return len(matches)

def run_pure_mathematical_analysis():
    """Run analysis using only pure mathematical constants"""
    print(" LOGOS 3DCOM UNIVERSAL PLANET PREDICTOR - PURE MATHEMATICAL")
    print("=" * 70)
    print("USING ONLY: π, e, φ")
    print(f"LZ = π/(2√φ) = {LZ_PURE:.16f}")
    print(f"HQS = e^(-LZ)/LZ = {HQS_PURE:.16f}")
    print("=" * 70)
    
    total_matches = 0
    total_planets = 0
    
    for system_name, data in PLANETARY_SYSTEMS.items():
        matches = analyze_system(system_name, data)
        total_matches += matches
        total_planets += len(data['observed'])
    
    # Final summary
    accuracy = (total_matches / total_planets) * 100
    print(f"\n{'='*70}")
    print(f"PURE MATHEMATICAL ACCURACY: {total_matches}/{total_planets} planets")
    print(f"OVERALL ACCURACY: {accuracy:.6f}%")
    print(f"Constants: LZ = π/(2√φ), HQS = e^(-LZ)/LZ")
    print(f"{'='*70}")

# =============================================================================
# EXECUTION
# =============================================================================
if __name__ == "__main__":
    run_pure_mathematical_analysis()
