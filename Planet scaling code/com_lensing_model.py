import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os

# Create directories for figures
os.makedirs('figures', exist_ok=True)

# COM Framework Constants
LZ = 1.23498  # Fundamental scaling constant
HQS = 0.235  # Harmonic Quantum Scalar (23.5% of LZ)
HQS_LZ = HQS * LZ  # HQS threshold in absolute terms (≈ 0.29022)

# 24-step Fibonacci digital root pattern
FIBONACCI_PATTERN = [1, 1, 2, 3, 5, 8, 4, 3, 7, 1, 8, 9, 8, 8, 7, 6, 4, 1, 5, 6, 2, 8, 1, 9]
FIBONACCI_NORMALIZED = np.array(FIBONACCI_PATTERN) / 9.0  # Normalize to 0-1 range

# Physical constants
G = 6.67430e-11  # Gravitational constant in m^3 kg^-1 s^-2
c = 2.99792458e8  # Speed of light in m/s
M_SUN = 1.989e30  # Solar mass in kg
AU = 1.496e11  # Astronomical unit in meters

# Astronomical objects for lensing validation
lensing_objects = [
    {"name": "Sun", "mass": 1.989e30, "radius": 6.957e8, "distance": 1.496e11},  # 1 AU
    {"name": "Jupiter", "mass": 1.898e27, "radius": 6.9911e7, "distance": 7.785e11},  # 5.2 AU
    {"name": "Sagittarius A*", "mass": 4.154e6 * 1.989e30, "radius": 1.2e10, "distance": 2.55e20},  # 8.178 kpc
    {"name": "M87 Black Hole", "mass": 6.5e9 * 1.989e30, "radius": 2.0e13, "distance": 5.23e23},  # 16.8 Mpc
    {"name": "Typical Galaxy Cluster", "mass": 1e15 * 1.989e30, "radius": 3.086e22, "distance": 3.086e24}  # 1 Gpc
]

# Einstein ring calculation (General Relativity)
def einstein_ring_radius(mass, distance_observer, distance_source):
    """Calculate Einstein ring radius using General Relativity
    
    Parameters:
    - mass: Mass of the lensing object in kg
    - distance_observer: Distance from observer to lens in meters
    - distance_source: Distance from lens to source in meters
    
    Returns:
    - Einstein ring radius in meters
    """
    # Calculate the reduced distance
    reduced_distance = (distance_observer * distance_source) / (distance_observer + distance_source)
    
    # Calculate Einstein radius
    einstein_radius = np.sqrt((4 * G * mass * reduced_distance) / c**2)
    
    return einstein_radius

# COM-based lensing model
def com_lensing_deflection(mass, impact_parameter, distance_observer, distance_source):
    """Calculate light deflection angle using COM framework
    
    Parameters:
    - mass: Mass of the lensing object in kg
    - impact_parameter: Closest approach distance in meters
    - distance_observer: Distance from observer to lens in meters
    - distance_source: Distance from lens to source in meters
    
    Returns:
    - Deflection angle in radians
    """
    # Calculate standard GR deflection angle
    gr_deflection = (4 * G * mass) / (c**2 * impact_parameter)
    
    # Calculate energy density ratio (normalized to critical density)
    energy_density_ratio = (G * mass) / (c**2 * impact_parameter**3)
    
    # Calculate octave position in COM framework
    octave_position = np.log(energy_density_ratio) / np.log(LZ) % 1
    
    # Get Fibonacci pattern value for this position
    pattern_index = int(octave_position * 24) % 24
    pattern_value = FIBONACCI_NORMALIZED[pattern_index]
    
    # Calculate HQS modulation factor
    hqs_factor = 1 + HQS * np.sin(np.pi * octave_position / HQS)
    
    # Calculate COM-modified deflection angle
    com_deflection = gr_deflection * hqs_factor * (1 + (LZ - 1) * pattern_value)
    
    return com_deflection

# COM-based Einstein ring calculation
def com_einstein_ring(mass, distance_observer, distance_source):
    """Calculate Einstein ring radius using COM framework
    
    Parameters:
    - mass: Mass of the lensing object in kg
    - distance_observer: Distance from observer to lens in meters
    - distance_source: Distance from lens to source in meters
    
    Returns:
    - COM-modified Einstein ring radius in meters
    """
    # Calculate standard Einstein radius
    standard_radius = einstein_ring_radius(mass, distance_observer, distance_source)
    
    # Calculate energy density at the Einstein radius
    energy_density_ratio = (G * mass) / (c**2 * standard_radius**3)
    
    # Calculate octave position in COM framework
    octave_position = np.log(energy_density_ratio) / np.log(LZ) % 1
    
    # Get Fibonacci pattern value for this position
    pattern_index = int(octave_position * 24) % 24
    pattern_value = FIBONACCI_NORMALIZED[pattern_index]
    
    # Calculate HQS modulation factor
    hqs_factor = 1 + HQS * np.sin(np.pi * octave_position / HQS)
    
    # Calculate COM-modified Einstein radius
    com_radius = standard_radius * np.sqrt(hqs_factor * (1 + (LZ - 1) * pattern_value))
    
    return com_radius

# Calculate lensing magnification
def magnification(mass, impact_parameter, distance_observer, distance_source, model="GR"):
    """Calculate magnification factor for gravitational lensing
    
    Parameters:
    - mass: Mass of the lensing object in kg
    - impact_parameter: Impact parameter in meters
    - distance_observer: Distance from observer to lens in meters
    - distance_source: Distance from lens to source in meters
    - model: "GR" for General Relativity or "COM" for COM framework
    
    Returns:
    - Magnification factor (dimensionless)
    """
    # Calculate Einstein radius
    if model == "GR":
        einstein_radius = einstein_ring_radius(mass, distance_observer, distance_source)
    else:  # COM model
        einstein_radius = com_einstein_ring(mass, distance_observer, distance_source)
    
    # Normalized impact parameter
    u = impact_parameter / einstein_radius
    
    # Magnification formula
    magnification = (u**2 + 2) / (u * np.sqrt(u**2 + 4))
    
    return magnification

# Simulation 1: Compare GR and COM lensing for different masses
def simulate_mass_scaling():
    # Mass range (solar masses)
    mass_range = np.logspace(0, 15, 1000)  # 1 to 10^15 solar masses
    
    # Fixed parameters
    distance_observer = 1.0e22  # 10 kpc (typical galactic distance)
    distance_source = 1.0e23    # 100 kpc (typical intergalactic distance)
    
    # Calculate Einstein radii
    gr_radii = np.array([einstein_ring_radius(m * M_SUN, distance_observer, distance_source) for m in mass_range])
    com_radii = np.array([com_einstein_ring(m * M_SUN, distance_observer, distance_source) for m in mass_range])
    
    # Calculate ratio
    ratio = com_radii / gr_radii
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot Einstein radii
    plt.subplot(2, 1, 1)
    plt.loglog(mass_range, gr_radii, 'b-', label='General Relativity', linewidth=2)
    plt.loglog(mass_range, com_radii, 'r-', label='COM Framework', linewidth=2)
    
    # Add reference points for known objects
    for obj in lensing_objects:
        mass_solar = obj["mass"] / M_SUN
        if 1 <= mass_solar <= 1e15:
            gr_radius = einstein_ring_radius(obj["mass"], distance_observer, distance_source)
            com_radius = com_einstein_ring(obj["mass"], distance_observer, distance_source)
            plt.plot(mass_solar, gr_radius, 'bo', markersize=8)
            plt.plot(mass_solar, com_radius, 'ro', markersize=8)
            plt.text(mass_solar, gr_radius*1.2, obj["name"], ha='center', fontsize=9)
    
    plt.xlabel('Mass (Solar Masses)')
    plt.ylabel('Einstein Ring Radius (m)')
    plt.title('Einstein Ring Radius vs. Mass')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot ratio
    plt.subplot(2, 1, 2)
    plt.semilogx(mass_range, ratio, 'k-', linewidth=2)
    
    # Add horizontal line at ratio=1
    plt.axhline(y=1, color='gray', linestyle='--', alpha=0.7)
    
    # Add vertical lines at pattern transitions
    for i in range(1, 25):
        transition_mass = 10**(i/8 * 15 / 24)
        plt.axvline(x=transition_mass, color='g', linestyle='-', alpha=0.2)
    
    plt.xlabel('Mass (Solar Masses)')
    plt.ylabel('COM/GR Ratio')
    plt.title('Ratio of COM to GR Einstein Ring Radius')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/lensing_mass_scaling.png', dpi=300)
    
    return {
        "mass_range": mass_range,
        "gr_radii": gr_radii,
        "com_radii": com_radii,
        "ratio": ratio
    }

# Simulation 2: Angular deflection vs. impact parameter
def simulate_deflection_profile():
    # Impact parameter range (as fraction of object radius)
    impact_range = np.logspace(0, 3, 1000)  # 1 to 1000 times the object radius
    
    # Results for different objects
    results = {}
    
    # Create figure
    plt.figure(figsize=(15, 10))
    
    # For each lensing object
    for i, obj in enumerate(lensing_objects):
        # Calculate deflection angles for different impact parameters
        impact_parameters = impact_range * obj["radius"]
        
        gr_deflections = np.array([(4 * G * obj["mass"]) / (c**2 * b) for b in impact_parameters])
        com_deflections = np.array([com_lensing_deflection(obj["mass"], b, obj["distance"], obj["distance"]*10) 
                                   for b in impact_parameters])
        
        # Convert to arcseconds
        gr_deflections_arcsec = gr_deflections * 206265  # radians to arcseconds
        com_deflections_arcsec = com_deflections * 206265
        
        # Store results
        results[obj["name"]] = {
            "impact_parameters": impact_parameters,
            "gr_deflections": gr_deflections_arcsec,
            "com_deflections": com_deflections_arcsec
        }
        
        # Plot deflection angles
        plt.subplot(len(lensing_objects), 2, 2*i+1)
        plt.loglog(impact_range, gr_deflections_arcsec, 'b-', label='General Relativity', linewidth=2)
        plt.loglog(impact_range, com_deflections_arcsec, 'r-', label='COM Framework', linewidth=2)
        
        plt.xlabel('Impact Parameter (object radii)')
        plt.ylabel('Deflection Angle (arcsec)')
        plt.title(f'Light Deflection by {obj["name"]}')
        plt.grid(True, alpha=0.3)
        if i == 0:
            plt.legend()
        
        # Plot ratio
        plt.subplot(len(lensing_objects), 2, 2*i+2)
        ratio = com_deflections_arcsec / gr_deflections_arcsec
        plt.semilogx(impact_range, ratio, 'k-', linewidth=2)
        
        # Add horizontal line at ratio=1
        plt.axhline(y=1, color='gray', linestyle='--', alpha=0.7)
        
        # Add pattern structure
        pattern_positions = []
        for j in range(24):
            if j % 8 == 0:  # Highlight every 8th position
                pattern_positions.append(j/24)
                
        for pos in pattern_positions:
            # Find approximate impact parameter for this pattern position
            octave_transitions = np.exp(np.log(LZ) * np.arange(10))  # Several octaves
            for oct in octave_transitions:
                # Calculate impact parameter that would give this octave position
                b_approx = obj["radius"] * oct
                if obj["radius"] <= b_approx <= 1000 * obj["radius"]:
                    plt.axvline(x=b_approx/obj["radius"], color='g', linestyle='-', alpha=0.2)
        
        plt.xlabel('Impact Parameter (object radii)')
        plt.ylabel('COM/GR Ratio')
        plt.title(f'Ratio of COM to GR Deflection for {obj["name"]}')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/lensing_deflection_profiles.png', dpi=300)
    
    return results

# Simulation 3: Magnification patterns
def simulate_magnification_patterns():
    # Impact parameter range (as fraction of Einstein radius)
    u_range = np.linspace(0.01, 3, 1000)  # 0.01 to 3 Einstein radii
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # For a typical galaxy cluster
    obj = lensing_objects[-1]  # Typical Galaxy Cluster
    
    # Calculate Einstein radius
    distance_observer = 1.0e25  # 1 Gpc (typical cosmological distance)
    distance_source = 2.0e25    # 2 Gpc (typical source distance)
    
    gr_einstein_radius = einstein_ring_radius(obj["mass"], distance_observer, distance_source)
    com_einstein_radius = com_einstein_ring(obj["mass"], distance_observer, distance_source)
    
    # Calculate impact parameters
    gr_impact_parameters = u_range * gr_einstein_radius
    com_impact_parameters = u_range * com_einstein_radius
    
    # Calculate magnifications
    gr_magnifications = np.array([magnification(obj["mass"], b, distance_observer, distance_source, "GR") 
                                 for b in gr_impact_parameters])
    
    com_magnifications = np.array([magnification(obj["mass"], b, distance_observer, distance_source, "COM") 
                                  for b in com_impact_parameters])
    
    # Plot magnifications
    plt.subplot(2, 1, 1)
    plt.plot(u_range, gr_magnifications, 'b-', label='General Relativity', linewidth=2)
    plt.plot(u_range, com_magnifications, 'r-', label='COM Framework', linewidth=2)
    
    plt.xlabel('Impact Parameter (Einstein radii)')
    plt.ylabel('Magnification')
    plt.title('Magnification vs. Impact Parameter')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot ratio
    plt.subplot(2, 1, 2)
    ratio = com_magnifications / gr_magnifications
    plt.plot(u_range, ratio, 'k-', linewidth=2)
    
    # Add horizontal line at ratio=1
    plt.axhline(y=1, color='gray', linestyle='--', alpha=0.7)
    
    # Add pattern structure
    for i in range(24):
        pattern_value = FIBONACCI_NORMALIZED[i]
        plt.axhline(y=1 + 0.1 * pattern_value, color='g', linestyle='-', alpha=0.1)
    
    plt.xlabel('Impact Parameter (Einstein radii)')
    plt.ylabel('COM/GR Ratio')
    plt.title('Ratio of COM to GR Magnification')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/lensing_magnification_patterns.png', dpi=300)
    
    return {
        "u_range": u_range,
        "gr_magnifications": gr_magnifications,
        "com_magnifications": com_magnifications,
        "ratio": ratio
    }

# Simulation 4: Scale invariance analysis
def simulate_scale_invariance():
    # Scale range (meters)
    scale_range = np.logspace(-15, 25, 1000)  # Quantum to cosmic scales
    
    # Calculate COM correction factor across scales
    correction_factors = []
    
    for scale in scale_range:
        # Calculate octave position
        octave_position = np.log(scale / 1e-15) / np.log(LZ) % 1
        
        # Get Fibonacci pattern value for this position
        pattern_index = int(octave_position * 24) % 24
        pattern_value = FIBONACCI_NORMALIZED[pattern_index]
        
        # Calculate HQS modulation factor
        hqs_factor = 1 + HQS * np.sin(np.pi * octave_position / HQS)
        
        # Calculate correction factor
        correction = hqs_factor * (1 + (LZ - 1) * pattern_value)
        correction_factors.append(correction)
    
    # Create figure
    plt.figure(figsize=(15, 10))
    
    # Plot correction factor
    plt.subplot(2, 1, 1)
    plt.semilogx(scale_range, correction_factors, 'k-', linewidth=1.5)
    
    # Add reference scales
    reference_scales = [
        {"name": "Proton", "scale": 1e-15},
        {"name": "Atom", "scale": 1e-10},
        {"name": "Virus", "scale": 1e-7},
        {"name": "Human", "scale": 1},
        {"name": "Earth", "scale": 1e7},
        {"name": "Solar System", "scale": 1e13},
        {"name": "Galaxy", "scale": 1e21},
        {"name": "Observable Universe", "scale": 1e26}
    ]
    
    for ref in reference_scales:
        if min(scale_range) <= ref["scale"] <= max(scale_range):
            idx = np.abs(scale_range - ref["scale"]).argmin()
            plt.plot(ref["scale"], correction_factors[idx], 'ro', markersize=8)
            plt.text(ref["scale"], correction_factors[idx]*1.05, ref["name"], ha='center', fontsize=9)
    
    plt.xlabel('Scale (meters)')
    plt.ylabel('COM Correction Factor')
    plt.title('Scale Invariance in COM Framework')
    plt.grid(True, alpha=0.3)
    
    # Plot pattern structure
    plt.subplot(2, 1, 2)
    
    # Calculate octave positi
(Content truncated due to size limit. Use line ranges to read in chunks)