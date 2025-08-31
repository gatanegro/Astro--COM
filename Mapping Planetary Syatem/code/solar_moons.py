import numpy as np
import matplotlib.pyplot as plt
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
# COMPLETE SOLAR SYSTEM DATA (Planets + Major Moons)
# =============================================================================
# Planets: [name, distance from parent (AU or km), parent body, type]
solar_system = [
    # Sun-centered (planets)
    ['Mercury', 0.387, 'Sun', 'planet'],
    ['Venus', 0.723, 'Sun', 'planet'],
    ['Earth', 1.000, 'Sun', 'planet'],
    ['Mars', 1.524, 'Sun', 'planet'],
    ['Jupiter', 5.203, 'Sun', 'planet'],
    ['Saturn', 9.539, 'Sun', 'planet'],
    ['Uranus', 19.18, 'Sun', 'planet'],
    ['Neptune', 30.06, 'Sun', 'planet'],
    
    # Earth moons
    ['Moon', 384400, 'Earth', 'moon'],  # distance in km
    
    # Mars moons
    ['Phobos', 9377, 'Mars', 'moon'],
    ['Deimos', 23460, 'Mars', 'moon'],
    
    # Jupiter moons (Galilean + major)
    ['Io', 421700, 'Jupiter', 'moon'],
    ['Europa', 671034, 'Jupiter', 'moon'],
    ['Ganymede', 1070412, 'Jupiter', 'moon'],
    ['Callisto', 1882709, 'Jupiter', 'moon'],
    
    # Saturn moons (major)
    ['Mimas', 185540, 'Saturn', 'moon'],
    ['Enceladus', 238040, 'Saturn', 'moon'],
    ['Tethys', 294670, 'Saturn', 'moon'],
    ['Dione', 377420, 'Saturn', 'moon'],
    ['Rhea', 527070, 'Saturn', 'moon'],
    ['Titan', 1221870, 'Saturn', 'moon'],
    ['Iapetus', 3561300, 'Saturn', 'moon'],
    
    # Uranus moons (major)
    ['Miranda', 129900, 'Uranus', 'moon'],
    ['Ariel', 191020, 'Uranus', 'moon'],
    ['Umbriel', 266300, 'Uranus', 'moon'],
    ['Titania', 435910, 'Uranus', 'moon'],
    ['Oberon', 583520, 'Uranus', 'moon'],
    
    # Neptune moons
    ['Triton', 354759, 'Neptune', 'moon'],
    ['Nereid', 5513818, 'Neptune', 'moon']
]

# =============================================================================
# RECURSIVE MAPPING FUNCTION
# =============================================================================
def find_recursive_depth(distance, a0):
    """Find the recursive depth n for a given orbital distance"""
    return np.log(distance / a0) / np.log(LZ)

def map_solar_system():
    """Map entire solar system to recursive depths"""
    results = []
    
    for body, distance, parent, body_type in solar_system:
        # Convert km to AU for moons (1 AU = 149,597,870.7 km)
        if 'moon' in body_type:
            distance_au = distance / 149597870.7
            # Use parent planet's distance as a0
            parent_distance = next(d for n, d, p, t in solar_system if n == parent)
            a0 = parent_distance  # moons reference their parent planet
        else:
            distance_au = distance
            a0 = 0.387  # Mercury's distance for planets
            
        n = find_recursive_depth(distance_au, a0)
        results.append((body, parent, distance_au, n, body_type))
    
    return results

# =============================================================================
# ANALYSIS AND VISUALIZATION
# =============================================================================
def analyze_recursive_patterns(mapped_data):
    """Analyze the recursive patterns in the solar system"""
    print("SOLAR SYSTEM RECURSIVE MAPPING")
    print("="*60)
    print(f"{'Body':<12} {'Parent':<8} {'Distance (AU)':<12} {'n-value':<8} {'Type':<6}")
    print("-"*60)
    
    for body, parent, dist, n, btype in mapped_data:
        print(f"{body:<12} {parent:<8} {dist:<12.6f} {n:<8.3f} {btype:<6}")
    
    # Check for quantization patterns
    print("\n" + "="*60)
    print("ANALYSIS: Looking for quantized n-values...")
    
    # Group by parent and type
    from collections import defaultdict
    groups = defaultdict(list)
    
    for body, parent, dist, n, btype in mapped_data:
        groups[(parent, btype)].append((body, n))
    
    for (parent, btype), bodies in groups.items():
        n_values = sorted([n for _, n in bodies])
        if len(n_values) > 1:
            spacing = np.mean(np.diff(n_values))
            print(f"{parent} {btype}: n-values {[f'{n:.3f}' for n in n_values]}, average spacing: {spacing:.3f}")
# =============================================================================
# RUN COMPLETE ANALYSIS
# =============================================================================
mapped_data = map_solar_system()
analyze_recursive_patterns(mapped_data)

# Create beautiful visualization
def analyze_recursive_patterns(mapped_data):
    """Analyze the recursive patterns in the solar system"""
    print("SOLAR SYSTEM RECURSIVE MAPPING")
    print("="*60)
    print(f"{'Body':<12} {'Parent':<8} {'Distance (AU)':<12} {'n-value':<8} {'Type':<6}")
    print("-"*60)
    
    for body, parent, dist, n, btype in mapped_data:
        print(f"{body:<12} {parent:<8} {dist:<12.6f} {n:<8.3f} {btype:<6}")
    
    # Check for quantization patterns
    print("\n" + "="*60)
    print("ANALYSIS: Looking for quantized n-values...")
    
    # Group by parent and type
    from collections import defaultdict
    groups = defaultdict(list)
    
    for body, parent, dist, n, btype in mapped_data:
        groups[(parent, btype)].append((body, n))
    
    for (parent, btype), bodies in groups.items():
        n_values = sorted([n for _, n in bodies])
        if len(n_values) > 1:
            spacing = np.mean(np.diff(n_values))
            print(f"{parent} {btype}: n-values {[f'{n:.3f}' for n in n_values]}, average spacing: {spacing:.3f}")
