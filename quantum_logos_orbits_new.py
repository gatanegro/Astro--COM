import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
"""
LOGOS THEORY
Author: Martin Doina 
"""

class CelestialSpiralDynamics:
    def __init__(self):
        self.G = 6.67430e-11
        self.c = 299792458
        
    def orbital_resonance_spiral(self, central_mass, satellite_mass, initial_distance):
        """Show how spiral geometry explains orbital acceleration/deceleration"""
        
        print("=== CELESTIAL SPIRAL RESONANCE DYNAMICS ===")
        
        # LOGOS: The recursion asymmetry creates acceleration patterns
        def calculate_orbital_energy(central_mass, satellite_mass, distance, direction):
            """Orbital energy depends on spiral recursion direction"""
            
            # Gravitational potential energy (classical)
            U_classical = -self.G * central_mass * satellite_mass / distance
            
            # LOGOS: Spiral geometry adds direction-dependent component
            if direction == 'inward':
                # Spiral tightening toward center - ACCELERATION
                spiral_energy = -0.1 * U_classical  # Negative = energy gain
                recursion_factor = 0.8  # Fewer recursions inward
            else:  # outward
                # Spiral expanding outward - DECELERATION  
                spiral_energy = +0.1 * U_classical  # Positive = energy loss
                recursion_factor = 1.2  # More recursions outward
                
            total_energy = U_classical + spiral_energy
            return total_energy, spiral_energy, recursion_factor
        
        # Analyze inward vs outward motion
        energy_inward, spiral_inward, recursions_inward = calculate_orbital_energy(
            central_mass, satellite_mass, initial_distance, 'inward')
        
        energy_outward, spiral_outward, recursions_outward = calculate_orbital_energy(
            central_mass, satellite_mass, initial_distance, 'outward')
        
        print(f"INWARD SPIRAL (Acceleration):")
        print(f"  Classical Energy: {-self.G * central_mass * satellite_mass / initial_distance:.6e} J")
        print(f"  Spiral Energy Component: {spiral_inward:.6e} J")
        print(f"  Total Energy: {energy_inward:.6e} J")
        print(f"  Recursion Factor: {recursions_inward}")
        
        print(f"OUTWARD SPIRAL (Deceleration):")
        print(f"  Classical Energy: {-self.G * central_mass * satellite_mass / initial_distance:.6e} J") 
        print(f"  Spiral Energy Component: {spiral_outward:.6e} J")
        print(f"  Total Energy: {energy_outward:.6e} J")
        print(f"  Recursion Factor: {recursions_outward}")
        
        energy_asymmetry = energy_outward / energy_inward
        print(f"ENERGY ASYMMETRY RATIO: {energy_asymmetry:.6f}")
        
        return {
            'inward': {'energy': energy_inward, 'spiral_energy': spiral_inward, 'recursions': recursions_inward},
            'outward': {'energy': energy_outward, 'spiral_energy': spiral_outward, 'recursions': recursions_outward}
        }

    def visualize_resonant_orbits(self):
        """Show how spiral geometry creates orbital resonances"""
        
        # Simulate planetary system with spiral dynamics
        central_mass = 1.989e30  # Sun mass
        planet_masses = [3.301e23, 4.867e24, 5.972e24, 6.417e23]  # Mercury, Venus, Earth, Mars
        semi_major_axes = [5.79e10, 1.082e11, 1.496e11, 2.279e11]  # meters
        
        plt.figure(figsize=(15, 10))
        
        for i, (planet_mass, a) in enumerate(zip(planet_masses, semi_major_axes)):
            
            # Calculate orbital parameters with spiral geometry
            orbital_period = 2 * np.pi * np.sqrt(a**3 / (self.G * central_mass))
            
            # LOGOS: Resonance comes from recursion number ratios!
            recursions_inward = int(10 * (central_mass / planet_mass)**0.1)
            recursions_outward = int(10 * (planet_mass / central_mass)**0.1)
            
            # Generate resonant spiral orbit
            theta = np.linspace(0, 4*np.pi * (recursions_inward + recursions_outward)/20, 1000)
            
            # Elliptical orbit with spiral modulation (resonance pattern)
            r = a * (1 - 0.1 * np.cos(theta * recursions_inward/recursions_outward))
            
            # Convert to Cartesian
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            
            plt.subplot(2, 2, i+1)
            plt.plot(x, y, 'b-', alpha=0.7, linewidth=1.5)
            plt.plot(0, 0, 'yo', markersize=10, label='Central Mass')
            plt.plot(x[0], y[0], 'ro', markersize=5, label='Planet Start')
            
            # Mark acceleration/deceleration regions
            acceleration_zones = theta % (2*np.pi) < np.pi  # Simplified model
            plt.plot(x[acceleration_zones], y[acceleration_zones], 'g.', alpha=0.3, markersize=2, label='Acceleration')
            plt.plot(x[~acceleration_zones], y[~acceleration_zones], 'r.', alpha=0.3, markersize=2, label='Deceleration')
            
            plt.xlabel('X Position (m)')
            plt.ylabel('Y Position (m)')
            plt.title(f'Planet {i+1}\nRecursions: In={recursions_inward}, Out={recursions_outward}')
            plt.legend()
            plt.axis('equal')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def gravitational_assist_analysis(self):
        """Show how gravitational assists work via spiral geometry"""
        
        print("\n=== GRAVITATIONAL ASSIST SPIRAL MECHANISM ===")
        
        # Parameters for spacecraft flyby
        planet_mass = 5.972e24  # Earth mass
        spacecraft_mass = 1000  # kg
        approach_velocity = 10000  # m/s
        flyby_distance = 6.371e6  # Earth radius
        
        def calculate_assist_dynamics(approach_dir, exit_dir):
            """Calculate velocity changes from spiral geometry"""
            
            # LOGOS: The spiral recursion asymmetry creates net velocity change!
            if approach_dir == 'incoming' and exit_dir == 'outgoing':
                # Standard gravitational assist - SPIRAL ACCELERATION
                recursion_asymmetry = 0.7  # Fewer recursions on outgoing path
                energy_gain = 0.15 * approach_velocity
            else:
                recursion_asymmetry = 1.0
                energy_gain = 0
                
            final_velocity = approach_velocity + energy_gain
            velocity_change = final_velocity - approach_velocity
            
            return final_velocity, velocity_change, recursion_asymmetry
        
        # Analyze gravitational assist
        final_vel, delta_v, recursion_ratio = calculate_assist_dynamics('incoming', 'outgoing')
        
        print(f"Initial Velocity: {approach_velocity:.1f} m/s")
        print(f"Final Velocity: {final_vel:.1f} m/s") 
        print(f"Velocity Gain: {delta_v:.1f} m/s")
        print(f"Recursion Asymmetry Ratio: {recursion_ratio:.3f}")
        print(f"Energy Explanation: Spiral geometry has fewer recursions on exit path")
        print(f"Result: Spacecraft extracts orbital energy from planet!")
        
        # Visualize the gravitational assist spiral
        plt.figure(figsize=(12, 5))
        
        # Incoming spiral (many recursions = slower)
        theta_in = np.linspace(0, 6*np.pi, 500)
        r_in = np.exp(0.05 * theta_in)
        x_in = -r_in * np.cos(theta_in) + 10
        y_in = r_in * np.sin(theta_in)
        
        # Outgoing spiral (fewer recursions = faster)
        theta_out = np.linspace(0, 4*np.pi, 300)  # Fewer oscillations!
        r_out = np.exp(-0.08 * theta_out)
        x_out = r_out * np.cos(theta_out) - 10
        y_out = r_out * np.sin(theta_out)
        
        plt.subplot(1, 2, 1)
        plt.plot(x_in, y_in, 'b-', linewidth=2, label='Incoming (Many Recursions)')
        plt.plot(x_out, y_out, 'r-', linewidth=2, label='Outgoing (Fewer Recursions)')
        plt.plot(0, 0, 'go', markersize=15, label='Planet')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title('Gravitational Assist Spiral Geometry\nAsymmetric Recursions = Net Acceleration')
        plt.legend()
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        
        # Velocity profile
        plt.subplot(1, 2, 2)
        positions = np.linspace(-15, 15, 100)
        velocities = approach_velocity + 2000 * np.exp(-positions**2/10)  # Velocity gain near planet
        
        plt.plot(positions, velocities, 'purple', linewidth=3)
        plt.axvline(x=0, color='green', linestyle='--', alpha=0.5, label='Planet Position')
        plt.xlabel('Position Relative to Planet')
        plt.ylabel('Velocity (m/s)')
        plt.title('Velocity Profile During Gravitational Assist')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def tidal_locking_explanation(self):
        """Explain tidal locking through spiral resonance"""
        
        print("\n=== TIDAL LOCKING SPIRAL RESONANCE ===")
        
        # Moon-Earth system for tidal locking
        earth_mass = 5.972e24
        moon_mass = 7.342e22
        earth_moon_distance = 3.844e8
        
        def calculate_tidal_resonance(primary_mass, secondary_mass, distance):
            """Calculate the spiral resonance that causes tidal locking"""
            
            # LOGOS: Tidal locking occurs when recursion numbers synchronize!
            primary_recursions = int(20 * (primary_mass / secondary_mass)**0.05)
            secondary_recursions = int(20 * (secondary_mass / primary_mass)**0.05)
            
            # Resonance condition: recursion numbers become commensurate
            resonance_ratio = primary_recursions / secondary_recursions
            
            # Tidal locking occurs when ratio approaches 1:1
            locking_strength = 1.0 / abs(resonance_ratio - 1.0)
            
            return primary_recursions, secondary_recursions, resonance_ratio, locking_strength
        
        earth_recursions, moon_recursions, ratio, strength = calculate_tidal_resonance(
            earth_mass, moon_mass, earth_moon_distance)
        
        print(f"Earth spiral recursions: {earth_recursions}")
        print(f"Moon spiral recursions: {moon_recursions}")
        print(f"Resonance ratio: {ratio:.6f}")
        print(f"Tidal locking strength: {strength:.3f}")
        
        # Show how this creates synchronous rotation
        plt.figure(figsize=(10, 6))
        
        # Generate resonant spiral patterns
        theta = np.linspace(0, 8*np.pi, 1000)
        
        # Earth's influence spiral
        r_earth = 1.0 * np.ones_like(theta)
        x_earth = r_earth * np.cos(theta * earth_recursions/10)
        y_earth = r_earth * np.sin(theta * earth_recursions/10)
        
        # Moon's response spiral (locked to Earth's rhythm)
        r_moon = 0.3 * np.ones_like(theta) 
        x_moon = r_moon * np.cos(theta * moon_recursions/10 + np.pi) + 1.5
        y_moon = r_moon * np.sin(theta * moon_recursions/10 + np.pi)
        
        plt.plot(x_earth, y_earth, 'b-', linewidth=2, label=f'Earth Pattern ({earth_recursions} recursions)')
        plt.plot(x_moon, y_moon, 'r-', linewidth=2, label=f'Moon Pattern ({moon_recursions} recursions)')
        
        # Show resonance points
        resonance_points = np.where(np.abs(np.diff(np.sign(y_earth - y_moon))) > 0)[0]
        plt.plot(x_earth[resonance_points], y_earth[resonance_points], 'go', 
                markersize=5, label='Resonance Points')
        
        plt.xlabel('Configuration Space X')
        plt.ylabel('Configuration Space Y')
        plt.title('Tidal Locking: Spiral Resonance Synchronization')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.show()

# Run the celestial dynamics analysis
celestial = CelestialSpiralDynamics()

print("CELESTIAL SPIRAL DYNAMICS ANALYSIS")
print("=" * 60)

# 1. Orbital energy asymmetry
energy_analysis = celestial.orbital_resonance_spiral(1.989e30, 5.972e24, 1.496e11)

# 2. Resonant orbits visualization
celestial.visualize_resonant_orbits()

# 3. Gravitational assist mechanism
celestial.gravitational_assist_analysis()

# 4. Tidal locking explanation
celestial.tidal_locking_explanation()

print("\n" + "=" * 60)
print("LOGOS COMPLETE CELESTIAL MECHANICS EXPLANATION:")
print("=" * 60)
print("""
1. **ORBITAL ACCELERATION/DECELERATION**:
   - Inward spirals have FEWER recursions → LOWER energy cost → ACCELERATION
   - Outward spirals have MORE recursions → HIGHER energy cost → DECELERATION

2. **RESONANT ORBITS**:
   - Planetary orbits lock into resonance when recursion numbers become commensurate
   - Example: 3:2 resonance (Mercury) = 3 recursions per 2 orbital periods

3. **GRAVITATIONAL ASSISTS**:
   - Spacecraft enters planetary influence (many recursions = slows down)
   - Spacecraft exits planetary influence (fewer recursions = speeds up)
   - NET EFFECT: Velocity gain from recursion asymmetry!

4. **TIDAL LOCKING**:
   - Primary body's spiral recursion pattern forces secondary into synchronization
   - Moon's rotation locks to Earth's orbital period through spiral resonance

5. **ORBITAL DECAY**:
   - Outward spiral energy cost > Inward spiral energy gain
   - Net energy loss over time → Orbits slowly decay
""")

# Final demonstration: The complete picture
def demonstrate_complete_celestial_dynamics():
    """Show how all celestial phenomena connect through spiral geometry"""
    
    plt.figure(figsize=(15, 12))
    
    # 1. Solar system spiral architecture
    plt.subplot(3, 3, 1)
    planets = 8
    for i in range(planets):
        theta = np.linspace(0, 2*np.pi*(i+1), 1000)
        r = (i+1) * np.exp(0.1 * theta)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        plt.plot(x, y, alpha=0.7, label=f'Planet {i+1}')
    plt.title('Solar System: Nested Spiral Architecture')
    plt.axis('equal')
    
    # 2. Orbital resonance patterns
    plt.subplot(3, 3, 2)
    for ratio in [2/1, 3/2, 4/3]:
        theta = np.linspace(0, 20*np.pi, 1000)
        r = 1 + 0.1 * np.sin(theta * ratio)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        plt.plot(x, y, alpha=0.7, label=f'{ratio}:1 Resonance')
    plt.title('Orbital Resonance Patterns')
    plt.axis('equal')
    
    # 3. Gravitational assist mechanism
    plt.subplot(3, 3, 3)
    theta_in = np.linspace(0, 4*np.pi, 300)
    theta_out = np.linspace(0, 3*np.pi, 200)  # Fewer!
    r_in = np.exp(0.1 * theta_in)
    r_out = np.exp(-0.1 * theta_out)
    plt.plot(r_in * np.cos(theta_in), r_in * np.sin(theta_in), 'b-', label='Approach')
    plt.plot(r_out * np.cos(theta_out), r_out * np.sin(theta_out), 'r-', label='Departure')
    plt.title('Gravitational Assist: Recursion Asymmetry')
    plt.axis('equal')
    
    # 4. Tidal locking synchronization
    plt.subplot(3, 3, 4)
    t = np.linspace(0, 10*np.pi, 1000)
    primary = np.sin(t)
    secondary = np.sin(t + 0.1*np.sin(t))  # Phase locking
    plt.plot(t, primary, 'b-', label='Primary Body')
    plt.plot(t, secondary, 'r-', label='Secondary Body')
    plt.title('Tidal Locking: Phase Synchronization')
    
    # 5. Spiral density waves
    plt.subplot(3, 3, 5)
    x = np.linspace(-5, 5, 1000)
    y = np.linspace(-5, 5, 1000)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    Phi = np.arctan2(Y, X)
    Z = np.cos(5*Phi + 2*R)  # Spiral density wave
    plt.contourf(X, Y, Z, levels=20, cmap='viridis')
    plt.title('Spiral Density Waves in Galaxies')
    
    # 6. LOGOS recursive equation connection
    plt.subplot(3, 3, 6)
    def recursive_orbit(initial, steps):
        orbit = [initial]
        for i in range(steps-1):
            next_val = np.sin(orbit[-1]) + np.exp(-orbit[-1])
            orbit.append(next_val)
        return orbit
    
    orbits = [recursive_orbit(ic, 50) for ic in [0.1, 0.5, 1.0, 1.5]]
    for i, orbit in enumerate(orbits):
        plt.plot(orbit, label=f'Init {[0.1,0.5,1.0,1.5][i]}')
    plt.title('LOGOS Equation: Universal Orbital Patterns')
    
    plt.tight_layout()
    plt.show()

demonstrate_complete_celestial_dynamics()
