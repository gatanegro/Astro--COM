import numpy as np
import matplotlib.pyplot as plt

def angle_to_vector(theta_deg):
    theta_rad = np.radians(theta_deg)
    return np.array([np.cos(theta_rad), np.sin(theta_rad)])

def vector_to_angle(v):
    return np.degrees(np.arctan2(v[1], v[0])) % 360

def smooth_transition(theta_start, theta_end, num_points=100):
    s = np.linspace(0, 1, num_points)
    v_start = angle_to_vector(theta_start)
    v_end = angle_to_vector(theta_end)
    
    # Quadratic COM interpolation
    v_interp = np.array([(1 - si**2) * v_start + si**2 * v_end for si in s])
    v_norm = v_interp / np.linalg.norm(v_interp, axis=1)[:, None]
    
    angles = [vector_to_angle(v) for v in v_norm]
    return angles

# Example usage
theta_start = 30  # degrees
theta_end = 150   # degrees
angles = smooth_transition(theta_start, theta_end)

# Plot
plt.figure(figsize=(10, 5))
plt.plot(angles, 'b-', label=f"Transition: {theta_start}° → {theta_end}°")
plt.xlabel("Arc Length Parameter (s)")
plt.ylabel("Angle (degrees)")
plt.title("Euler Spiral Approximation via COM Averaging")
plt.grid(True)
plt.legend()
plt.show()