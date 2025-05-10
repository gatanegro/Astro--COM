import numpy as np
import matplotlib.pyplot as plt

def angle_to_vector(theta_deg):
    """Convert degrees to a unit vector."""
    theta_rad = np.radians(theta_deg)
    return np.array([np.cos(theta_rad), np.sin(theta_rad)])

def vector_to_angle(v):
    """Convert a unit vector to degrees (0-360)."""
    return np.degrees(np.arctan2(v[1], v[0])) % 360

def smooth_transition(theta_start, theta_end, num_points=100):
    """Generate smooth angles using quadratic COM averaging."""
    s = np.linspace(0, 1, num_points)
    v_start = angle_to_vector(theta_start)
    v_end = angle_to_vector(theta_end)
    
    # Quadratic COM interpolation: (1 - s²) * start + s² * end
    v_interp = np.array([(1 - si**2) * v_start + si**2 * v_end for si in s])
    v_norm = v_interp / np.linalg.norm(v_interp, axis=1)[:, np.newaxis]
    
    angles = np.array([vector_to_angle(v) for v in v_norm])
    return s, angles

def compute_curvature(angles, s):
    """Numerically compute curvature κ = dθ/ds."""
    dtheta = np.gradient(angles, s)
    return dtheta

# Example: Smooth transition from 30° to 150°
theta_start = 30
theta_end = 150
s, angles = smooth_transition(theta_start, theta_end, num_points=200)
curvature = compute_curvature(angles, s)

# Plotting
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Angle Transition Plot
ax1.plot(s, angles, 'b-', linewidth=2, label=f"{theta_start}° → {theta_end}°")
ax1.set_ylabel("Angle (degrees)")
ax1.set_title("Smooth Euler Spiral Approximation via COM Averaging")
ax1.grid(True)
ax1.legend()

# Curvature Plot (Validation)
ax2.plot(s, curvature, 'r--', linewidth=2, label="Curvature (κ)")
ax2.set_xlabel("Arc Length Parameter (s)")
ax2.set_ylabel("Curvature (dθ/ds)")
ax2.set_title("Curvature Linearity Check")
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.show()