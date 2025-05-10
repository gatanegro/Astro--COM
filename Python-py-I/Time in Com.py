import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def angle_to_vector(theta):
    return np.array([np.cos(np.radians(theta)), np.sin(np.radians(theta))])  # Fixed

def vector_to_angle(v):
    return np.degrees(np.arctan2(v[1], v[0])) % 360

def damped_oscillator(y, t, Lz, theta_com):
    theta, dtheta_dt = y
    k = 1.0  # Stiffness
    d2theta_dt2 = -Lz * dtheta_dt - k * (theta - theta_com)
    return [dtheta_dt, d2theta_dt2]

def smooth_angle_transition(theta_start, theta_end, Lz, t_max=10, dt=0.1):
    # Compute COM of start and end angles
    v_start = angle_to_vector(theta_start)
    v_end = angle_to_vector(theta_end)
    v_com = (v_start + v_end) / 2
    theta_com = vector_to_angle(v_com)
    
    # Solve ODE: y = [theta, dtheta/dt]
    t = np.arange(0, t_max, dt)
    y0 = [theta_start, 0]  # Initial angle and velocity
    sol = odeint(damped_oscillator, y0, t, args=(Lz, theta_com))
    theta_trajectory = sol[:, 0] % 360  # Ensure angles wrap correctly
    
    return t, theta_trajectory

# Example usage
theta_start = 350
theta_end = 10
Lz = 1.23489  # Universal damping rate

t, angles = smooth_angle_transition(theta_start, theta_end, Lz)

# Plot
plt.figure(figsize=(10, 5))
plt.plot(t, angles, 'b-', label=f"Transition (Lᴢ={Lz})")
plt.xlabel("Time")
plt.ylabel("Angle (degrees)")
plt.title(f"Smooth Transition: {theta_start}° → {theta_end}°")
plt.legend()
plt.grid(True)
plt.show()