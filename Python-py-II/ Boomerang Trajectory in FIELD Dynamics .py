from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

# Function to simulate a boomerang trajectory in Field dynamics
def boomerang_trajectory(time_steps=100, loops=2, height=10):
    """
    Simulates the trajectory of a boomerang in a spiral-like Field framework.
    :param time_steps: Number of time steps for the simulation.
    :param loops: Number of full loops the boomerang makes.
    :param height: Maximum vertical height of the boomerang trajectory.
    """
    # Time array for parametric equations
    t = np.linspace(0, loops * 2 * np.pi, time_steps)
    
    # Parametric equations for the trajectory
    x = 10 * np.cos(t)  # Horizontal oscillation (E-W)
    y = 10 * np.sin(t)  # Horizontal oscillation (N-S)
    z = height * (t / (loops * 2 * np.pi))  # Gradual upward scaling
    
    return x, y, z

# Generate the boomerang trajectory
x, y, z = boomerang_trajectory()

# Plot the trajectory
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot the boomerang path
ax.plot(x, y, z, color='blue', label="Boomerang Path (FIELD Dynamics)")
ax.scatter(x[0], y[0], z[0], color='green', s=100, label="Throw Point")  # Starting point
ax.scatter(x[-1], y[-1], z[-1], color='red', s=100, label="Return Point")  # End point

# Add labels and legend
ax.set_title("Boomerang Trajectory in FIELD Dynamics", fontsize=14)
ax.set_xlabel("X (Horizontal Oscillation)")
ax.set_ylabel("Y (Horizontal Oscillation)")
ax.set_zlabel("Z (Vertical Oscillation)")
ax.legend()

plt.show()