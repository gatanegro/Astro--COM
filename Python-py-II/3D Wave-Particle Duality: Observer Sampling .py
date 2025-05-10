import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# 3D grid
x = np.linspace(0, 2 * np.pi, 30)
y = np.linspace(0, 2 * np.pi, 30)
z = np.linspace(0, 2 * np.pi, 30)
X, Y, Z = np.meshgrid(x, y, z)

# Wave parameters (space = amplitude, time = frequency)
frequencies = [(1, 2, 3), (2, 1, 2.5), (1.5, 2.5, 1)]
amplitudes = [1, 0.7, 0.5]
phases = [0, np.pi/4, np.pi/2]

def wave_field(X, Y, Z, t):
    field = np.zeros_like(X)
    for (fx, fy, fz), amp, phase in zip(frequencies, amplitudes, phases):
        field += amp * np.sin(fx * X + fy * Y + fz * Z + phase + t)
    return field

# Observer parameters
threshold = 1.6  # Detection threshold for "particle"
num_frames = 60  # Number of animation frames

# Set up the figure and 3D axis
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim([0, 2 * np.pi])
ax.set_ylim([0, 2 * np.pi])
ax.set_zlim([0, 2 * np.pi])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Wave-Particle Duality: Observer Sampling')

# Initial scatter plot (empty)
scat = ax.scatter([], [], [], s=20, c='red', alpha=0.8)

def update(frame):
    t = 2 * np.pi * frame / num_frames
    wf = wave_field(X, Y, Z, t)
    # Find "particle" positions where amplitude exceeds threshold
    idx = np.where(np.abs(wf) > threshold)
    x_part, y_part, z_part = X[idx], Y[idx], Z[idx]
    scat._offsets3d = (x_part, y_part, z_part)
    ax.set_title(f'3D Wave-Particle Duality\nTime slice {frame+1}/{num_frames}')
    return scat,

ani = FuncAnimation(fig, update, frames=num_frames, interval=100, blit=False)

plt.show()