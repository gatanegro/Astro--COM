import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 10          # Spatial domain length (space amplitude)
T = 10          # Temporal domain length (time longitude)
Nx = 200        # Number of spatial points
Nt = 200        # Number of temporal points
k = 2 * np.pi / 5  # Wave number (spatial frequency)
omega = 2 * np.pi / T  # Angular frequency (temporal oscillations)

x = np.linspace(0, L, Nx)  # Space grid
t = np.linspace(0, T, Nt)  # Time grid
X, T_grid = np.meshgrid(x, t)  # Create 2D grid for space and time

# Define the boomerang energy node as a localized oscillator
A = np.exp(-((X - L / 2) ** 2) / (2 * (L / 10) ** 2))  # Amplitude envelope (localized)
E_boomerang = A * np.sin(k * X - omega * T_grid)  # Energy density of the boomerang

# Define the background FIELD oscillation
E_field = 0.5 * np.sin(k * X + omega * T_grid)  # Background oscillatory field

# Total FIELD energy
E_total = E_boomerang + E_field

# Plot the energy redistribution over space and time (surface plot)
plt.figure(figsize=(10, 6))
plt.imshow(E_total, extent=[0, L, 0, T], aspect='auto', origin='lower', cmap='jet')
plt.colorbar(label='Energy Density')
plt.title('Energy Density in THE FIELD with Boomerang Node')
plt.xlabel('Space (Amplitude)')
plt.ylabel('Time (Longitude)')
plt.show()

# Dynamic 2D plot of energy node behavior over time
for i in range(0, Nt, 10):  # Every 10th time step
    plt.figure(figsize=(8, 4))
    plt.plot(x, E_total[i, :], linewidth=1.5)
    plt.ylim([-1.5, 1.5])
    plt.title(f'Energy Profile at Time {t[i]:.2f}')
    plt.xlabel('Space (Amplitude)')
    plt.ylabel('Energy Density')
    plt.grid()
    # 1. Analyze Energy Contributions

# Separate plots for boomerang node energy and background field energy
plt.figure(figsize=(10, 6))
plt.imshow(E_boomerang, extent=[0, L, 0, T], aspect='auto', origin='lower', cmap='viridis')
plt.colorbar(label='Boomerang Node Energy Density')
plt.title('Boomerang Node Energy Density')
plt.xlabel('Space (Amplitude)')
plt.ylabel('Time (Longitude)')
plt.show()

plt.figure(figsize=(10, 6))
plt.imshow(E_field, extent=[0, L, 0, T], aspect='auto', origin='lower', cmap='plasma')
plt.colorbar(label='Background Field Energy Density')
plt.title('Background Field Energy Density')
plt.xlabel('Space (Amplitude)')
plt.ylabel('Time (Longitude)')
plt.show()

# 2. Temporal and Spatial Analysis

# Energy density at a fixed spatial point (e.g., midpoint of the domain) over time
mid_point_idx = Nx // 2  # Index for the spatial midpoint
plt.figure(figsize=(10, 4))
plt.plot(t, E_total[:, mid_point_idx], label='Total Energy')
plt.plot(t, E_boomerang[:, mid_point_idx], label='Boomerang Node Energy', linestyle='--')
plt.plot(t, E_field[:, mid_point_idx], label='Background Field Energy', linestyle=':')
plt.title(f'Energy Density Over Time at Space Midpoint (x = {x[mid_point_idx]:.2f})')
plt.xlabel('Time (Longitude)')
plt.ylabel('Energy Density')
plt.legend()
plt.grid()
plt.show()

# Energy density across space at a fixed moment in time (e.g., halfway through the simulation)
mid_time_idx = Nt // 2  # Index for the temporal midpoint
plt.figure(figsize=(10, 4))
plt.plot(x, E_total[mid_time_idx, :], label='Total Energy')
plt.plot(x, E_boomerang[mid_time_idx, :], label='Boomerang Node Energy', linestyle='--')
plt.plot(x, E_field[mid_time_idx, :], label='Background Field Energy', linestyle=':')
plt.title(f'Energy Density Across Space at Time t = {t[mid_time_idx]:.2f}')
plt.xlabel('Space (Amplitude)')
plt.ylabel('Energy Density')
plt.legend()
plt.grid()
plt.show()

# 3. Summary Statistics

# Compute maximum, minimum, and mean energy density over time and space
max_energy_time = E_total.max(axis=1)  # Max energy density for each time step
min_energy_time = E_total.min(axis=1)  # Min energy density for each time step
mean_energy_time = E_total.mean(axis=1)  # Mean energy density for each time step

plt.figure(figsize=(10, 4))
plt.plot(t, max_energy_time, label='Max Energy', linewidth=1.5)
plt.plot(t, min_energy_time, label='Min Energy', linewidth=1.5)
plt.plot(t, mean_energy_time, label='Mean Energy', linewidth=1.5)
plt.title('Energy Density Statistics Over Time')
plt.xlabel('Time (Longitude)')
plt.ylabel('Energy Density')
plt.legend()
plt.grid()
plt.show()