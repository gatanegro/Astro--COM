import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Constants
h = 4.1357e-15  # Planck's constant (eV·s)
c = 3e8  # Speed of light (m/s)
m_e = 9.11e-31  # Electron mass (kg)
m_p = 1.672e-27  # Proton mass (kg)
E_e = 0.511e6  # Electron rest energy (eV)
E_p = 938e6  # Proton rest energy (eV)
LZ = 1.23498  # Loop Zero constant
HQS = 0.00235  # Harmonic Quantum Shift (0.235%)
stack_spacing = 1.0

# Recursive wave evolution
def recursive_wave_evolution(psi_0, max_iter=1000, tol=1e-6):
    psi = psi_0
    for n in range(max_iter):
        psi_next = np.sin(psi) + np.exp(-psi)
        if abs(psi_next - psi) < tol:
            return psi_next, n
        psi = psi_next
    return psi, max_iter

# Photon wave function
def photon_wave(x, t, k, omega, A=1.0):
    return A * np.exp(1j * (k * x - omega * t))

# Energy density from two photon waves
def energy_density(x, t, k1, omega1, k2, omega2, A1=1.0, A2=1.0):
    psi1 = photon_wave(x, t, k1, omega1, A1)
    psi2 = photon_wave(x, t, k2, omega2, A2)
    psi_total = psi1 + psi2
    return np.abs(psi_total)**2

# Collatz sequence generation
def generate_collatz_sequence(n):
    sequence = [n]
    while n != 1:
        if n % 2 == 0:
            n = n // 2
        else:
            n = 3 * n + 1
        sequence.append(n)
    return sequence

# Reduce to single-digit (octave reduction)
def reduce_to_single_digit(value):
    return (value - 1) % 9 + 1

# Map to octave structure
def map_to_octave(value, layer):
    angle = (value / 9) * 2 * np.pi
    x = np.cos(angle) * (layer + 1)
    y = np.sin(angle) * (layer + 1)
    return x, y

# Simulate particle creation
def simulate_particle_creation(E1, E2, particle_type="electron"):
    # Convert photon energies to frequencies
    nu1 = E1 / h
    nu2 = E2 / h
    omega1 = 2 * np.pi * nu1
    omega2 = 2 * np.pi * nu2
    k1 = omega1 / c
    k2 = omega2 / c

    # Compute total energy and threshold
    E_total = E1 + E2
    E_particle = E_e if particle_type == "electron" else E_p
    E_threshold = HQS * E_particle
    E_required = 2 * E_particle + E_threshold

    if E_total < E_required:
        print(f"Insufficient energy: {E_total:.2e} eV < {E_required:.2e} eV")
        return None

    # Compute energy density
    x = np.linspace(-1e-10, 1e-10, 100)
    t = 0
    density = energy_density(x, t, k1, omega1, k2, omega2)
    rho_max = np.max(density)
    psi_0 = rho_max / (E_total / h)  # Normalize initial state

    # Apply recursive wave evolution
    psi_final, n_iter = recursive_wave_evolution(psi_0)
    if abs(psi_final - LZ) > 1e-3:
        print(f"Warning: Wave evolution did not converge to LZ ({psi_final:.6f} != {LZ})")
    
    # Compute required iterations for energy scaling
    n = int(np.log(2 * E_particle / E_total) / np.log(LZ)) if E_total > 0 else 0
    sequence = generate_collatz_sequence(n + 1)
    positions = []
    for layer, value in enumerate(sequence):
        reduced_value = reduce_to_single_digit(value)
        x, y = map_to_octave(reduced_value, layer)
        z = layer * stack_spacing
        positions.append((x, y, z))

    return positions, psi_final

# Visualization
def plot_com(positions, particle_type):
    if positions is None:
        return
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    x_vals = [pos[0] for pos in positions]
    y_vals = [pos[1] for pos in positions]
    z_vals = [pos[2] for pos in positions]
    ax.plot(x_vals, y_vals, z_vals, label=f"{particle_type} Node")
    ax.scatter(x_vals, y_vals, z_vals, s=20, zorder=5)
    ax.set_title(f"COM for {particle_type} Creation")
    ax.set_xlabel("X (Horizontal Oscillation)")
    ax.set_ylabel("Y (Vertical Oscillation)")
    ax.set_zlabel("Z (Octave Layer)")
    plt.legend()
    plt.show()

# Test for electron and hydrogen
E1 = 0.512e6  # Photon energy (eV)
E2 = 0.512e6
positions_e, psi_e = simulate_particle_creation(E1, E2, "electron")
plot_com(positions_e, "Electron")

# For hydrogen (simplified: proton creation)
E1_p = 940e6
E2_p = 940e6
positions_p, psi_p = simulate_particle_creation(E1_p, E2_p, "proton")
plot_com(positions_p, "Proton")