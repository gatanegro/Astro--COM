import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp

# Parameters
N = 512                          # Grid points
L = 10.0                         # System size
x = np.linspace(0, L, N)         # Spatial grid
LZ = 1.23498                     # Nonlinearity strength
HQS = 0.00235                    # Damping coefficient
CMB_baseline = 2.725             # CMB temperature (energy floor)

# Initial condition: Gaussian perturbation + CMB
Psi0 = CMB_baseline + 0.01 * np.exp(-(x - L/2)**2) * np.sin(4 * np.pi * x / L)

# Nonlinear wave equation
def wave_eq(t, y):
    Psi, dPsi_dt = y[:N], y[N:]
    d2Psi_dx2 = np.gradient(np.gradient(Psi, x), x)  # Laplacian
    self_interaction = LZ * (Psi - CMB_baseline)**3  # Nonlinear term
    damping = HQS * dPsi_dt
    ddPsi = d2Psi_dx2 - self_interaction - damping
    return np.concatenate([dPsi_dt, ddPsi])

# Solve the PDE
sol = solve_ivp(wave_eq, [0, 50], np.concatenate([Psi0, np.zeros(N)]), 
                t_eval=np.linspace(0, 50, 100), method='BDF')

# Set up blue-themed plot
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(12, 6))
ax.set_facecolor('#0a0a2a')  # Dark blue background
line, = ax.plot(x, sol.y[:N, 0], 'cyan', lw=1.5, alpha=0.8)
ax.set_ylim(2.724, 2.726)
ax.set_xlabel("Position on Horizon", fontsize=12, color='white')
ax.set_ylabel("Energy Density Ψ", fontsize=12, color='white')
ax.set_title("Nonlinear Energy Resonance (Standing Waves = Particles)", 
             fontsize=14, color='white')
ax.grid(True, color='dodgerblue', alpha=0.2)

# Animation
def update(frame):
    line.set_ydata(sol.y[:N, frame])
    return line,

ani = FuncAnimation(fig, update, frames=len(sol.t), interval=50, blit=True)
plt.tight_layout()
plt.show()