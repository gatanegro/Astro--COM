import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# =====================
# COLLATZ CORE FUNCTIONS
# =====================
def generate_collatz_sequence(n):
    """Generate full Collatz sequence for number n"""
    sequence = [n]
    while n != 1:
        n = 3 * n + 1 if n % 2 else n // 2
        sequence.append(n)
    return sequence

def reduce_to_single_digit(value):
    """Added missing digit reduction function"""
    return (value - 1) % 9 + 1

def compute_energy(sequence):
    """Calculate topological energy of a sequence"""
    return len(sequence) * np.log(len(sequence) + 0.5772)  # Fixed parenthesis

def compute_eta(sequence):
    """Simulate η-invariant from sequence oscillations"""
    if len(sequence) < 2:
        return 0.0
    oscillations = sum(1 if sequence[i] > sequence[i-1] else -1 for i in range(1, len(sequence)))
    return np.arctan(oscillations) / (2 * np.pi)

# ======================
# TOPOLOGICAL DATA (MOCK)
# ======================
MILNOR_EXOTIC_SPHERES = {
    1: {"eta": 0.0357, "name": "Milnor's First Exotic 7-Sphere"},
    2: {"eta": 0.8929, "name": "Milnor's Second Exotic 7-Sphere"},
    3: {"eta": -0.1429, "name": "Connected Sum Exotic Sphere"}
}

# ====================
# PARAMETERS & ANALYSIS
# ====================
LZ_SCALE = 1.23489
NUM_RANGE = (1, 1000)  # Reduced for faster execution
ETA_TOLERANCE = 0.05    # Increased tolerance

# Generate Collatz data
collatz_data = {}
for n in range(NUM_RANGE[0], NUM_RANGE[1] + 1):
    seq = generate_collatz_sequence(n)
    collatz_data[n] = {
        "sequence": seq,
        "energy": compute_energy(seq),
        "eta": compute_eta(seq)
    }

# Find high-energy nodes
energy_values = np.array([d["energy"] for d in collatz_data.values()])
energy_threshold = np.percentile(energy_values, 99)
high_energy_nodes = [n for n, d in collatz_data.items() if d["energy"] > energy_threshold]

# ==============
# 3D VISUALIZATION
# ==============
def map_to_octave(value, layer):
    """Convert number to 3D octave coordinates"""
    reduced = reduce_to_single_digit(value)
    angle = (reduced / 9) * 2 * np.pi  # Fixed modulo 9 calculation
    radius = LZ_SCALE * (layer + 1)
    return (
        radius * np.cos(angle),
        radius * np.sin(angle),
        layer
    )

fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(111, projection='3d')

# Plot all sequences
for n, data in collatz_data.items():
    positions = [map_to_octave(v, i) for i, v in enumerate(data["sequence"])]
    x, y, z = zip(*positions)
    ax.plot(x, y, z, color=plt.cm.plasma(data["energy"]/max(energy_values)), alpha=0.5)

# Highlight matches
matches = []
for n in high_energy_nodes:
    data = collatz_data[n]
    for sphere_id, sphere_data in MILNOR_EXOTIC_SPHERES.items():
        if abs(data["eta"] - sphere_data["eta"]) < ETA_TOLERANCE:
            matches.append(n)
            positions = [map_to_octave(v, i) for i, v in enumerate(data["sequence"])]
            x, y, z = zip(*positions)
            ax.plot(x, y, z, color='lime', linewidth=2)
            ax.scatter(x[-1], y[-1], z[-1], s=100, c='red', marker='*')

ax.set_title(f"Collatz-Topology Interface\nMatches Found: {len(matches)}")
plt.show()

print("High-energy nodes:", high_energy_nodes)
print("Matching exotic spheres:", matches)