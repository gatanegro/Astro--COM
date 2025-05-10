import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# =====================
# COLLATZ CORE FUNCTIONS
# =====================
def generate_collatz_sequence(n):
    sequence = [n]
    while n != 1:
        n = 3 * n + 1 if n % 2 else n // 2
        sequence.append(n)
    return sequence

def reduce_to_single_digit(value):
    return (value - 1) % 9 + 1

def compute_energy(sequence):
    return len(sequence) * np.log(len(sequence) + 0.5772)

def compute_eta(sequence):
    if len(sequence) < 2:
        return 0.0
    oscillations = sum(1 if x > y else -1 for x, y in zip(sequence[1:], sequence[:-1]))
    return np.arctan(oscillations) / (2 * np.pi)

# ======================
# TOPOLOGICAL DATA (MOCK)
# ======================
MILNOR_EXOTIC_SPHERES = {
    1: {"eta": -0.1429, "name": "Exotic Sphere 3"}
}

# ====================
# PARAMETERS & ANALYSIS
# ====================
LZ_SCALE = 1.23489
NUM_RANGE = (1, 100)
ETA_TOLERANCE = 0.05

# Generate Collatz data
collatz_data = {}
for n in range(NUM_RANGE[0], NUM_RANGE[1] + 1):
    seq = generate_collatz_sequence(n)
    collatz_data[n] = {
        "sequence": seq,
        "energy": compute_energy(seq),
        "eta": compute_eta(seq)
    }

# Calculate energy values
energy_values = np.array([d["energy"] for d in collatz_data.values()])  # FIX ADDED

# Find high-energy nodes
energy_threshold = np.percentile(energy_values, 99)
high_energy_nodes = [n for n, d in collatz_data.items() if d["energy"] > energy_threshold]

# Find matches
matches = []
for n in high_energy_nodes:
    data = collatz_data[n]
    for sphere_id, sphere in MILNOR_EXOTIC_SPHERES.items():
        if abs(data["eta"] - sphere["eta"]) < ETA_TOLERANCE:
            matches.append(n)

# =============
# PRINT RESULTS
# =============
print("=== Topological Matches ===")
for n in matches:
    print(f"Number: {n}")
    print(f"Collatz η: {collatz_data[n]['eta']:.4f}")
    print(f"Sphere η: {MILNOR_EXOTIC_SPHERES[1]['eta']:.4f}")
    print(f"Sphere: {MILNOR_EXOTIC_SPHERES[1]['name']}\n")

# ==============
# 3D VISUALIZATION
# ==============
def map_to_octave(value, layer):
    reduced = reduce_to_single_digit(value)
    angle = (reduced / 9) * 2 * np.pi
    radius = LZ_SCALE * (layer + 1)
    return (radius*np.cos(angle), radius*np.sin(angle), layer)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

for n, data in collatz_data.items():
    positions = [map_to_octave(v, i) for i, v in enumerate(data["sequence"])]
    x, y, z = zip(*positions)
    ax.plot(x, y, z, color=plt.cm.plasma(data["energy"]/max(energy_values)), alpha=0.6)

plt.title("Collatz-Topology Interface")
plt.show()