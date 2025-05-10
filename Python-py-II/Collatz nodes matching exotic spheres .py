import numpy as np

# Mock data for Milnor's exotic 7-spheres (manual entry)
exotic_spheres = {
    # Format: SphereID: [Chern_class, Pontryagin_class]
    1: [3, 28],
    2: [5, 112],
    3: [7, 224],  # Hypothetical values for demonstration
}

def generate_collatz_sequence(n):
    sequence = [n]
    while n != 1:
        n = 3 * n + 1 if n % 2 else n // 2
        sequence.append(n)
    return sequence

def compute_energy(sequence):
    return len(sequence) * np.log(len(sequence) + 1)  # +1 to avoid log(0)

# Generate Collatz sequences and energy for numbers 1 to 10,000
collatz_energy = {}
for n in range(1, 10001):
    seq = generate_collatz_sequence(n)
    collatz_energy[n] = compute_energy(seq)

# Find high-energy nodes (top 0.1%)
energy_values = np.array(list(collatz_energy.values()))
threshold = np.percentile(energy_values, 99.9)
high_energy_nodes = [n for n, e in collatz_energy.items() if e > threshold]

# Match nodes to exotic spheres via Pontryagin class mod 28
matches = []
for node in high_energy_nodes:
    hypothetical_pontryagin = node % 28  # Simplified mapping
    for sphere_id, data in exotic_spheres.items():
        if data[1] == hypothetical_pontryagin:
            matches.append((node, sphere_id))

print("Collatz nodes matching exotic spheres:")
print(matches)