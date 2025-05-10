import numpy as np

# Constants
LZ = 1.23498       # Scaling attractor (adjust empirically)
HQS = 0.00235      # Energy threshold for particle formation
OCTAVE_BASE = 9    # Numbers reduce to 1-9

def collatz_step(n):
    """Single Collatz step (wave oscillation)."""
    if n % 2 == 0:
        return n // 2
    else:
        return 3 * n + 1

def octave_reduce(n):
    """Reduce energy to root number (1-9)."""
    return (n - 1) % OCTAVE_BASE + 1

def simulate_photon_interaction(initial_energy, max_steps=1000):
    """Simulate a photon's Collatz evolution until loop detection."""
    energy = initial_energy
    history = []
    
    for step in range(max_steps):
        energy = collatz_step(energy)
        root_energy = octave_reduce(energy)
        history.append((step, energy, root_energy))
        
        # Check for stable loop (particle formation)
        if len(history) > 3 and root_energy in [3, 6, 9]:  # Tesla's resonant numbers
            print(f"LOOP DETECTED at step {step}: Energy={energy}, Root={root_energy}")
            break
    
    return history

# Run simulation for a photon (initial energy = 1 CEU ≈ 0.511 MeV)
photon_history = simulate_photon_interaction(initial_energy=1)

# Print energy transitions
print("\nEnergy Evolution:")
for step, energy, root in photon_history[:10]:  # Show first 10 steps
    print(f"Step {step}: Energy={energy}, Root={root}")

# Output particle-like states (roots 1,5,9)
particles = [(step, root) for step, _, root in photon_history if root in {1, 5, 9}]
print("\nParticle-like States (Roots 1,5,9):", particles)