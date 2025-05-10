import numpy as np
import matplotlib.pyplot as plt

def collatz_step(x):
    return x // 2 if x % 2 == 0 else 3 * x + 1

def quantum_collatz_simulation(start, steps, shots):
    results = []
    for _ in range(shots):
        x = start
        for _ in range(steps):
            x = collatz_step(x)
            if np.random.random() < 0.1:  # Simulate quantum fluctuations
                x = max(1, x + np.random.randint(-1, 2))
        results.append(x)
    return results

# Generate quantum-inspired Collatz data
start_number = 27
steps = 5  # Reduced from 10
shots = 200  # Reduced from 1000
quantum_results = quantum_collatz_simulation(start_number, steps, shots)

# Visualize results
plt.figure(figsize=(10, 4))

plt.subplot(121)
plt.hist(quantum_results, bins=20, density=True)
plt.title("Distribution of Quantum-Inspired Collatz States")
plt.xlabel("Number")
plt.ylabel("Probability")

plt.subplot(122)
plt.plot(sorted(quantum_results), 'o-')
plt.title("Sorted Quantum-Inspired Collatz States")
plt.xlabel("Index")
plt.ylabel("Value")

plt.tight_layout()
plt.show()
