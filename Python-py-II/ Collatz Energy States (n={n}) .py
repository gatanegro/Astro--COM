# Collatz spacetime mapping (simplified)
import numpy as np
import matplotlib.pyplot as plt

LZ = 1.23498
def collatz_sequence(n):
    sequence = []
    while n != 1:
        n = n // 2 if n % 2 == 0 else 3 * n + 1
        sequence.append((n - 1) % 9 + 1)  # Octave reduction
    return sequence

n = 7  # Try 7, 9, 15, etc.
sequence = collatz_sequence(n)
plt.plot(sequence, marker='o')
plt.title(f"Collatz Energy States (n={n})")
plt.xlabel("Step"); plt.ylabel("Energy Density (1-9)")
plt.show()