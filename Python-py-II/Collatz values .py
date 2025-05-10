import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def collatz(n):
    sequence = []
    while n != 1:
        sequence.append(n)
        n = n // 2 if n % 2 == 0 else 3 * n + 1
    sequence.append(1)
    return sequence

# Generate Collatz sequences for n=1 to 20
sequences = [collatz(i) for i in range(1, 21)]

# Create 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i, seq in enumerate(sequences):
    x = [(-1)**k * np.log(n) for k, n in enumerate(seq)]  # Oscillation
    y = np.arange(len(seq))                                # Steps (time)
    z = seq                                                # Collatz values (energy)
    ax.plot(x, y, z, label=f'Collatz {i+1}')

ax.set_xlabel('X (Horizontal Oscillation)')
ax.set_ylabel('Iteration Step')
ax.set_zlabel('Collatz Value')
plt.show()