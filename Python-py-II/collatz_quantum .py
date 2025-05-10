# Save as collatz_quantum.py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

LZ, HQS = 1.23498, 0.235
def collatz(n):
    seq = [n]
    while n != 1 and len(seq) < 100:
        n = n//2 if n%2==0 else 3*n+1
        seq.append(n)
    return seq

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for n in range(1, 51):
    x, y, z = [], [], []
    for layer, val in enumerate(collatz(n)):
        phase = ((val-1)%9 +1)/9 * 2*np.pi * HQS
        radius = LZ**(-layer/10)
        x.append(radius * np.cos(phase))
        y.append(radius * np.sin(phase))
        z.append(layer)
    ax.plot(x, y, z, linewidth=0.5)
plt.savefig('collatz_trajectories.png')
plt.show()