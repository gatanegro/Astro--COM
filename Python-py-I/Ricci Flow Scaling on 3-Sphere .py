import numpy as np
import matplotlib.pyplot as plt

# Initial radius
r0 = 1.0

# Time array from 0 to extinction time
T = r0**2 / 4
t = np.linspace(0, T, 500)

# Radius evolution under Ricci flow
r = np.sqrt(r0**2 - 4*t)

# Scaling factor s(t)
s = r / r0

plt.plot(t, s, label='Scaling factor s(t)')
plt.axhline(0.765, color='red', linestyle='--', label='Threshold 0.765 (23.5% shrink)')
plt.xlabel('Time t')
plt.ylabel('Relative radius s(t)')
plt.title('Ricci Flow Scaling on 3-Sphere')
plt.legend()
plt.grid(True)
plt.show()
