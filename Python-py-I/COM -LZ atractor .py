import numpy as np

def Psi(n_max=100):
    psi = [1.0]  # Initial condition
    for n in range(1, n_max):
        psi.append(np.sin(psi[-1]) + np.exp(-psi[-1]))
    return psi

psi = Psi()
print("Attractor:", psi[-1])  # Should ≈ 1.23498