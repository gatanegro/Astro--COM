import numpy as np
from scipy.special import erf

def quantum_collatz(n, max_iter=1000):
    """Generates quantum-perturbed Collatz sequence with HQS-LZ effects"""
    sequence = []
    for _ in range(max_iter):
        if n == 1:
            break
        if n % 2 == 0:
            # Apply HQS perturbation to even operation
            q_factor = 1 + 0.235*np.random.normal(0, 0.01)
            n = int(n/2 * q_factor)
        else:
            # Apply LZ scaling to odd operation
            lz_scale = 1.23498 * (1 - 0.01*erf(n/100))
            n = int((3*n + 1) * lz_scale)
        sequence.append(n)
    return sequence

def octave_mapping(value, layer):
    """Enhanced octave mapping with field effects"""
    hqs_mod = 0.235 * np.sin(layer * np.pi/4)
    lz_scale = 1.23498 ** layer
    scaled_val = (value * lz_scale) % (9 + hqs_mod)
    angle = 2*np.pi * (scaled_val / (9 + hqs_mod))
    radius = np.sqrt(layer + 1) * (1 + 0.1175*np.cos(angle))  # HQS/2 modulation
    x = radius * np.cos(angle)
    y = radius * np.sin(angle)
    z = layer * (1 + 0.0785*layer)  # Stacking with LZ-derived spacing
    return x, y, z