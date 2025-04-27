"""
COM Framework Constants Module

This module defines the fundamental constants and parameters used in the
Continuous Oscillatory Model (COM) framework.

Author: Martin Doina
Date: April 24, 2025
"""

import numpy as np

# COM Framework Fundamental Constants
LZ = 1.23498  # Fundamental scaling constant (λ)
HQS = 0.235   # Harmonic Quantum Scalar (η)
HQS_LZ = HQS * LZ  # HQS threshold in absolute terms (≈ 0.29022)

# 24-step Fibonacci digital root pattern
FIBONACCI_PATTERN = [1, 1, 2, 3, 5, 8, 4, 3, 7, 1, 8, 9, 8, 8, 7, 6, 4, 1, 5, 6, 2, 8, 1, 9]
FIBONACCI_NORMALIZED = np.array(FIBONACCI_PATTERN) / 9.0  # Normalize to 0-1 range

# Physical constants
G = 6.67430e-11  # Gravitational constant in m^3 kg^-1 s^-2
c = 2.99792458e8  # Speed of light in m/s
M_SUN = 1.989e30  # Solar mass in kg
AU = 1.496e11    # Astronomical unit in meters

# Planetary data (semi-major axes in AU)
PLANETS = [
    {"name": "Mercury", "semi_major_axis": 0.39},
    {"name": "Venus", "semi_major_axis": 0.72},
    {"name": "Earth", "semi_major_axis": 1.00},
    {"name": "Mars", "semi_major_axis": 1.52},
    {"name": "Ceres", "semi_major_axis": 2.77},
    {"name": "Jupiter", "semi_major_axis": 5.20},
    {"name": "Saturn", "semi_major_axis": 9.54},
    {"name": "Uranus", "semi_major_axis": 19.18},
    {"name": "Neptune", "semi_major_axis": 30.07}
]

# Reference scales spanning quantum to cosmic
REFERENCE_SCALES = [
    {"name": "Proton", "scale": 1e-15, "color": "purple"},
    {"name": "Atom", "scale": 1e-10, "color": "blue"},
    {"name": "Virus", "scale": 1e-7, "color": "cyan"},
    {"name": "Human", "scale": 1, "color": "green"},
    {"name": "Earth", "scale": 1e7, "color": "yellow"},
    {"name": "Mercury Orbit", "scale": 0.39 * AU, "color": "orange"},
    {"name": "Jupiter Orbit", "scale": 5.2 * AU, "color": "red"},
    {"name": "Solar System", "scale": 40 * AU, "color": "brown"},
    {"name": "Galaxy", "scale": 1e21, "color": "gray"},
    {"name": "Observable Universe", "scale": 1e26, "color": "black"}
]
