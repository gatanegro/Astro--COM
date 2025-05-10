import math

# Parameters for inner planets (Mercury to Mars)
a0_inner = 0.387       # Mercury's semi-major axis (AU)
lambda_inner = 1.7     # Example geometric factor for inner planets
eta = 0.1              # Harmonic modulation amplitude
p = 3                  # Period for harmonic term

# Parameters for outer planets (Jupiter to Neptune)
a0_outer = 5.20        # Jupiter's semi-major axis (AU)
lambda_outer = 1.77    # Example geometric factor for outer planets

# Actual semi-major axes (AU) for comparison
actual_au = [
    0.387,   # Mercury
    0.723,   # Venus
    1.000,   # Earth
    1.524,   # Mars
    5.203,   # Jupiter
    9.537,   # Saturn
    19.191,  # Uranus
    30.068   # Neptune
]

planet_names = [
    "Mercury", "Venus", "Earth", "Mars",
    "Jupiter", "Saturn", "Uranus", "Neptune"
]

print(f"{'Planet':<8} {'n':<2} {'Model (AU)':>10} {'Actual (AU)':>12} {'% Error':>10}")
print("-" * 48)

for n, name in enumerate(planet_names):
    if n <= 3:
        theta_n = 2 * math.pi * n / p
        a_n = a0_inner * (lambda_inner ** n) * (1 + eta * math.sin(theta_n))
    else:
        a_n = a0_outer * (lambda_outer ** (n - 4))
    error_pct = 100 * (a_n - actual_au[n]) / actual_au[n]
    print(f"{name:<8} {n:<2} {a_n:10.3f} {actual_au[n]:12.3f} {error_pct:10.2f}")
