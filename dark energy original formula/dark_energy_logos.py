import numpy as np
HQS =  0.2355433068453462
LZ = 1.23488369648610768
alpha = 0.0072973525643

# Compute the sum S with maximum precision
S = (np.pi / 2) + LZ + np.sqrt(alpha) + (np.pi / 100)
# Compute Omega_Lambda
omega_lambda = HQS * S
print(f"S = {S:.16f}")
print(f"Ω_Λ = {omega_lambda:.16f}")


