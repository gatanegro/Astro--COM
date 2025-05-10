import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigs

# Define Collatz operator (simplified 10x10 matrix)
n = 10
diagonals = [np.ones(n), np.zeros(n)]
offsets = [0, 1]
C_op = diags(diagonals, offsets, shape=(n, n)).toarray()

# Add 3n+1 transitions (example: odd indices)
for i in range(1, n, 2):
    if 3*i + 1 < n:
        C_op[i, 3*i + 1] = 1

# Compute eigenvalues
eigenvalues = eigs(C_op, k=4, which='LM')[0]  # Largest magnitude
print("Top Eigenvalues:", eigenvalues)