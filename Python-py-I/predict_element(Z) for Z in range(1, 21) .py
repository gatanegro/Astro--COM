import numpy as np

# Constants
LZ = 1.23498          # Attractor constant
HQS = 0.00235         # Fluctuation strength
E0 = 511e3            # Base energy = electron mass (eV)

def predict_element(Z):
    """Predict properties for atomic number Z using COM."""
    # Energy calculation
    energy = E0 * (LZ ** Z) * (1 + HQS * np.sin(4 * np.pi * Z))
    
    # Element type (metal/nonmetal)
    element_type = "metal" if np.sin(4 * np.pi * Z) > 0 else "nonmetal"
    
    # Stability (cycles after Collatz steps)
    stability = "stable" if Z in {1, 2, 4, 8} else "unstable"
    
    return {
        "Z": Z,
        "energy (eV)": energy,
        "type": element_type,
        "stability": stability
    }

# Generate predictions for Z=1 to 20
elements_com = [predict_element(Z) for Z in range(1, 21)]

# Print results
print("Z | Energy (eV)       | Type      | Stability")
print("--------------------------------------------")
for elem in elements_com:
    print(f"{elem['Z']:2d} | {elem['energy (eV)']:.3e} | {elem['type']:8s} | {elem['stability']}")