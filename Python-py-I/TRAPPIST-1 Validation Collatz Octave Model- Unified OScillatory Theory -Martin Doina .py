import numpy as np

# Constants
LZ = 1.23498 
HQS = 0.235
a0 = 0.0115  # TRAPPIST-1b's observed distance (AU)

def com_hqs_lz_orbit(n, phase_func=lambda n: np.sin(n * np.pi / 3)):
    """Predicts orbital distance for octave layer n with phase modulation."""
    return a0 * (LZ ** n) * (1 + HQS * phase_func(n))

# TRAPPIST-1 observed distances (AU) - NASA data
observed = [0.0115, 0.0158, 0.0223, 0.0293, 0.0385, 0.0469, 0.0619]
layers = np.arange(len(observed))

# Compare predictions
predicted = [com_hqs_lz_orbit(n) for n in layers]
residuals = (np.array(predicted) - np.array(observed)) / observed * 100  # % error

print("TRAPPIST-1 Validation:")
for n, obs, pred, err in zip(layers, observed, predicted, residuals):
    print(f"Layer {n}: Obs={obs:.4f} AU | Pred={pred:.4f} AU | Error={err:.2f}%")