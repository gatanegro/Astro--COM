import numpy as np  
import matplotlib.pyplot as plt  

# Consciousness field (Ψ)  
x = np.linspace(0, 10, 1000)  
Ψ = np.exp(-(x-5)**2) * np.sin(10 * x)  # Localized awareness  

# Observe (collapse)  
def observe(Ψ, threshold=0.5):  
    return np.where(np.abs(Ψ) > threshold, Ψ, 0)  

# Render reality  
plt.plot(x, Ψ, 'b', label="Pure Ψ")  
plt.plot(x, observe(Ψ), 'r', label="Observed Reality")  
plt.legend(); plt.show()  