rho_planck = 5e96       # Planck energy density (kg/m³)
V_proton = 1e-45        # Proton volume (m³)
LZ = 1.23498
c = 3e8                 # m/s

m_p_com = (rho_planck * LZ**40 * V_proton) / (LZ * c)**2
print(f"COM Proton Mass: {m_p_com:.2e} kg")
print(f"Observed Proton Mass: 1.67e-27 kg")
