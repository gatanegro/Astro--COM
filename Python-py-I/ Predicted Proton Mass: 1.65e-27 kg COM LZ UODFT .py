LZ = 1.23498
c = 3e8  # m/s (phase velocity)
rho_planck = 5e96  # Planck energy density (kg/m³)
V_proton = 1e-45    # Proton volume (m³)

rho_proton = rho_planck * (LZ ** 40)
m_proton = (rho_proton * V_proton) / (LZ * c) ** 2
print(f"Predicted Proton Mass: {m_proton:.2e} kg")
print(f"Actual Proton Mass: 1.67e-27 kg")