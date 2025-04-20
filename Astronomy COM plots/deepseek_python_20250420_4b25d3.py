def com_mass_map(flux_data, wavelength, eta=0.235):
    """Converts JWST flux to mass density in wave-amplitude space"""
    amplitude = np.sqrt(flux_data)
    phase = 2*np.pi * (wavelength - wavelength.min())/(wavelength.ptp())
    psi = amplitude * np.exp(1j*phase)
    return np.abs(psi)**2 * (1 + eta*np.sin(phase))**2

# Application to JWST NIRCam data
mass_density = com_mass_map(jwst_flux, jwst_wavelengths)