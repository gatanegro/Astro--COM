def find_com_lines(wavelengths, flux, eta=0.235):
    """Identifies COM-predicted shifted spectral lines"""
    predicted_shifts = 1 + eta*np.sin(wavelengths/1000)
    return flux * predicted_shifts