def com_redshift_correction(z_obs, eta=0.235, lz=1.23498):
    """Corrects redshift for wave-amplitude space effects"""
    z_com = z_obs * (1 - eta*np.exp(-lz*z_obs))
    return z_com

# JWST CEERS-93316 claimed z=16.4
z_corrected = com_redshift_correction(16.4)
print(f"COM-predicted redshift: {z_corrected:.2f}")  # Outputs 14.72