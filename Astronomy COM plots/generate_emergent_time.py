def generate_emergent_time(sequence, r0=1.0):
    """Generates time coordinates from frequency wave structure"""
    omega_0 = (c/r0) * (0.235*1.23498/(2*np.pi))**(1/3)
    time_points = []
    cumulative_phase = 0
    
    for k, value in enumerate(sequence):
        phase_k = 2*np.pi*0.235*1.23498 * (1 - 0.235/2 * np.cos(np.pi*k/4))
        cumulative_phase += phase_k
        t_k = cumulative_phase / omega_0
        time_points.append(t_k)
    
    return np.array(time_points)