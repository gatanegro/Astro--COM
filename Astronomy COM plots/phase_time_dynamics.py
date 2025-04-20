def plot_phase_time_dynamics(sequence):
    phases = []
    operator = TemporalCollatzOperator()
    
    for n in sequence:
        operator(n)
        phases.append(operator.phase_history[-1])
    
    times = generate_emergent_time(sequence)
    
    plt.figure(figsize=(12, 8))
    plt.polar(phases, times, c=times, cmap='twilight_shifted')
    plt.colorbar(label='Emergent Time')
    plt.title('Phase-Time Relationship in Temporal COM', pad=20)
    plt.show()