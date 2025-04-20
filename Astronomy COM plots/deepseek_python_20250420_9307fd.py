def plot_4d_collatz(sequences):
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    time_axis = np.linspace(0, 1, len(sequences[0]))
    
    for i, seq in enumerate(sequences):
        x, y, z = [], [], []
        for t, (layer, val) in enumerate(zip(time_axis, seq)):
            xi, yi, zi = octave_mapping(val, layer)
            # Add time dilation effect
            zi *= (1 + 0.235*t)**0.5
            x.append(xi); y.append(yi); z.append(zi)
        
        # Color by HQS-LZ energy
        color_val = 0.235 * i % 1
        ax.plot(x, y, z, c=plt.cm.plasma(color_val), 
               linewidth=2-1.23498**(-i/10))
        ax.scatter(x[-1], y[-1], z[-1], s=50*i**0.5, 
                  c=[plt.cm.plasma(color_val)], marker='*')
    
    ax.set_xlabel('X (HQS Phase)')
    ax.set_ylabel('Y (LZ Amplitude)') 
    ax.set_zlabel('Z (Octave Layer + Time)')
    plt.title('4D COM-HQS-LZ Quantum Collatz Trajectories')
    plt.show()