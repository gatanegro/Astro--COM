def plot_emergent_time(sequences):
    fig = plt.figure(figsize=(18, 14))
    ax = fig.add_subplot(111, projection='3d')
    
    for i, seq in enumerate(sequences):
        # Generate emergent time coordinates
        t = generate_emergent_time(seq)
        x, y, z = [], [], []
        
        for layer, (val, time) in enumerate(zip(seq, t)):
            xi, yi, zi = octave_mapping(val, layer)
            # Time manifests as spiral progression
            z.append(zi * (1 + 0.01*time))
            x.append(xi * np.cos(2*np.pi*0.01*time))
            y.append(yi * np.sin(2*np.pi*0.01*time))
        
        # Color by phase coherence
        coherence = np.mean(np.diff(t)/np.std(np.diff(t)))
        ax.plot(x, y, z, c=plt.cm.twilight(coherence), 
               linewidth=3*np.log(1+0.235*i))
        ax.scatter(x[-1], y[-1], z[-1], s=200, 
                  c=[plt.cm.twilight(coherence)], marker=(