import numpy as np
import matplotlib.pyplot as plt

# Parameters
x = np.linspace(0, 10, 1000)           # Space axis
frequencies = [2, 3.5, 5]              # Frequencies (time as frequency)
amplitudes = [1, 0.7, 0.5]             # Amplitudes (space as amplitude)
phases = [0, np.pi/4, np.pi/2]         # Phase offsets

def wave_function(x, t):
    """Sum of sinusoids representing photon waves."""
    return sum(a * np.sin(2 * np.pi * f * t + p + x)
               for a, f, p in zip(amplitudes, frequencies, phases))

# Observer parameters
sampling_interval = 0.2                 # "Frame rate" of observer (Δt)
num_samples = 20                        # Number of photograms
threshold = 1.5                         # Detection threshold for "particle"

# Prepare the plot
fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
plt.subplots_adjust(hspace=0.4)

for i, t in enumerate(np.arange(0, sampling_interval * num_samples, sampling_interval)):
    wf = wave_function(x, t)
    if i < 3:  # Show first 3 slices
        axes[i].plot(x, wf, label=f'Photogram at t={t:.2f}')
        # Mark detected "particles"
        particles = np.where(np.abs(wf) > threshold)[0]
        axes[i].scatter(x[particles], wf[particles], color='red', label='Detected "particle"')
        axes[i].legend()
        axes[i].set_ylabel('Amplitude')
    elif i == 3:  # Overlay all photograms for "spaghetti" view
        axes[3].plot(x, wf, alpha=0.3, color='gray')
axes[3].set_title('All photograms overlayed: "Spaghetti wave"')
axes[3].set_xlabel('Space (x)')
axes[3].set_ylabel('Amplitude')

plt.suptitle("Wave-Particle Duality via Observer Sampling\n"
             "Space = amplitude, Time = frequency, Observation = time slicing")
plt.show()