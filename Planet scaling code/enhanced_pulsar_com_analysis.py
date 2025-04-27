"""
Enhanced Pulsar Frequency Analysis using the COM Framework

This script applies the Continuous Oscillatory Model (COM) framework to analyze
pulsar frequencies, testing whether the same mathematical patterns that describe
planetary spacing might also apply to these astrophysical objects.

Author: Martin Doina
Date: April 24, 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
import pandas as pd

# --- Step 1: Define COM frequency generation with different phase functions ---
def com_frequencies(base_freq, n_steps=24, phase_func='sin', lz=1.23498, hqs=0.235):
    """
    Generate frequencies using the COM framework with different phase functions.
    
    Parameters:
    - base_freq: Starting frequency (Hz)
    - n_steps: Number of frequency steps to generate
    - phase_func: Phase function to use ('sin', 'tanh', 'cos', or 'none')
    - lz: LZ constant (default: 1.23498)
    - hqs: HQS constant (default: 0.235)
    
    Returns:
    - List of frequencies following the COM pattern
    """
    frequencies = []
    
    for n in range(n_steps):
        # Apply different phase functions
        if phase_func == 'sin':
            phase = np.sin(2 * np.pi * n / 24)
        elif phase_func == 'cos':
            phase = np.cos(2 * np.pi * n / 24)
        elif phase_func == 'tanh':
            phase = np.tanh(n / 2)
        elif phase_func == 'none':
            phase = 0  # No phase modulation, pure exponential growth
        else:
            raise ValueError(f"Unknown phase function: {phase_func}")
        
        # Apply COM formula: base_freq * LZ^n * (1 + HQS * phase)
        freq = base_freq * (lz ** n) * (1 + hqs * phase)
        frequencies.append(freq)
    
    return frequencies

# --- Step 2: Load and prepare pulsar data ---
# Example pulsar frequencies (Hz)
# In a real analysis, you would load this from a file with more pulsars
pulsar_data = {
    'Name': ['Crab Pulsar', 'PSR B1937+21', 'PSR J0437-4715', 'PSR B0833-45', 'PSR B0531+21'],
    'Frequency (Hz)': [29.6, 641.9, 173.7, 11.2, 30.2]
}

# Convert to DataFrame for easier manipulation
pulsars_df = pd.DataFrame(pulsar_data)
pulsar_freqs = np.array(pulsars_df['Frequency (Hz)'])

# Sort frequencies for better visualization and comparison
pulsar_freqs = np.sort(pulsar_freqs)

# --- Step 3: Analyze with different phase functions and base frequencies ---
# Use minimum pulsar frequency as base (alignment suggestion)
base_freq = min(pulsar_freqs)

# Test different phase functions
phase_functions = ['sin', 'cos', 'tanh', 'none']
results = {}

for phase_func in phase_functions:
    # Generate COM frequencies with this phase function
    com_freqs = com_frequencies(base_freq, n_steps=len(pulsar_freqs), phase_func=phase_func)
    
    # Compare distributions (linear)
    ks_stat, p_value = ks_2samp(pulsar_freqs, com_freqs)
    
    # Compare distributions (logarithmic - often better for astronomical data)
    log_ks_stat, log_p_value = ks_2samp(np.log10(pulsar_freqs), np.log10(com_freqs))
    
    # Store results
    results[phase_func] = {
        'com_freqs': com_freqs,
        'linear_ks': ks_stat,
        'linear_p': p_value,
        'log_ks': log_ks_stat,
        'log_p': log_p_value
    }

# --- Step 4: Find best phase function based on p-value ---
best_phase = max(phase_functions, key=lambda x: results[x]['log_p'])
print(f"Best phase function: {best_phase} (log p-value: {results[best_phase]['log_p']:.3f})")

# --- Step 5: Calculate frequency ratios for harmonic analysis ---
def calculate_ratios(frequencies):
    """Calculate ratios between consecutive frequencies to identify harmonic patterns"""
    return [frequencies[i+1]/frequencies[i] for i in range(len(frequencies)-1)]

pulsar_ratios = calculate_ratios(pulsar_freqs)
com_ratios = calculate_ratios(results[best_phase]['com_freqs'])

# --- Step 6: Visualize results ---
# Create figure with 2x2 subplots
fig, axs = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Pulsar Frequency Analysis with COM Framework', fontsize=16)

# Plot 1: Linear frequency comparison
axs[0, 0].scatter(range(len(pulsar_freqs)), pulsar_freqs, color='red', s=100, label='Observed Pulsars')
for phase_func in phase_functions:
    axs[0, 0].plot(results[phase_func]['com_freqs'], 'o-', label=f'COM ({phase_func})')
axs[0, 0].set_xlabel("Step (n)")
axs[0, 0].set_ylabel("Frequency (Hz)")
axs[0, 0].set_title("Linear Frequency Comparison")
axs[0, 0].legend()
axs[0, 0].grid(True, alpha=0.3)

# Plot 2: Logarithmic frequency comparison
axs[0, 1].scatter(range(len(pulsar_freqs)), pulsar_freqs, color='red', s=100, label='Observed Pulsars')
for phase_func in phase_functions:
    axs[0, 1].plot(results[phase_func]['com_freqs'], 'o-', label=f'COM ({phase_func})')
axs[0, 1].set_xlabel("Step (n)")
axs[0, 1].set_ylabel("Frequency (Hz)")
axs[0, 1].set_title("Logarithmic Frequency Comparison")
axs[0, 1].set_yscale('log')
axs[0, 1].legend()
axs[0, 1].grid(True, alpha=0.3)

# Plot 3: Frequency ratios comparison (harmonic analysis)
axs[1, 0].bar(range(len(pulsar_ratios)), pulsar_ratios, width=0.4, alpha=0.7, label='Pulsar Ratios')
axs[1, 0].bar([x + 0.4 for x in range(len(com_ratios))], com_ratios, width=0.4, alpha=0.7, label='COM Ratios')
axs[1, 0].axhline(y=1.23498, color='green', linestyle='--', label='LZ Constant')
axs[1, 0].set_xlabel("Step Pair")
axs[1, 0].set_ylabel("Frequency Ratio")
axs[1, 0].set_title("Harmonic Analysis: Frequency Ratios")
axs[1, 0].legend()
axs[1, 0].grid(True, alpha=0.3)

# Plot 4: Results table
axs[1, 1].axis('tight')
axs[1, 1].axis('off')
table_data = [
    ['Phase Function', 'Linear p-value', 'Log p-value'],
]
for phase_func in phase_functions:
    table_data.append([
        phase_func, 
        f"{results[phase_func]['linear_p']:.3f}", 
        f"{results[phase_func]['log_p']:.3f}"
    ])
table = axs[1, 1].table(cellText=table_data, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 1.5)
axs[1, 1].set_title("Statistical Comparison Results")

# Add COM parameters text
param_text = (
    f"COM Framework Parameters:\n"
    f"LZ = 1.23498\n"
    f"HQS = 0.235\n"
    f"Base Frequency = {base_freq:.2f} Hz\n"
    f"Best Phase Function: {best_phase}\n"
)
fig.text(0.02, 0.02, param_text, fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.savefig('pulsar_com_analysis.png', dpi=300)
plt.show()

# --- Step 7: Extended analysis with more steps ---
# Generate extended COM frequencies with best phase function
extended_steps = 50
extended_com_freqs = com_frequencies(base_freq, n_steps=extended_steps, phase_func=best_phase)

# Create figure for extended prediction
plt.figure(figsize=(12, 8))
plt.scatter(range(len(pulsar_freqs)), pulsar_freqs, color='red', s=100, label='Observed Pulsars')
plt.plot(range(extended_steps), extended_com_freqs, 'o-', label=f'COM Extended Prediction')
plt.xlabel("Step (n)")
plt.ylabel("Frequency (Hz)")
plt.title("Extended COM Prediction for Pulsar Frequencies")
plt.yscale('log')
plt.grid(True, alpha=0.3)
plt.legend()

# Add pulsar names to the plot
for i, name in enumerate(pulsars_df['Name']):
    plt.annotate(name, (i, pulsar_freqs[i]), textcoords="offset points", 
                 xytext=(0,10), ha='center')

# Add potential prediction markers
for i in range(len(pulsar_freqs), extended_steps):
    if i % 5 == 0:  # Mark every 5th prediction
        plt.axhline(y=extended_com_freqs[i], color='green', linestyle='--', alpha=0.3)
        plt.text(extended_steps-1, extended_com_freqs[i], f"Predicted: {extended_com_freqs[i]:.1f} Hz", 
                 va='center', ha='right', bbox=dict(facecolor='white', alpha=0.7))

plt.tight_layout()
plt.savefig('pulsar_com_extended_prediction.png', dpi=300)
plt.show()

# Print summary of findings
print("\n" + "="*80)
print("COM Framework Analysis of Pulsar Frequencies")
print("="*80)
print(f"Number of pulsars analyzed: {len(pulsar_freqs)}")
print(f"Frequency range: {min(pulsar_freqs):.2f} Hz to {max(pulsar_freqs):.2f} Hz")
print(f"Best phase function: {best_phase}")
print(f"Statistical significance (log scale): p-value = {results[best_phase]['log_p']:.3f}")
print("\nObserved vs. COM-predicted frequencies:")
print("-"*60)
print(f"{'Pulsar':<15} {'Observed (Hz)':<15} {'Predicted (Hz)':<15} {'Error (%)':<10}")
print("-"*60)
for i, name in enumerate(pulsars_df['Name']):
    error_pct = (results[best_phase]['com_freqs'][i] - pulsar_freqs[i]) / pulsar_freqs[i] * 100
    print(f"{name:<15} {pulsar_freqs[i]:<15.2f} {results[best_phase]['com_freqs'][i]:<15.2f} {error_pct:<10.2f}")
print("="*80)

# Save results to CSV
output_df = pd.DataFrame({
    'Pulsar': pulsars_df['Name'],
    'Observed_Hz': pulsar_freqs,
    'COM_Predicted_Hz': results[best_phase]['com_freqs'],
    'Error_Percent': [(results[best_phase]['com_freqs'][i] - pulsar_freqs[i]) / pulsar_freqs[i] * 100 
                      for i in range(len(pulsar_freqs))]
})
output_df.to_csv('pulsar_com_analysis_results.csv', index=False)
print("Results saved to 'pulsar_com_analysis_results.csv'")
