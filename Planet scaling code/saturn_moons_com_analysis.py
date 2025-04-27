"""
Saturn Moons COM Framework Analysis

This script applies the Continuous Oscillatory Model (COM) framework to analyze
the spacing of Saturn's major moons, testing whether the same mathematical patterns
that describe planetary spacing might also apply to satellite systems.

Author: Martin Doina
Date: April 25, 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import r2_score

# --- Step 1: Define Saturn's major moons data ---
# Data source: NASA Planetary Fact Sheets and JPL Solar System Dynamics
saturn_moons_data = {
    'Name': [
        'Pan', 'Atlas', 'Prometheus', 'Pandora', 'Epimetheus', 'Janus', 
        'Mimas', 'Enceladus', 'Tethys', 'Dione', 'Rhea', 'Titan', 
        'Hyperion', 'Iapetus', 'Phoebe'
    ],
    'Semi-Major Axis (km)': [
        133584, 137670, 139380, 141720, 151422, 151472, 
        185539, 238042, 294619, 377396, 527108, 1221870, 
        1500934, 3560820, 12947918
    ],
    'Type': [
        'Ring', 'Ring', 'Ring', 'Ring', 'Co-orbital', 'Co-orbital',
        'Regular', 'Regular', 'Regular', 'Regular', 'Regular', 'Regular',
        'Irregular', 'Irregular', 'Irregular'
    ]
}

# Convert to DataFrame for easier manipulation
moons_df = pd.DataFrame(saturn_moons_data)

# --- Step 2: Define COM model functions ---
def com_model(a0, n, phase_func='sin', lz=1.23498, hqs=0.235):
    """
    Generate semi-major axis values using the COM framework.
    
    Parameters:
    - a0: Baseline semi-major axis (innermost moon)
    - n: Number of positions to generate
    - phase_func: Phase function to use ('sin', 'tanh', 'cos', or 'none')
    - lz: LZ constant (default: 1.23498)
    - hqs: HQS constant (default: 0.235)
    
    Returns:
    - List of semi-major axis values following the COM pattern
    """
    semi_major_axes = []
    
    for i in range(n):
        # Apply different phase functions
        if phase_func == 'sin':
            phase = np.sin(4 * np.pi * i)  # Similar to Solar System
        elif phase_func == 'sin2':
            phase = np.sin(2 * np.pi * i / 24)  # Similar to pulsar analysis
        elif phase_func == 'cos':
            phase = np.cos(4 * np.pi * i)
        elif phase_func == 'tanh':
            phase = np.tanh(i / 2)  # Similar to TRAPPIST-1
        elif phase_func == 'none':
            phase = 0  # No phase modulation, pure exponential growth
        else:
            raise ValueError(f"Unknown phase function: {phase_func}")
        
        # Apply COM formula: a0 * LZ^n * (1 + HQS * phase)
        a_n = a0 * (lz ** i) * (1 + hqs * phase)
        semi_major_axes.append(a_n)
    
    return semi_major_axes

def evaluate_model(observed, predicted):
    """
    Evaluate model performance using R² score and mean absolute percentage error.
    
    Parameters:
    - observed: Observed semi-major axis values
    - predicted: Predicted semi-major axis values
    
    Returns:
    - Dictionary with evaluation metrics
    """
    # Calculate R² score
    r2 = r2_score(observed, predicted)
    
    # Calculate mean absolute percentage error (MAPE)
    mape = np.mean(np.abs((observed - predicted) / observed)) * 100
    
    # Calculate individual errors
    errors = [(predicted[i] - observed[i]) / observed[i] * 100 for i in range(len(observed))]
    
    return {
        'r2': r2,
        'mape': mape,
        'errors': errors
    }

# --- Step 3: Analyze different moon subsets ---
# We'll analyze different subsets of moons to find patterns
subsets = {
    'All Moons': moons_df['Semi-Major Axis (km)'].values,
    'Regular Moons': moons_df[moons_df['Type'] == 'Regular']['Semi-Major Axis (km)'].values,
    'Inner Moons': moons_df.iloc[:6]['Semi-Major Axis (km)'].values,
    'Major Moons': moons_df.iloc[6:12]['Semi-Major Axis (km)'].values,
    'Outer Moons': moons_df.iloc[12:]['Semi-Major Axis (km)'].values
}

# --- Step 4: Test different phase functions on each subset ---
phase_functions = ['sin', 'sin2', 'cos', 'tanh', 'none']
results = {}

for subset_name, subset_data in subsets.items():
    subset_results = {}
    
    # Use the innermost moon as baseline
    a0 = subset_data[0]
    n = len(subset_data)
    
    for phase_func in phase_functions:
        # Generate COM predictions
        predicted = np.array(com_model(a0, n, phase_func))
        
        # Evaluate model
        evaluation = evaluate_model(subset_data, predicted)
        
        # Store results
        subset_results[phase_func] = {
            'predicted': predicted,
            'r2': evaluation['r2'],
            'mape': evaluation['mape'],
            'errors': evaluation['errors']
        }
    
    # Find best phase function based on R² score
    best_phase = max(phase_functions, key=lambda x: subset_results[x]['r2'])
    subset_results['best_phase'] = best_phase
    
    results[subset_name] = subset_results

# --- Step 5: Visualize results ---
# Create figure with subplots
fig, axs = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Saturn Moons: COM Framework Analysis', fontsize=16)

# Plot 1: Linear comparison of all moons
axs[0, 0].scatter(range(len(subsets['All Moons'])), subsets['All Moons'], color='red', s=100, label='Observed')
for phase_func in phase_functions:
    predicted = results['All Moons'][phase_func]['predicted']
    axs[0, 0].plot(range(len(predicted)), predicted, 'o-', label=f'COM ({phase_func})')
axs[0, 0].set_xlabel("Moon Index")
axs[0, 0].set_ylabel("Semi-Major Axis (km)")
axs[0, 0].set_title("All Saturn Moons - Linear Scale")
axs[0, 0].legend()
axs[0, 0].grid(True, alpha=0.3)

# Plot 2: Logarithmic comparison of all moons
axs[0, 1].scatter(range(len(subsets['All Moons'])), subsets['All Moons'], color='red', s=100, label='Observed')
for phase_func in phase_functions:
    predicted = results['All Moons'][phase_func]['predicted']
    axs[0, 1].plot(range(len(predicted)), predicted, 'o-', label=f'COM ({phase_func})')
axs[0, 1].set_xlabel("Moon Index")
axs[0, 1].set_ylabel("Semi-Major Axis (km)")
axs[0, 1].set_title("All Saturn Moons - Log Scale")
axs[0, 1].set_yscale('log')
axs[0, 1].legend()
axs[0, 1].grid(True, alpha=0.3)

# Plot 3: Regular moons comparison
axs[1, 0].scatter(range(len(subsets['Regular Moons'])), subsets['Regular Moons'], color='red', s=100, label='Observed')
best_phase = results['Regular Moons']['best_phase']
predicted = results['Regular Moons'][best_phase]['predicted']
axs[1, 0].plot(range(len(predicted)), predicted, 'o-', label=f'COM ({best_phase})')
axs[1, 0].set_xlabel("Moon Index")
axs[1, 0].set_ylabel("Semi-Major Axis (km)")
axs[1, 0].set_title(f"Regular Moons - Best Phase: {best_phase}")
axs[1, 0].legend()
axs[1, 0].grid(True, alpha=0.3)

# Plot 4: Results table
axs[1, 1].axis('tight')
axs[1, 1].axis('off')
table_data = [
    ['Subset', 'Best Phase', 'R² Score', 'MAPE (%)'],
]
for subset_name in subsets.keys():
    best_phase = results[subset_name]['best_phase']
    r2 = results[subset_name][best_phase]['r2']
    mape = results[subset_name][best_phase]['mape']
    table_data.append([
        subset_name, 
        best_phase,
        f"{r2:.3f}", 
        f"{mape:.2f}"
    ])
table = axs[1, 1].table(cellText=table_data, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 1.5)
axs[1, 1].set_title("COM Framework Performance by Moon Subset")

# Add COM parameters text
param_text = (
    f"COM Framework Parameters:\n"
    f"LZ = 1.23498\n"
    f"HQS = 0.235\n"
)
fig.text(0.02, 0.02, param_text, fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.savefig('saturn_moons_com_analysis.png', dpi=300)

# --- Step 6: Create detailed visualization for regular moons ---
# Regular moons are most likely to follow COM patterns due to their formation process
regular_moons = moons_df[moons_df['Type'] == 'Regular']
regular_names = regular_moons['Name'].values
regular_distances = regular_moons['Semi-Major Axis (km)'].values

best_phase = results['Regular Moons']['best_phase']
predicted = results['Regular Moons'][best_phase]['predicted']
errors = results['Regular Moons'][best_phase]['errors']

plt.figure(figsize=(12, 8))
plt.scatter(range(len(regular_distances)), regular_distances, color='red', s=100, label='Observed')
plt.plot(range(len(predicted)), predicted, 'o-', color='blue', label=f'COM ({best_phase})')

# Add moon names to the plot
for i, name in enumerate(regular_names):
    plt.annotate(name, (i, regular_distances[i]), textcoords="offset points", 
                 xytext=(0,10), ha='center')
    
    # Add error percentage
    plt.annotate(f"{errors[i]:.1f}%", (i, regular_distances[i]), textcoords="offset points", 
                 xytext=(0,-15), ha='center', color='green')

plt.xlabel("Moon Index")
plt.ylabel("Semi-Major Axis (km)")
plt.title(f"Saturn's Regular Moons: COM Framework Analysis (R² = {results['Regular Moons'][best_phase]['r2']:.3f})")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('saturn_regular_moons_com_analysis.png', dpi=300)

# --- Step 7: Print summary of findings ---
print("\n" + "="*80)
print("COM Framework Analysis of Saturn's Moons")
print("="*80)

# Print results for each subset
for subset_name in subsets.keys():
    best_phase = results[subset_name]['best_phase']
    r2 = results[subset_name][best_phase]['r2']
    mape = results[subset_name][best_phase]['mape']
    
    print(f"\n{subset_name}:")
    print(f"  Best phase function: {best_phase}")
    print(f"  R² Score: {r2:.3f}")
    print(f"  Mean Absolute Percentage Error: {mape:.2f}%")
    
    if subset_name == 'Regular Moons':
        print("\nDetailed results for Regular Moons:")
        print("-"*60)
        print(f"{'Moon':<12} {'Observed (km)':<15} {'Predicted (km)':<15} {'Error (%)':<10}")
        print("-"*60)
        
        for i, name in enumerate(regular_names):
            observed = regular_distances[i]
            predicted_val = predicted[i]
            error = errors[i]
            print(f"{name:<12} {observed:<15.0f} {predicted_val:<15.0f} {error:<10.2f}")

print("="*80)

# --- Step 8: Save results to CSV ---
# Create results dataframe for regular moons
results_df = pd.DataFrame({
    'Moon': regular_names,
    'Observed_km': regular_distances,
    'COM_Predicted_km': predicted,
    'Error_Percent': errors
})
results_df.to_csv('saturn_moons_com_analysis_results.csv', index=False)
print("Results saved to 'saturn_moons_com_analysis_results.csv'")

# --- Step 9: Predict potential undiscovered moons ---
# Extend the COM model to predict additional moons beyond the known ones
best_phase = results['Regular Moons']['best_phase']
a0 = regular_distances[0]  # Use Mimas as baseline
extended_n = 12  # Predict 6 additional moons beyond the 6 known regular moons

extended_predictions = com_model(a0, extended_n, best_phase)

plt.figure(figsize=(12, 8))
plt.scatter(range(len(regular_distances)), regular_distances, color='red', s=100, label='Observed Moons')
plt.plot(range(extended_n), extended_predictions, 'o-', label=f'COM Extended Prediction')
plt.xlabel("Moon Index")
plt.ylabel("Semi-Major Axis (km)")
plt.title("Extended COM Prediction for Saturn's Moons")
plt.yscale('log')
plt.grid(True, alpha=0.3)
plt.legend()

# Add moon names to the plot
for i, name in enumerate(regular_names):
    plt.annotate(name, (i, regular_distances[i]), textcoords="offset points", 
                 xytext=(0,10), ha='center')

# Add potential prediction markers
for i in range(len(regular_distances), extended_n):
    plt.axhline(y=extended_predictions[i], color='green', linestyle='--', alpha=0.3)
    plt.text(extended_n-1, extended_predictions[i], f"Predicted: {extended_predictions[i]:.0f} km", 
             va='center', ha='right', bbox=dict(facecolor='white', alpha=0.7))

plt.tight_layout()
plt.savefig('saturn_moons_extended_prediction.png', dpi=300)

# Print predictions for potential undiscovered moons
print("\nPredictions for potential additional moons:")
print("-"*60)
print(f"{'Moon':<12} {'Predicted Distance (km)':<25}")
print("-"*60)
for i in range(len(regular_distances), extended_n):
    print(f"Moon-{i+1:<8} {extended_predictions[i]:<25.0f}")
print("="*80)
