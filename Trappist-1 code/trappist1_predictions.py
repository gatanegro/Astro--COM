"""
COM Framework Analysis of TRAPPIST-1 Exoplanetary System: Additional Planet Predictions

This module extends the COM-HQS-LZ model for the TRAPPIST-1 system
to predict potential additional planets beyond the currently known seven.

Author: Martin Doina
Date: April 24, 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import os

# COM framework constants
LZ = 1.23498  # LZ scaling constant
HQS = 0.235   # HQS modulation constant
a0_TRAPPIST = 0.0115   # TRAPPIST-1b's observed distance (AU)

def tanh_phase(n):
    """Best performing phase function: tanh(n/2)"""
    return np.tanh(n / 2)

def com_hqs_lz_orbit(n, phase_func=tanh_phase):
    """
    Predicts orbital distance for octave layer n using the COM-HQS-LZ model.
    
    Parameters:
    - n: Octave layer number
    - phase_func: Phase function to use (default: tanh(n/2))
    
    Returns:
    - Predicted orbital distance in AU
    """
    return a0_TRAPPIST * (LZ ** n) * (1 + HQS * phase_func(n))

def analyze_trappist1_with_predictions(max_n=12):
    """
    Analyze the TRAPPIST-1 system using the COM-HQS-LZ model and predict additional planets.
    
    Parameters:
    - max_n: Maximum octave layer to consider for predictions
    
    Returns:
    - Dictionary with analysis results including predictions
    """
    # TRAPPIST-1 observed distances (AU) - NASA data
    planet_names = ['TRAPPIST-1b', 'TRAPPIST-1c', 'TRAPPIST-1d', 'TRAPPIST-1e', 
                   'TRAPPIST-1f', 'TRAPPIST-1g', 'TRAPPIST-1h']
    observed = [0.0115, 0.0158, 0.0223, 0.0293, 0.0385, 0.0469, 0.0619]
    known_layers = np.arange(len(observed))
    
    # Calculate predictions for known planets
    predicted_known = [com_hqs_lz_orbit(n) for n in known_layers]
    
    # Calculate residuals (% error) for known planets
    residuals_known = (np.array(predicted_known) - np.array(observed)) / np.array(observed) * 100
    avg_abs_error_known = np.mean(np.abs(residuals_known))
    
    # Predict additional planets
    additional_layers = np.arange(len(observed), max_n)
    predicted_additional = [com_hqs_lz_orbit(n) for n in additional_layers]
    
    # Create names for predicted planets
    additional_names = []
    for i, n in enumerate(additional_layers):
        if i == 0:  # First predicted planet named MANUS
            additional_names.append('TRAPPIST-1 MANUS')
        else:
            # Continue with alphabetical naming after 'h'
            next_letter = chr(ord('h') + i)
            additional_names.append(f'TRAPPIST-1{next_letter}')
    
    # Estimate error margins for predictions based on known error pattern
    # Error tends to increase with distance, so we'll use a conservative approach
    error_margins = []
    for i, n in enumerate(additional_layers):
        # Base error on the average of the three outermost known planets
        base_error = np.mean(np.abs(residuals_known[-3:]))
        # Add a safety factor that increases with distance
        safety_factor = 1.0 + (i * 0.1)  # 10% increase per additional planet
        error_margins.append(base_error * safety_factor)
    
    # Prepare results for return
    results = {
        "known_planet_names": planet_names,
        "known_observed": observed,
        "known_predicted": predicted_known,
        "known_residuals": residuals_known,
        "known_avg_abs_error": avg_abs_error_known,
        "known_layers": known_layers,
        "additional_layers": additional_layers,
        "additional_names": additional_names,
        "additional_predicted": predicted_additional,
        "additional_error_margins": error_margins
    }
    
    return results

def plot_predictions(results, output_dir="figures", filename="trappist1_predictions.png"):
    """
    Create a plot showing known planets and predictions for additional planets.
    
    Parameters:
    - results: Dictionary with analysis results
    - output_dir: Directory to save the figure
    - filename: Name of the output file
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Extract data for plotting
    known_names = results["known_planet_names"]
    known_observed = results["known_observed"]
    known_predicted = results["known_predicted"]
    known_layers = results["known_layers"]
    additional_names = results["additional_names"]
    additional_predicted = results["additional_predicted"]
    additional_layers = results["additional_layers"]
    additional_error_margins = results["additional_error_margins"]
    
    # Create figure
    plt.figure(figsize=(14, 8))
    
    # Plot known planets
    plt.scatter(known_layers, known_observed, s=100, marker='o', color='blue', 
                label='Known Planets (Observed)', zorder=3)
    plt.scatter(known_layers, known_predicted, s=80, marker='x', color='green', 
                label='Known Planets (COM Model)', zorder=2)
    
    # Plot predicted planets
    plt.scatter(additional_layers, additional_predicted, s=100, marker='*', color='red', 
                label='Predicted Additional Planets', zorder=3)
    
    # Add error bars for predictions
    for i, (n, pred, err) in enumerate(zip(additional_layers, additional_predicted, additional_error_margins)):
        lower = pred * (1 - err/100)
        upper = pred * (1 + err/100)
        plt.fill_between([n-0.2, n+0.2], [lower, lower], [upper, upper], color='red', alpha=0.2)
        
    # Generate continuous model
    n_values = np.linspace(0, max(additional_layers) + 0.5, 1000)
    a_values = [com_hqs_lz_orbit(n) for n in n_values]
    
    # Plot continuous model
    plt.plot(n_values, a_values, '-', color='green', alpha=0.7, linewidth=2, 
             label=f'COM-HQS-LZ Model: $a_n = a_0 \\cdot \\lambda^n \\cdot (1 + \\eta \\cdot tanh(n/2))$')
    
    # Add planet labels
    for i, name in enumerate(known_names):
        plt.text(known_layers[i], known_observed[i]*1.1, name, ha='center', va='bottom', fontsize=9)
    
    for i, name in enumerate(additional_names):
        plt.text(additional_layers[i], additional_predicted[i]*1.1, name, ha='center', va='bottom', 
                 fontsize=9, color='red', fontweight='bold')
    
    # Add labels and title
    plt.xlabel('Octave Layer (n)', fontsize=12)
    plt.ylabel('Semi-Major Axis (AU)', fontsize=12)
    plt.title(f'TRAPPIST-1 System: Known Planets and COM-HQS-LZ Predictions for Additional Planets', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add legend
    plt.legend(loc='upper left')
    
    # Add model parameters text
    param_text = f"COM Framework Parameters:\n" \
                 f"LZ (λ) = {LZ}\n" \
                 f"HQS (η) = {HQS}\n" \
                 f"a₀ = {a0_TRAPPIST} AU\n" \
                 f"Phase function = tanh(n/2)\n" \
                 f"Known planets avg. error = {results['known_avg_abs_error']:.2f}%"
    plt.figtext(0.02, 0.02, param_text, fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.8))
    
    # Save figure
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()
    
    print(f"Predictions plot saved to {os.path.join(output_dir, filename)}")

def plot_log_predictions(results, output_dir="figures", filename="trappist1_log_predictions.png"):
    """
    Create a logarithmic plot showing known planets and predictions for additional planets.
    
    Parameters:
    - results: Dictionary with analysis results
    - output_dir: Directory to save the figure
    - filename: Name of the output file
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Extract data for plotting
    known_names = results["known_planet_names"]
    known_observed = results["known_observed"]
    known_predicted = results["known_predicted"]
    known_layers = results["known_layers"]
    additional_names = results["additional_names"]
    additional_predicted = results["additional_predicted"]
    additional_layers = results["additional_layers"]
    additional_error_margins = results["additional_error_margins"]
    
    # Create figure
    plt.figure(figsize=(14, 8))
    
    # Plot known planets
    plt.semilogy(known_layers, known_observed, 'o', markersize=10, color='blue', 
                label='Known Planets (Observed)')
    plt.semilogy(known_layers, known_predicted, 'x', markersize=8, color='green', 
                label='Known Planets (COM Model)')
    
    # Plot predicted planets
    plt.semilogy(additional_layers, additional_predicted, '*', markersize=12, color='red', 
                label='Predicted Additional Planets')
    
    # Add error ranges for predictions
    for i, (n, pred, err) in enumerate(zip(additional_layers, additional_predicted, additional_error_margins)):
        lower = pred * (1 - err/100)
        upper = pred * (1 + err/100)
        plt.fill_between([n-0.2, n+0.2], [lower, lower], [upper, upper], color='red', alpha=0.2)
    
    # Generate continuous model
    n_values = np.linspace(0, max(additional_layers) + 0.5, 1000)
    a_values = [com_hqs_lz_orbit(n) for n in n_values]
    
    # Plot continuous model
    plt.semilogy(n_values, a_values, '-', color='green', alpha=0.7, linewidth=2, 
             label=f'COM-HQS-LZ Model')
    
    # Add planet labels
    for i, name in enumerate(known_names):
        plt.text(known_layers[i], known_observed[i]*1.1, name, ha='center', va='bottom', fontsize=9)
    
    for i, name in enumerate(additional_names):
        plt.text(additional_layers[i], additional_predicted[i]*1.1, name, ha='center', va='bottom', 
                 fontsize=9, color='red', fontweight='bold')
    
    # Add labels and title
    plt.xlabel('Octave Layer (n)', fontsize=12)
    plt.ylabel('Semi-Major Axis (AU) - Log Scale', fontsize=12)
    plt.title(f'TRAPPIST-1 System: Logarithmic Plot with Predicted Additional Planets', fontsize=14)
    plt.grid(True, which="both", linestyle='--', alpha=0.7)
    
    # Add legend
    plt.legend(loc='upper left')
    
    # Add model parameters text
    param_text = f"COM Framework Parameters:\n" \
                 f"LZ (λ) = {LZ}\n" \
                 f"HQS (η) = {HQS}\n" \
                 f"a₀ = {a0_TRAPPIST} AU\n" \
                 f"Phase function = tanh(n/2)"
    plt.figtext(0.02, 0.02, param_text, fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.8))
    
    # Save figure
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()
    
    print(f"Logarithmic predictions plot saved to {os.path.join(output_dir, filename)}")

def main():
    """Main function to analyze TRAPPIST-1 system and predict additional planets."""
    print("=" * 80)
    print("COM-HQS-LZ Analysis of TRAPPIST-1 System: Predictions for Additional Planets")
    print("=" * 80)
    
    # Analyze TRAPPIST-1 system with predictions for additional planets
    results = analyze_trappist1_with_predictions(max_n=12)
    
    # Generate visualizations
    plot_predictions(results)
    plot_log_predictions(results)
    
    print("\n" + "=" * 80)
    print("Known planets:")
    print(f"{'Planet':<15} {'Observed (AU)':<15} {'Predicted (AU)':<15} {'Error (%)':<10}")
    print("-" * 60)
    for i, name in enumerate(results["known_planet_names"]):
        print(f"{name:<15} {results['known_observed'][i]:<15.4f} {results['known_predicted'][i]:<15.4f} {results['known_residuals'][i]:<10.2f}")
    print("-" * 60)
    print(f"Average absolute error for known planets: {results['known_avg_abs_error']:.2f}%")
    print("=" * 80)
    
    print("\nPredicted additional planets:")
    print(f"{'Planet':<20} {'Predicted (AU)':<15} {'Error Margin (%)':<20}")
    print("-" * 60)
    for i, name in enumerate(results["additional_names"]):
        print(f"{name:<20} {results['additional_predicted'][i]:<15.4f} ±{results['additional_error_margins'][i]:<18.2f}")
    print("=" * 80)
    
    # Return results for potential further analysis
    return results

if __name__ == "__main__":
    main()
