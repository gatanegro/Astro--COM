"""
COM Framework Analysis of TRAPPIST-1 Exoplanetary System with Error-Based Constant Scaling

This module implements the COM-HQS-LZ model for the TRAPPIST-1 system
with error-based scaling of constants to compensate for observer perspective bias.

Author: Martin Doina
Date: April 24, 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import os

# Original COM framework constants
LZ_ORIGINAL = 1.23498  # LZ scaling constant
HQS_ORIGINAL = 0.235   # HQS modulation constant
a0_TRAPPIST = 0.0115   # TRAPPIST-1b's observed distance (AU)

# Best phase function from previous analysis
def tanh_phase(n):
    """Best performing phase function: tanh(n/2)"""
    return np.tanh(n / 2)

# Error-based scaling approach
def scale_constants_by_error(error_percentage):
    """
    Scale LZ and HQS constants by the error percentage.
    
    Parameters:
    - error_percentage: Error percentage to scale by
    
    Returns:
    - Scaled LZ and HQS constants
    """
    # Calculate scaling factor (1 + error_percentage/100)
    scaling_factor = 1 + error_percentage/100
    
    # Scale constants
    LZ_scaled = LZ_ORIGINAL * scaling_factor
    HQS_scaled = HQS_ORIGINAL * scaling_factor
    
    return LZ_scaled, HQS_scaled

def com_hqs_lz_orbit(n, LZ, HQS, phase_func):
    """
    Predicts orbital distance for octave layer n with specified constants and phase function.
    
    Parameters:
    - n: Octave layer number
    - LZ: LZ constant value
    - HQS: HQS constant value
    - phase_func: Phase function to use
    
    Returns:
    - Predicted orbital distance in AU
    """
    return a0_TRAPPIST * (LZ ** n) * (1 + HQS * phase_func(n))

def analyze_trappist1_with_scaled_constants(error_percentage):
    """
    Analyze the TRAPPIST-1 system using the COM-HQS-LZ model with error-scaled constants.
    
    Parameters:
    - error_percentage: Error percentage to scale constants by
    
    Returns:
    - Dictionary with analysis results
    """
    # Scale constants
    LZ_scaled, HQS_scaled = scale_constants_by_error(error_percentage)
    
    # TRAPPIST-1 observed distances (AU) - NASA data
    planet_names = ['TRAPPIST-1b', 'TRAPPIST-1c', 'TRAPPIST-1d', 'TRAPPIST-1e', 
                   'TRAPPIST-1f', 'TRAPPIST-1g', 'TRAPPIST-1h']
    observed = [0.0115, 0.0158, 0.0223, 0.0293, 0.0385, 0.0469, 0.0619]
    layers = np.arange(len(observed))
    
    # Compare predictions with original constants
    predicted_original = [com_hqs_lz_orbit(n, LZ_ORIGINAL, HQS_ORIGINAL, tanh_phase) for n in layers]
    residuals_original = (np.array(predicted_original) - np.array(observed)) / np.array(observed) * 100  # % error
    avg_abs_error_original = np.mean(np.abs(residuals_original))
    
    # Compare predictions with scaled constants
    predicted_scaled = [com_hqs_lz_orbit(n, LZ_scaled, HQS_scaled, tanh_phase) for n in layers]
    residuals_scaled = (np.array(predicted_scaled) - np.array(observed)) / np.array(observed) * 100  # % error
    avg_abs_error_scaled = np.mean(np.abs(residuals_scaled))
    
    # Prepare results for return
    results = {
        "planet_names": planet_names,
        "observed": observed,
        "predicted_original": predicted_original,
        "residuals_original": residuals_original,
        "avg_abs_error_original": avg_abs_error_original,
        "predicted_scaled": predicted_scaled,
        "residuals_scaled": residuals_scaled,
        "avg_abs_error_scaled": avg_abs_error_scaled,
        "layers": layers,
        "LZ_original": LZ_ORIGINAL,
        "HQS_original": HQS_ORIGINAL,
        "LZ_scaled": LZ_scaled,
        "HQS_scaled": HQS_scaled,
        "scaling_factor": 1 + error_percentage/100,
        "error_percentage": error_percentage
    }
    
    return results

def plot_scaled_constants_comparison(results, output_dir="figures", filename="trappist1_scaled_constants_comparison.png"):
    """
    Create a bar chart comparing original and scaled constant models.
    
    Parameters:
    - results: Dictionary with analysis results
    - output_dir: Directory to save the figure
    - filename: Name of the output file
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Extract data for plotting
    planet_names = results["planet_names"]
    observed = results["observed"]
    predicted_original = results["predicted_original"]
    predicted_scaled = results["predicted_scaled"]
    residuals_original = results["residuals_original"]
    residuals_scaled = results["residuals_scaled"]
    
    # Create figure
    plt.figure(figsize=(14, 8))
    
    # Plot bars
    x = np.arange(len(planet_names))
    width = 0.25
    
    plt.bar(x - width, observed, width, label='Observed', color='blue', alpha=0.7)
    plt.bar(x, predicted_original, width, label=f'Original Constants (Error: {results["avg_abs_error_original"]:.2f}%)', color='green', alpha=0.7)
    plt.bar(x + width, predicted_scaled, width, label=f'Scaled Constants (Error: {results["avg_abs_error_scaled"]:.2f}%)', color='red', alpha=0.7)
    
    # Add error labels
    for i in range(len(planet_names)):
        plt.text(i - width, observed[i] + 0.002, 
                 f"Obs", 
                 ha='center', va='bottom', fontsize=8, rotation=90)
        plt.text(i, predicted_original[i] + 0.002, 
                 f"{residuals_original[i]:.1f}%", 
                 ha='center', va='bottom', fontsize=8, rotation=90)
        plt.text(i + width, predicted_scaled[i] + 0.002, 
                 f"{residuals_scaled[i]:.1f}%", 
                 ha='center', va='bottom', fontsize=8, rotation=90)
    
    # Add labels and title
    plt.xlabel('Planet', fontsize=12)
    plt.ylabel('Semi-Major Axis (AU)', fontsize=12)
    plt.title(f'TRAPPIST-1 System: Original vs. Error-Scaled Constants', fontsize=14)
    plt.xticks(x, planet_names, rotation=45)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add model parameters text
    param_text = f"Original Constants:\n" \
                 f"LZ = {results['LZ_original']}\n" \
                 f"HQS = {results['HQS_original']}\n\n" \
                 f"Scaled Constants ({results['error_percentage']}%):\n" \
                 f"LZ = {results['LZ_scaled']:.5f}\n" \
                 f"HQS = {results['HQS_scaled']:.5f}\n" \
                 f"Scaling Factor = {results['scaling_factor']:.5f}"
    plt.figtext(0.02, 0.02, param_text, fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.8))
    
    # Add average error text
    error_text = f"Average Absolute Error:\n" \
                 f"Original: {results['avg_abs_error_original']:.2f}%\n" \
                 f"Scaled: {results['avg_abs_error_scaled']:.2f}%\n" \
                 f"Improvement: {results['avg_abs_error_original'] - results['avg_abs_error_scaled']:.2f}%"
    plt.figtext(0.98, 0.02, error_text, fontsize=10, ha='right',
                bbox=dict(facecolor='white', alpha=0.8))
    
    # Save figure
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()
    
    print(f"Scaled constants comparison plot saved to {os.path.join(output_dir, filename)}")

def plot_continuous_scaled_model(results, output_dir="figures", filename="trappist1_continuous_scaled.png"):
    """
    Create a plot of the continuous model with original and scaled constants.
    
    Parameters:
    - results: Dictionary with analysis results
    - output_dir: Directory to save the figure
    - filename: Name of the output file
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Extract data for plotting
    planet_names = results["planet_names"]
    observed = results["observed"]
    predicted_original = results["predicted_original"]
    predicted_scaled = results["predicted_scaled"]
    residuals_original = results["residuals_original"]
    residuals_scaled = results["residuals_scaled"]
    layers = results["layers"]
    
    # Generate continuous model
    n_values = np.linspace(0, len(observed) - 0.5, 1000)
    a_values_original = [com_hqs_lz_orbit(n, results["LZ_original"], results["HQS_original"], tanh_phase) for n in n_values]
    a_values_scaled = [com_hqs_lz_orbit(n, results["LZ_scaled"], results["HQS_scaled"], tanh_phase) for n in n_values]
    
    # Create figure
    plt.figure(figsize=(14, 8))
    
    # Plot continuous models
    plt.plot(n_values, a_values_original, '-', color='green', alpha=0.7, linewidth=2, 
             label=f'Original Constants: $a_n = a_0 \\cdot \\lambda^n \\cdot (1 + \\eta \\cdot tanh(n/2))$')
    plt.plot(n_values, a_values_scaled, '-', color='red', alpha=0.7, linewidth=2, 
             label=f'Scaled Constants: $a_n = a_0 \\cdot \\lambda_{{\mathrm{{scaled}}}}^n \\cdot (1 + \\eta_{{\mathrm{{scaled}}}} \\cdot tanh(n/2))$')
    
    # Plot planet positions
    for i, (name, obs, n) in enumerate(zip(planet_names, observed, layers)):
        plt.plot(n, obs, 'o', markersize=10, label=f"{name} (Observed)" if i == 0 else "")
    
    # Add labels and title
    plt.xlabel('Octave Layer (n)', fontsize=12)
    plt.ylabel('Semi-Major Axis (AU)', fontsize=12)
    plt.title(f'Continuous COM-HQS-LZ Model for TRAPPIST-1 System: Original vs. Scaled Constants', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add legend with smaller font size
    plt.legend(fontsize=10, loc='upper left')
    
    # Add model parameters text
    param_text = f"Original Constants:\n" \
                 f"LZ = {results['LZ_original']}\n" \
                 f"HQS = {results['HQS_original']}\n\n" \
                 f"Scaled Constants ({results['error_percentage']}%):\n" \
                 f"LZ = {results['LZ_scaled']:.5f}\n" \
                 f"HQS = {results['HQS_scaled']:.5f}\n" \
                 f"Scaling Factor = {results['scaling_factor']:.5f}"
    plt.figtext(0.02, 0.02, param_text, fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.8))
    
    # Add average error text
    error_text = f"Average Absolute Error:\n" \
                 f"Original: {results['avg_abs_error_original']:.2f}%\n" \
                 f"Scaled: {results['avg_abs_error_scaled']:.2f}%\n" \
                 f"Improvement: {results['avg_abs_error_original'] - results['avg_abs_error_scaled']:.2f}%"
    plt.figtext(0.98, 0.02, error_text, fontsize=10, ha='right',
                bbox=dict(facecolor='white', alpha=0.8))
    
    # Save figure
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()
    
    print(f"Continuous scaled model plot saved to {os.path.join(output_dir, filename)}")

def plot_error_reduction(results, output_dir="figures", filename="trappist1_error_reduction.png"):
    """
    Create a plot showing the error reduction for each planet.
    
    Parameters:
    - results: Dictionary with analysis results
    - output_dir: Directory to save the figure
    - filename: Name of the output file
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Extract data for plotting
    planet_names = results["planet_names"]
    residuals_original = np.abs(results["residuals_original"])
    residuals_scaled = np.abs(results["residuals_scaled"])
    
    # Calculate error reduction
    error_reduction = residuals_original - residuals_scaled
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot bars
    x = np.arange(len(planet_names))
    width = 0.35
    
    plt.bar(x - width/2, residuals_original, width, label='Original Error', color='red', alpha=0.7)
    plt.bar(x + width/2, residuals_scaled, width, label='Scaled Error', color='green', alpha=0.7)
    
    # Add error reduction labels
    for i in range(len(planet_names)):
        plt.text(i, max(residuals_original[i], residuals_scaled[i]) + 1, 
                 f"Reduction: {error_reduction[i]:.2f}%", 
                 ha='center', va='bottom', fontsize=10)
    
    # Add labels and title
    plt.xlabel('Planet', fontsize=12)
    plt.ylabel('Absolute Error (%)', fontsize=12)
    plt.title(f'TRAPPIST-1 System: Error Reduction with Scaled Constants', fontsize=14)
    plt.xticks(x, planet_names, rotation=45)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add model parameters text
    param_text = f"Original Constants:\n" \
                 f"LZ = {results['LZ_original']}\n" \
                 f"HQS = {results['HQS_original']}\n\n" \
                 f"Scaled Constants ({results['error_percentage']}%):\n" \
                 f"LZ = {results['LZ_scaled']:.5f}\n" \
                 f"HQS = {results['HQS_scaled']:.5f}"
    plt.figtext(0.02, 0.02, param_text, fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.8))
    
    # Add average error text
    error_text = f"Average Absolute Error:\n" \
                 f"Original: {results['avg_abs_error_original']:.2f}%\n" \
                 f"Scaled: {results['avg_abs_error_scaled']:.2f}%\n" \
                 f"Improvement: {results['avg_abs_error_original'] - results['avg_abs_error_scaled']:.2f}%"
    plt.figtext(0.98, 0.02, error_text, fontsize=10, ha='right',
                bbox=dict(facecolor='white', alpha=0.8))
    
    # Save figure
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()
    
    print(f"Error reduction plot saved to {os.path.join(output_dir, filename)}")

def main():
    """Main function to analyze TRAPPIST-1 system with error-scaled constants."""
    print("=" * 60)
    print("COM-HQS-LZ Analysis of TRAPPIST-1 System with Error-Scaled Constants")
    print("=" * 60)
    
    # Use the best phase function (tanh(n/2)) error as scaling factor
    error_percentage = 9.24
    
    # Analyze with scaled constants
    results = analyze_trappist1_with_scaled_constants(error_percentage)
    
    # Generate visualizations
    plot_scaled_constants_comparison(results)
    plot_continuous_scaled_model(results)
    plot_error_reduction(results)
    
    print("\n" + "=" * 60)
    print("Model parameters:")
    print(f"Original LZ (λ) = {results['LZ_original']}")
    print(f"Original HQS (η) = {results['HQS_original']}")
    print(f"Scaled LZ (λ) = {results['LZ_scaled']:.5f}")
    print(f"Scaled HQS (η) = {results['HQS_scaled']:.5f}")
    print(f"Scaling Factor = {results['scaling_factor']:.5f}")
    print(f"a0 = {a0_TRAPPIST} AU")
    print(f"Phase function = tanh(n/2)")
    print(f"Original average absolute error = {results['avg_abs_error_original']:.2f}%")
    print(f"Scaled average absolute error = {results['avg_abs_error_scaled']:.2f}%")
    print(f"Error reduction = {results['avg_abs_error_original'] - results['avg_abs_error_scaled']:.2f}%")
    print("=" * 60)
    
    # Return results for potential further analysis
    return results

if __name__ == "__main__":
    main()
