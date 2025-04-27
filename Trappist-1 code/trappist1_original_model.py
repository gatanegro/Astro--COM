"""
COM Framework Analysis of TRAPPIST-1 Exoplanetary System with Original Constants

This module implements the original COM-HQS-LZ model for the TRAPPIST-1 system
using the unmodified constants and TRAPPIST-1b as the baseline.

Author: Martin Doina
Date: April 24, 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import os

# Original COM framework constants
LZ = 1.23498  # LZ scaling constant
HQS = 0.235   # HQS modulation constant
a0_TRAPPIST = 0.0115   # TRAPPIST-1b's observed distance (AU)

def com_hqs_lz_orbit(n, phase_func=lambda n: np.sin(n * np.pi / 3)):
    """
    Predicts orbital distance for octave layer n with phase modulation.
    
    Parameters:
    - n: Octave layer number
    - phase_func: Phase function to use (default: sin(n*π/3))
    
    Returns:
    - Predicted orbital distance in AU
    """
    return a0_TRAPPIST * (LZ ** n) * (1 + HQS * phase_func(n))

def analyze_trappist1():
    """
    Analyze the TRAPPIST-1 system using the original COM-HQS-LZ model.
    
    Returns:
    - Dictionary with analysis results
    """
    # TRAPPIST-1 observed distances (AU) - NASA data
    planet_names = ['TRAPPIST-1b', 'TRAPPIST-1c', 'TRAPPIST-1d', 'TRAPPIST-1e', 
                   'TRAPPIST-1f', 'TRAPPIST-1g', 'TRAPPIST-1h']
    observed = [0.0115, 0.0158, 0.0223, 0.0293, 0.0385, 0.0469, 0.0619]
    layers = np.arange(len(observed))
    
    # Compare predictions
    predicted = [com_hqs_lz_orbit(n) for n in layers]
    residuals = (np.array(predicted) - np.array(observed)) / np.array(observed) * 100  # % error
    
    # Print results
    print("TRAPPIST-1 Validation with Original COM Constants:")
    print("{:<12} {:<15} {:<15} {:<15}".format("Planet", "Observed (AU)", "Predicted (AU)", "Error (%)"))
    print("-" * 60)
    
    for i, (name, obs, pred, err) in enumerate(zip(planet_names, observed, predicted, residuals)):
        print("{:<12} {:<15.4f} {:<15.4f} {:<15.2f}".format(name, obs, pred, err))
    
    # Calculate average absolute error
    avg_abs_error = np.mean(np.abs(residuals))
    print("-" * 60)
    print(f"Average Absolute Error: {avg_abs_error:.2f}%")
    
    # Prepare results for return
    results = {
        "planet_names": planet_names,
        "observed": observed,
        "predicted": predicted,
        "residuals": residuals,
        "avg_abs_error": avg_abs_error,
        "layers": layers
    }
    
    return results

def plot_comparison(results, output_dir="figures", filename="trappist1_original_comparison.png"):
    """
    Create a bar chart comparing observed and predicted values.
    
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
    predicted = results["predicted"]
    residuals = results["residuals"]
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot bars
    x = np.arange(len(planet_names))
    width = 0.35
    
    plt.bar(x - width/2, observed, width, label='Observed', color='blue', alpha=0.7)
    plt.bar(x + width/2, predicted, width, label='COM-HQS-LZ Predicted', color='red', alpha=0.7)
    
    # Add error labels
    for i in range(len(planet_names)):
        plt.text(i, max(observed[i], predicted[i]) + 0.002, 
                 f"Error: {residuals[i]:.2f}%", 
                 ha='center', va='bottom', fontsize=10)
    
    # Add labels and title
    plt.xlabel('Planet', fontsize=12)
    plt.ylabel('Semi-Major Axis (AU)', fontsize=12)
    plt.title('TRAPPIST-1 System: Observed vs. Original COM-HQS-LZ Predicted Orbital Distances', fontsize=14)
    plt.xticks(x, planet_names, rotation=45)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add average error text
    plt.figtext(0.5, 0.01, f"Average Absolute Error: {results['avg_abs_error']:.2f}%", 
                ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    # Save figure
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()
    
    print(f"Comparison plot saved to {os.path.join(output_dir, filename)}")

def plot_log_scale(results, output_dir="figures", filename="trappist1_original_log.png"):
    """
    Create a logarithmic plot of observed and predicted values.
    
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
    predicted = results["predicted"]
    residuals = results["residuals"]
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot on log scale
    plt.semilogy(planet_names, observed, 'o-', label='Observed', color='blue', markersize=10)
    plt.semilogy(planet_names, predicted, 's-', label='COM-HQS-LZ Predicted', color='red', markersize=10)
    
    # Add error labels
    for i in range(len(planet_names)):
        plt.text(i, observed[i] * 1.1, 
                 f"Error: {residuals[i]:.2f}%", 
                 ha='center', va='bottom', fontsize=10)
    
    # Add labels and title
    plt.xlabel('Planet', fontsize=12)
    plt.ylabel('Semi-Major Axis (AU) - Log Scale', fontsize=12)
    plt.title('TRAPPIST-1 System (Log Scale): Observed vs. Original COM-HQS-LZ Predicted', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.xticks(rotation=45)
    
    # Format y-axis to show actual values instead of powers
    plt.gca().yaxis.set_major_formatter(ScalarFormatter())
    
    # Add average error text
    plt.figtext(0.5, 0.01, f"Average Absolute Error: {results['avg_abs_error']:.2f}%", 
                ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    # Save figure
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()
    
    print(f"Log scale plot saved to {os.path.join(output_dir, filename)}")

def plot_continuous_model(results, output_dir="figures", filename="trappist1_original_continuous.png"):
    """
    Create a plot of the continuous model with planet positions marked.
    
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
    predicted = results["predicted"]
    residuals = results["residuals"]
    layers = results["layers"]
    
    # Generate continuous model
    n_values = np.linspace(0, len(observed) - 0.5, 1000)
    a_values = [com_hqs_lz_orbit(n) for n in n_values]
    
    # Create figure
    plt.figure(figsize=(14, 8))
    
    # Plot continuous model
    plt.plot(n_values, a_values, '-', color='blue', alpha=0.7, linewidth=2, 
             label='COM-HQS-LZ Model: $a_n = a_0 \\cdot \\lambda^n \\cdot (1 + \\eta \\cdot \\sin(n\\pi/3))$')
    
    # Plot planet positions
    for i, (name, obs, pred, n) in enumerate(zip(planet_names, observed, predicted, layers)):
        plt.plot(n, obs, 'o', markersize=10, label=f"{name} (Observed)")
        plt.plot(n, pred, 's', markersize=8, label=f"{name} (Predicted)")
        
        # Add error label
        plt.text(n, obs * 1.1, 
                 f"Error: {residuals[i]:.2f}%", 
                 ha='center', va='bottom', fontsize=10)
    
    # Add labels and title
    plt.xlabel('Octave Layer (n)', fontsize=12)
    plt.ylabel('Semi-Major Axis (AU)', fontsize=12)
    plt.title('Continuous Original COM-HQS-LZ Model for TRAPPIST-1 System', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add legend with smaller font size and multiple columns
    plt.legend(fontsize=10, ncol=3, loc='upper left')
    
    # Add model parameters text
    param_text = f"Model Parameters:\n" \
                 f"$\\lambda$ (LZ) = {LZ}\n" \
                 f"$\\eta$ (HQS) = {HQS}\n" \
                 f"$a_0$ = {a0_TRAPPIST} AU\n" \
                 f"Phase function = $\\sin(n\\pi/3)$"
    plt.figtext(0.02, 0.02, param_text, fontsize=12, 
                bbox=dict(facecolor='white', alpha=0.8))
    
    # Add average error text
    plt.figtext(0.98, 0.02, f"Average Absolute Error: {results['avg_abs_error']:.2f}%", 
                ha='right', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    # Save figure
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()
    
    print(f"Continuous model plot saved to {os.path.join(output_dir, filename)}")

def plot_solar_trappist_comparison(output_dir="figures", filename="solar_trappist_comparison.png"):
    """
    Create a plot comparing Solar System and TRAPPIST-1 system with the COM model.
    
    Parameters:
    - output_dir: Directory to save the figure
    - filename: Name of the output file
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Solar System data (AU)
    solar_planets = ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']
    solar_observed = [0.39, 0.72, 1.00, 1.52, 5.20, 9.58, 19.18, 30.07]
    
    # TRAPPIST-1 data (AU)
    trappist_planets = ['TRAPPIST-1b', 'TRAPPIST-1c', 'TRAPPIST-1d', 'TRAPPIST-1e', 
                       'TRAPPIST-1f', 'TRAPPIST-1g', 'TRAPPIST-1h']
    trappist_observed = [0.0115, 0.0158, 0.0223, 0.0293, 0.0385, 0.0469, 0.0619]
    
    # Create figure
    plt.figure(figsize=(14, 10))
    
    # Create two subplots
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212)
    
    # Plot Solar System
    x_solar = np.arange(len(solar_planets))
    ax1.semilogy(x_solar, solar_observed, 'o-', color='orange', markersize=10, label='Solar System')
    ax1.set_xticks(x_solar)
    ax1.set_xticklabels(solar_planets, rotation=45)
    ax1.set_ylabel('Semi-Major Axis (AU) - Log Scale', fontsize=12)
    ax1.set_title('Solar System Planetary Distances', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Plot TRAPPIST-1
    x_trappist = np.arange(len(trappist_planets))
    ax2.semilogy(x_trappist, trappist_observed, 'o-', color='blue', markersize=10, label='TRAPPIST-1')
    ax2.set_xticks(x_trappist)
    ax2.set_xticklabels(trappist_planets, rotation=45)
    ax2.set_ylabel('Semi-Major Axis (AU) - Log Scale', fontsize=12)
    ax2.set_title('TRAPPIST-1 System Planetary Distances', fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Add text comparing the systems
    comparison_text = "System Comparison:\n" \
                      f"- Solar System spans {solar_observed[-1]/solar_observed[0]:.1f}x distance from first to last planet\n" \
                      f"- TRAPPIST-1 spans {trappist_observed[-1]/trappist_observed[0]:.1f}x distance from first to last planet\n" \
                      f"- TRAPPIST-1b to TRAPPIST-1h: {trappist_observed[-1]} AU\n" \
                      f"- Mercury to Neptune: {solar_observed[-1]} AU\n" \
                      f"- Scale difference: {solar_observed[-1]/trappist_observed[-1]:.0f}x"
    plt.figtext(0.5, 0.01, comparison_text, ha='center', fontsize=12, 
                bbox=dict(facecolor='white', alpha=0.8))
    
    # Save figure
    plt.tight_layout(rect=[0, 0.08, 1, 0.98])
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()
    
    print(f"Solar-TRAPPIST comparison plot saved to {os.path.join(output_dir, filename)}")

def main():
    """Main function to analyze TRAPPIST-1 system with original COM constants."""
    print("=" * 60)
    print("Original COM-HQS-LZ Analysis of TRAPPIST-1 System")
    print("=" * 60)
    
    # Analyze TRAPPIST-1 system
    results = analyze_trappist1()
    
    # Generate visualizations
    plot_comparison(results)
    plot_log_scale(results)
    plot_continuous_model(results)
    plot_solar_trappist_comparison()
    
    print("\n" + "=" * 60)
    print("Model parameters:")
    print(f"LZ (λ) = {LZ}")
    print(f"HQS (η) = {HQS}")
    print(f"a0 = {a0_TRAPPIST} AU")
    print(f"Phase function = sin(nπ/3)")
    print("=" * 60)
    
    # Return results for potential further analysis
    return results

if __name__ == "__main__":
    main()
