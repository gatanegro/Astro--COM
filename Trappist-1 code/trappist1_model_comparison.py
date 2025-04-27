"""
Comparison of COM Model vs. Kepler's Law/Titius-Bode for TRAPPIST-1 System

This script computes the R² score to compare the COM model with traditional
planetary spacing laws (Kepler's Law and Titius-Bode Law) for the TRAPPIST-1 system.

Author: Martin Doina
Date: April 24, 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
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

def titius_bode_law(n):
    """
    Predicts orbital distance using the Titius-Bode Law.
    
    Parameters:
    - n: Planet index (0-based)
    
    Returns:
    - Predicted orbital distance in AU
    """
    # Traditional Titius-Bode Law: a = 0.4 + 0.3 * 2^n
    # We need to scale it for TRAPPIST-1 system
    scale_factor = a0_TRAPPIST / 0.4  # Scale to match TRAPPIST-1b
    return scale_factor * (0.4 + 0.3 * (2 ** n))

def kepler_law(n, period_ratio=1.5):
    """
    Predicts orbital distance using Kepler's Third Law with a fixed period ratio.
    
    Parameters:
    - n: Planet index (0-based)
    - period_ratio: Ratio between consecutive orbital periods
    
    Returns:
    - Predicted orbital distance in AU
    """
    # Kepler's Third Law: a^3 proportional to P^2
    # If we assume a fixed period ratio between consecutive planets
    # a_n = a_0 * (period_ratio)^(2/3 * n)
    return a0_TRAPPIST * (period_ratio ** ((2/3) * n))

def compute_r2_scores():
    """
    Compute R² scores for different models compared to observed TRAPPIST-1 data.
    
    Returns:
    - Dictionary with R² scores for each model
    """
    # TRAPPIST-1 observed distances (AU) - NASA data
    planet_names = ['TRAPPIST-1b', 'TRAPPIST-1c', 'TRAPPIST-1d', 'TRAPPIST-1e', 
                   'TRAPPIST-1f', 'TRAPPIST-1g', 'TRAPPIST-1h']
    observed = np.array([0.0115, 0.0158, 0.0223, 0.0293, 0.0385, 0.0469, 0.0619])
    layers = np.arange(len(observed))
    
    # Calculate predictions for each model
    com_predicted = np.array([com_hqs_lz_orbit(n) for n in layers])
    titius_bode_predicted = np.array([titius_bode_law(n) for n in layers])
    
    # Find optimal period ratio for Kepler's Law
    best_r2 = -np.inf
    best_period_ratio = 1.5
    for period_ratio in np.linspace(1.1, 2.0, 100):
        kepler_predicted = np.array([kepler_law(n, period_ratio) for n in layers])
        r2 = r2_score(observed, kepler_predicted)
        if r2 > best_r2:
            best_r2 = r2
            best_period_ratio = period_ratio
    
    kepler_predicted = np.array([kepler_law(n, best_period_ratio) for n in layers])
    
    # Compute R² scores
    com_r2 = r2_score(observed, com_predicted)
    titius_bode_r2 = r2_score(observed, titius_bode_predicted)
    kepler_r2 = r2_score(observed, kepler_predicted)
    
    # Prepare results
    results = {
        "planet_names": planet_names,
        "observed": observed,
        "com_predicted": com_predicted,
        "titius_bode_predicted": titius_bode_predicted,
        "kepler_predicted": kepler_predicted,
        "com_r2": com_r2,
        "titius_bode_r2": titius_bode_r2,
        "kepler_r2": kepler_r2,
        "best_period_ratio": best_period_ratio,
        "layers": layers
    }
    
    return results

def plot_model_comparison(results, output_dir="figures", filename="trappist1_model_comparison.png"):
    """
    Create a plot comparing different models for the TRAPPIST-1 system.
    
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
    com_predicted = results["com_predicted"]
    titius_bode_predicted = results["titius_bode_predicted"]
    kepler_predicted = results["kepler_predicted"]
    layers = results["layers"]
    
    # Create figure
    plt.figure(figsize=(14, 8))
    
    # Plot data points
    plt.scatter(layers, observed, s=100, marker='o', color='blue', 
                label='Observed', zorder=3)
    plt.scatter(layers, com_predicted, s=80, marker='x', color='green', 
                label=f'COM Model (R² = {results["com_r2"]:.3f})', zorder=2)
    plt.scatter(layers, titius_bode_predicted, s=80, marker='s', color='red', 
                label=f'Titius-Bode Law (R² = {results["titius_bode_r2"]:.3f})', zorder=2)
    plt.scatter(layers, kepler_predicted, s=80, marker='^', color='purple', 
                label=f'Kepler\'s Law (R² = {results["kepler_r2"]:.3f})', zorder=2)
    
    # Connect points with lines
    plt.plot(layers, observed, '--', color='blue', alpha=0.5)
    plt.plot(layers, com_predicted, '--', color='green', alpha=0.5)
    plt.plot(layers, titius_bode_predicted, '--', color='red', alpha=0.5)
    plt.plot(layers, kepler_predicted, '--', color='purple', alpha=0.5)
    
    # Add planet labels
    for i, name in enumerate(planet_names):
        plt.text(layers[i], observed[i]*1.1, name, ha='center', va='bottom', fontsize=9)
    
    # Add labels and title
    plt.xlabel('Planet Index (n)', fontsize=12)
    plt.ylabel('Semi-Major Axis (AU)', fontsize=12)
    plt.title(f'TRAPPIST-1 System: Comparison of Planetary Spacing Models', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add legend
    plt.legend(loc='upper left')
    
    # Add model parameters text
    param_text = f"COM Framework Parameters:\n" \
                 f"LZ (λ) = {LZ}\n" \
                 f"HQS (η) = {HQS}\n" \
                 f"a₀ = {a0_TRAPPIST} AU\n" \
                 f"Phase function = tanh(n/2)\n\n" \
                 f"Kepler's Law:\n" \
                 f"Period ratio = {results['best_period_ratio']:.3f}"
    plt.figtext(0.02, 0.02, param_text, fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.8))
    
    # Save figure
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()
    
    print(f"Model comparison plot saved to {os.path.join(output_dir, filename)}")

def plot_log_model_comparison(results, output_dir="figures", filename="trappist1_log_model_comparison.png"):
    """
    Create a logarithmic plot comparing different models for the TRAPPIST-1 system.
    
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
    com_predicted = results["com_predicted"]
    titius_bode_predicted = results["titius_bode_predicted"]
    kepler_predicted = results["kepler_predicted"]
    layers = results["layers"]
    
    # Create figure
    plt.figure(figsize=(14, 8))
    
    # Plot data points
    plt.semilogy(layers, observed, 'o', markersize=10, color='blue', 
                label='Observed')
    plt.semilogy(layers, com_predicted, 'x', markersize=8, color='green', 
                label=f'COM Model (R² = {results["com_r2"]:.3f})')
    plt.semilogy(layers, titius_bode_predicted, 's', markersize=8, color='red', 
                label=f'Titius-Bode Law (R² = {results["titius_bode_r2"]:.3f})')
    plt.semilogy(layers, kepler_predicted, '^', markersize=8, color='purple', 
                label=f'Kepler\'s Law (R² = {results["kepler_r2"]:.3f})')
    
    # Connect points with lines
    plt.semilogy(layers, observed, '--', color='blue', alpha=0.5)
    plt.semilogy(layers, com_predicted, '--', color='green', alpha=0.5)
    plt.semilogy(layers, titius_bode_predicted, '--', color='red', alpha=0.5)
    plt.semilogy(layers, kepler_predicted, '--', color='purple', alpha=0.5)
    
    # Add planet labels
    for i, name in enumerate(planet_names):
        plt.text(layers[i], observed[i]*1.1, name, ha='center', va='bottom', fontsize=9)
    
    # Add labels and title
    plt.xlabel('Planet Index (n)', fontsize=12)
    plt.ylabel('Semi-Major Axis (AU) - Log Scale', fontsize=12)
    plt.title(f'TRAPPIST-1 System: Logarithmic Comparison of Planetary Spacing Models', fontsize=14)
    plt.grid(True, which="both", linestyle='--', alpha=0.7)
    
    # Add legend
    plt.legend(loc='upper left')
    
    # Add model parameters text
    param_text = f"COM Framework Parameters:\n" \
                 f"LZ (λ) = {LZ}\n" \
                 f"HQS (η) = {HQS}\n" \
                 f"a₀ = {a0_TRAPPIST} AU\n" \
                 f"Phase function = tanh(n/2)\n\n" \
                 f"Kepler's Law:\n" \
                 f"Period ratio = {results['best_period_ratio']:.3f}"
    plt.figtext(0.02, 0.02, param_text, fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.8))
    
    # Save figure
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()
    
    print(f"Logarithmic model comparison plot saved to {os.path.join(output_dir, filename)}")

def plot_residuals(results, output_dir="figures", filename="trappist1_model_residuals.png"):
    """
    Create a plot showing residuals for different models.
    
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
    com_predicted = results["com_predicted"]
    titius_bode_predicted = results["titius_bode_predicted"]
    kepler_predicted = results["kepler_predicted"]
    layers = results["layers"]
    
    # Calculate residuals (% error)
    com_residuals = (com_predicted - observed) / observed * 100
    titius_bode_residuals = (titius_bode_predicted - observed) / observed * 100
    kepler_residuals = (kepler_predicted - observed) / observed * 100
    
    # Create figure
    plt.figure(figsize=(14, 8))
    
    # Plot residuals
    plt.bar(layers - 0.2, com_residuals, width=0.2, color='green', 
            label=f'COM Model (R² = {results["com_r2"]:.3f})')
    plt.bar(layers, titius_bode_residuals, width=0.2, color='red', 
            label=f'Titius-Bode Law (R² = {results["titius_bode_r2"]:.3f})')
    plt.bar(layers + 0.2, kepler_residuals, width=0.2, color='purple', 
            label=f'Kepler\'s Law (R² = {results["kepler_r2"]:.3f})')
    
    # Add zero line
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Add labels and title
    plt.xlabel('Planet', fontsize=12)
    plt.ylabel('Residual Error (%)', fontsize=12)
    plt.title(f'TRAPPIST-1 System: Residual Errors for Different Planetary Spacing Models', fontsize=14)
    plt.xticks(layers, planet_names, rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add legend
    plt.legend(loc='upper left')
    
    # Add model parameters text
    param_text = f"COM Framework Parameters:\n" \
                 f"LZ (λ) = {LZ}\n" \
                 f"HQS (η) = {HQS}\n" \
                 f"a₀ = {a0_TRAPPIST} AU\n" \
                 f"Phase function = tanh(n/2)\n\n" \
                 f"Kepler's Law:\n" \
                 f"Period ratio = {results['best_period_ratio']:.3f}"
    plt.figtext(0.02, 0.02, param_text, fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.8))
    
    # Save figure
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()
    
    print(f"Residuals plot saved to {os.path.join(output_dir, filename)}")

def main():
    """Main function to compare different models for the TRAPPIST-1 system."""
    print("=" * 80)
    print("Comparison of COM Model vs. Kepler's Law/Titius-Bode for TRAPPIST-1 System")
    print("=" * 80)
    
    # Compute R² scores
    results = compute_r2_scores()
    
    # Print R² scores
    print(f"COM Model R² Score: {results['com_r2']:.3f}")
    print(f"Titius-Bode Law R² Score: {results['titius_bode_r2']:.3f}")
    print(f"Kepler's Law R² Score: {results['kepler_r2']:.3f} (with period ratio = {results['best_period_ratio']:.3f})")
    
    # Generate visualizations
    plot_model_comparison(results)
    plot_log_model_comparison(results)
    plot_residuals(results)
    
    print("\n" + "=" * 80)
    print("Detailed results:")
    print(f"{'Planet':<12} {'Observed (AU)':<15} {'COM (AU)':<15} {'T-B (AU)':<15} {'Kepler (AU)':<15}")
    print("-" * 75)
    for i, name in enumerate(results["planet_names"]):
        print(f"{name:<12} {results['observed'][i]:<15.4f} {results['com_predicted'][i]:<15.4f} "
              f"{results['titius_bode_predicted'][i]:<15.4f} {results['kepler_predicted'][i]:<15.4f}")
    print("=" * 80)
    
    # Return results for potential further analysis
    return results

if __name__ == "__main__":
    main()
