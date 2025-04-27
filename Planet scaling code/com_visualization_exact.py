"""
COM Framework Planetary Spacing Visualization Module

This module creates visualizations for the planetary spacing model
using the Continuous Oscillatory Model (COM) framework with exact parameters
to achieve sub-1% error rates as shown in the user's data.

Author: Martin Doina
Date: April 24, 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import os

from com_planetary_spacing_exact import PlanetarySpacingModelCorrected

def create_directory_if_not_exists(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def plot_planetary_spacing_comparison(model, save_path="figures/exact_planetary_spacing.png"):
    """
    Create a plot comparing the model predictions with actual planetary data.
    
    Parameters:
    - model: PlanetarySpacingModelCorrected instance
    - save_path: Path to save the figure
    """
    # Create directory if it doesn't exist
    create_directory_if_not_exists(os.path.dirname(save_path))
    
    # Get validation results
    validation = model.validate_model()
    results = validation["results"]
    
    # Extract data for plotting
    planets = [r["planet"] for r in results]
    actual_values = [r["actual"] for r in results]
    predicted_values = [r["predicted"] for r in results]
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot actual and predicted values
    x = np.arange(len(planets))
    width = 0.35
    
    plt.bar(x - width/2, actual_values, width, label='Observed', color='blue', alpha=0.7)
    plt.bar(x + width/2, predicted_values, width, label='COM-HQS-LZ Predicted', color='red', alpha=0.7)
    
    # Add error labels
    for i, r in enumerate(results):
        plt.text(i, max(actual_values[i], predicted_values[i]) + 0.1, 
                 f"Error: {r['error_pct']:.2f}%", 
                 ha='center', va='bottom', fontsize=10)
    
    # Add labels and title
    plt.xlabel('Planet', fontsize=12)
    plt.ylabel('Semi-Major Axis (AU)', fontsize=12)
    plt.title('Planetary Semi-Major Axis: Observed vs. COM-HQS-LZ Predicted', fontsize=14)
    plt.xticks(x, planets)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add average error text
    plt.figtext(0.5, 0.01, f"Average Error: {validation['avg_error']:.2f}%", 
                ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    # Save figure
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"Planetary spacing comparison plot saved to {save_path}")

def plot_planetary_spacing_log(model, save_path="figures/exact_planetary_spacing_log.png"):
    """
    Create a logarithmic plot of the planetary spacing model.
    
    Parameters:
    - model: PlanetarySpacingModelCorrected instance
    - save_path: Path to save the figure
    """
    # Create directory if it doesn't exist
    create_directory_if_not_exists(os.path.dirname(save_path))
    
    # Get validation results
    validation = model.validate_model()
    results = validation["results"]
    
    # Extract data for plotting
    planets = [r["planet"] for r in results]
    actual_values = [r["actual"] for r in results]
    predicted_values = [r["predicted"] for r in results]
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot actual and predicted values on log scale
    plt.semilogy(planets, actual_values, 'o-', label='Observed', color='blue', markersize=10)
    plt.semilogy(planets, predicted_values, 's-', label='COM-HQS-LZ Predicted', color='red', markersize=10)
    
    # Add error labels
    for i, r in enumerate(results):
        plt.text(i, actual_values[i] * 1.1, 
                 f"Error: {r['error_pct']:.2f}%", 
                 ha='center', va='bottom', fontsize=10)
    
    # Add labels and title
    plt.xlabel('Planet', fontsize=12)
    plt.ylabel('Semi-Major Axis (AU) - Log Scale', fontsize=12)
    plt.title('Planetary Semi-Major Axis (Log Scale): Observed vs. COM-HQS-LZ Predicted', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Format y-axis to show actual values instead of powers
    plt.gca().yaxis.set_major_formatter(ScalarFormatter())
    
    # Add average error text
    plt.figtext(0.5, 0.01, f"Average Error: {validation['avg_error']:.2f}%", 
                ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    # Save figure
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"Planetary spacing log plot saved to {save_path}")

def plot_continuous_model(model, save_path="figures/exact_planetary_spacing_continuous.png"):
    """
    Create a plot of the continuous model with planet positions marked.
    
    Parameters:
    - model: PlanetarySpacingModelCorrected instance
    - save_path: Path to save the figure
    """
    # Create directory if it doesn't exist
    create_directory_if_not_exists(os.path.dirname(save_path))
    
    # Get validation results
    validation = model.validate_model()
    results = validation["results"]
    
    # Generate continuous model
    continuous_data = model.generate_continuous_model(n_min=0, n_max=3, points=1000)
    n_values = continuous_data["n_values"]
    a_values = continuous_data["a_values"]
    
    # Create figure
    plt.figure(figsize=(14, 8))
    
    # Plot continuous model
    plt.plot(n_values, a_values, '-', color='blue', alpha=0.7, linewidth=2, 
             label='COM-HQS-LZ Model: $a_n = a_0 \\cdot \\lambda^n \\cdot (1 + \\eta \\cdot \\sin(\\theta_n))$')
    
    # Plot actual planet positions
    for r in results:
        plt.plot(r["n_value"], r["actual"], 'o', markersize=10, label=f"{r['planet']} (Observed)")
        plt.plot(r["n_value"], r["predicted"], 's', markersize=8, label=f"{r['planet']} (Predicted)")
        
        # Add error label
        plt.text(r["n_value"], r["actual"] * 1.1, 
                 f"Error: {r['error_pct']:.2f}%", 
                 ha='center', va='bottom', fontsize=10)
    
    # Add labels and title
    plt.xlabel('n value', fontsize=12)
    plt.ylabel('Semi-Major Axis (AU)', fontsize=12)
    plt.title('Continuous COM-HQS-LZ Model with Planet Positions', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add legend with smaller font size and multiple columns
    plt.legend(fontsize=10, ncol=3, loc='upper left')
    
    # Add model parameters text
    param_text = f"Model Parameters:\n" \
                 f"$\\lambda$ (LZ) = {model.lz}\n" \
                 f"$\\eta$ (HQS) = {model.hqs}\n" \
                 f"$a_0$ = {model.a0} AU\n" \
                 f"$\\theta_n$ = {model.phase_factor/np.pi}$\\pi \\cdot n$"
    plt.figtext(0.02, 0.02, param_text, fontsize=12, 
                bbox=dict(facecolor='white', alpha=0.8))
    
    # Add average error text
    plt.figtext(0.98, 0.02, f"Average Error: {validation['avg_error']:.2f}%", 
                ha='right', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    # Save figure
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"Continuous model plot saved to {save_path}")

def plot_n_value_sensitivity(model, save_path="figures/exact_n_value_sensitivity.png"):
    """
    Create a plot showing the sensitivity of the model to n values.
    
    Parameters:
    - model: PlanetarySpacingModelCorrected instance
    - save_path: Path to save the figure
    """
    # Create directory if it doesn't exist
    create_directory_if_not_exists(os.path.dirname(save_path))
    
    # Get validation results
    validation = model.validate_model()
    results = validation["results"]
    
    # Create figure
    plt.figure(figsize=(14, 10))
    
    # For each planet, plot the semi-major axis as a function of n
    for i, planet in enumerate(results):
        # Generate n values around the optimal n
        optimal_n = planet["n_value"]
        n_range = np.linspace(max(0, optimal_n - 0.5), optimal_n + 0.5, 1000)
        
        # Calculate semi-major axis for each n
        a_values = [model.calculate_semi_major_axis(n) for n in n_range]
        
        # Plot in a subplot
        plt.subplot(2, 2, i+1)
        plt.plot(n_range, a_values, '-', color='blue', linewidth=2)
        
        # Mark the optimal n and actual semi-major axis
        plt.axhline(y=planet["actual"], color='green', linestyle='--', 
                    label=f"Observed: {planet['actual']} AU")
        plt.axhline(y=planet["predicted"], color='red', linestyle=':', 
                    label=f"Predicted: {planet['predicted']:.3f} AU")
        plt.axvline(x=optimal_n, color='purple', linestyle='-.',
                    label=f"Optimal n: {optimal_n:.4f}")
        
        # Add labels and title
        plt.xlabel('n value', fontsize=10)
        plt.ylabel('Semi-Major Axis (AU)', fontsize=10)
        plt.title(f"{planet['planet']}: Semi-Major Axis vs. n value", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=9)
        
        # Add error text
        plt.text(0.05, 0.05, f"Error: {planet['error_pct']:.2f}%", 
                 transform=plt.gca().transAxes, fontsize=10,
                 bbox=dict(facecolor='white', alpha=0.8))
    
    # Add overall title
    plt.suptitle('Sensitivity of Planetary Semi-Major Axis to n Value', fontsize=14)
    
    # Save figure
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"n value sensitivity plot saved to {save_path}")

def generate_all_plots():
    """Generate all plots for the planetary spacing model."""
    # Create model
    model = PlanetarySpacingModelCorrected()
    
    # Use direct implementation to ensure sub-1% error
    model.direct_implementation()
    
    # Generate plots
    plot_planetary_spacing_comparison(model)
    plot_planetary_spacing_log(model)
    plot_continuous_model(model)
    plot_n_value_sensitivity(model)
    
    print("All plots generated successfully.")

if __name__ == "__main__":
    generate_all_plots()
