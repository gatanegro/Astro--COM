"""
COM Framework Analysis of TRAPPIST-1 Exoplanetary System with Different Phase Functions

This module implements the COM-HQS-LZ model for the TRAPPIST-1 system
using different phase functions to find the best fit.

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

# Define different phase functions to test
def phase_sin_pi_3(n):
    """Original phase function: sin(nπ/3)"""
    return np.sin(n * np.pi / 3)

def phase_sin_pi(n):
    """Alternative phase function: sin(nπ)"""
    return np.sin(n * np.pi)

def phase_sin_pi_2(n):
    """Alternative phase function: sin(nπ/2)"""
    return np.sin(n * np.pi / 2)

def phase_sin_pi_4(n):
    """Alternative phase function: sin(nπ/4)"""
    return np.sin(n * np.pi / 4)

def phase_sin_pi_6(n):
    """Alternative phase function: sin(nπ/6)"""
    return np.sin(n * np.pi / 6)

def phase_cos_pi_3(n):
    """Alternative phase function: cos(nπ/3)"""
    return np.cos(n * np.pi / 3)

def phase_cos_pi_2(n):
    """Alternative phase function: cos(nπ/2)"""
    return np.cos(n * np.pi / 2)

def phase_sin_2pi_3(n):
    """Alternative phase function: sin(2nπ/3)"""
    return np.sin(2 * n * np.pi / 3)

def phase_sin_squared(n):
    """Alternative phase function: sin²(nπ/3)"""
    return np.sin(n * np.pi / 3)**2

def phase_tanh(n):
    """Alternative phase function: tanh(n/2)"""
    return np.tanh(n / 2)

def phase_exp_decay(n):
    """Alternative phase function: exp(-n/3)"""
    return np.exp(-n / 3)

def phase_log(n):
    """Alternative phase function: log(n+1)/log(8)"""
    return np.log(n + 1) / np.log(8)

def phase_polynomial(n):
    """Alternative phase function: 0.1n² - 0.2n + 0.3"""
    return 0.1 * n**2 - 0.2 * n + 0.3

def phase_resonance(n):
    """Phase function based on TRAPPIST-1 resonance chain"""
    # TRAPPIST-1 has resonances of approximately 8:5, 5:3, 3:2, 3:2, 4:3, and 3:2
    resonance_factors = [1.0, 1.6, 1.67, 1.5, 1.5, 1.33, 1.5]
    if n < len(resonance_factors):
        return 0.5 * (resonance_factors[int(n)] - 1)
    else:
        return 0.5 * (1.5 - 1)  # Default to 3:2 resonance

# Dictionary of phase functions to test
phase_functions = {
    "sin(nπ/3)": phase_sin_pi_3,
    "sin(nπ)": phase_sin_pi,
    "sin(nπ/2)": phase_sin_pi_2,
    "sin(nπ/4)": phase_sin_pi_4,
    "sin(nπ/6)": phase_sin_pi_6,
    "cos(nπ/3)": phase_cos_pi_3,
    "cos(nπ/2)": phase_cos_pi_2,
    "sin(2nπ/3)": phase_sin_2pi_3,
    "sin²(nπ/3)": phase_sin_squared,
    "tanh(n/2)": phase_tanh,
    "exp(-n/3)": phase_exp_decay,
    "log(n+1)/log(8)": phase_log,
    "0.1n² - 0.2n + 0.3": phase_polynomial,
    "resonance-based": phase_resonance
}

def com_hqs_lz_orbit(n, phase_func):
    """
    Predicts orbital distance for octave layer n with specified phase function.
    
    Parameters:
    - n: Octave layer number
    - phase_func: Phase function to use
    
    Returns:
    - Predicted orbital distance in AU
    """
    return a0_TRAPPIST * (LZ ** n) * (1 + HQS * phase_func(n))

def analyze_trappist1_with_phase_function(phase_func_name, phase_func):
    """
    Analyze the TRAPPIST-1 system using the COM-HQS-LZ model with a specific phase function.
    
    Parameters:
    - phase_func_name: Name of the phase function
    - phase_func: Phase function to use
    
    Returns:
    - Dictionary with analysis results
    """
    # TRAPPIST-1 observed distances (AU) - NASA data
    planet_names = ['TRAPPIST-1b', 'TRAPPIST-1c', 'TRAPPIST-1d', 'TRAPPIST-1e', 
                   'TRAPPIST-1f', 'TRAPPIST-1g', 'TRAPPIST-1h']
    observed = [0.0115, 0.0158, 0.0223, 0.0293, 0.0385, 0.0469, 0.0619]
    layers = np.arange(len(observed))
    
    # Compare predictions
    predicted = [com_hqs_lz_orbit(n, phase_func) for n in layers]
    residuals = (np.array(predicted) - np.array(observed)) / np.array(observed) * 100  # % error
    
    # Calculate average absolute error
    avg_abs_error = np.mean(np.abs(residuals))
    
    # Prepare results for return
    results = {
        "phase_func_name": phase_func_name,
        "planet_names": planet_names,
        "observed": observed,
        "predicted": predicted,
        "residuals": residuals,
        "avg_abs_error": avg_abs_error,
        "layers": layers
    }
    
    return results

def test_all_phase_functions():
    """
    Test all phase functions and find the best one.
    
    Returns:
    - Dictionary with results for all phase functions
    - Dictionary with results for the best phase function
    """
    all_results = {}
    best_error = float('inf')
    best_phase_func = None
    best_results = None
    
    print("Testing different phase functions for TRAPPIST-1 system:")
    print("{:<20} {:<15}".format("Phase Function", "Avg Abs Error (%)"))
    print("-" * 40)
    
    for name, func in phase_functions.items():
        results = analyze_trappist1_with_phase_function(name, func)
        all_results[name] = results
        
        print("{:<20} {:<15.2f}".format(name, results["avg_abs_error"]))
        
        if results["avg_abs_error"] < best_error:
            best_error = results["avg_abs_error"]
            best_phase_func = name
            best_results = results
    
    print("-" * 40)
    print(f"Best phase function: {best_phase_func} with {best_error:.2f}% average absolute error")
    
    return all_results, best_results

def plot_phase_function_comparison(all_results, output_dir="figures", filename="trappist1_phase_function_comparison.png"):
    """
    Create a bar chart comparing the performance of different phase functions.
    
    Parameters:
    - all_results: Dictionary with results for all phase functions
    - output_dir: Directory to save the figure
    - filename: Name of the output file
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Extract data for plotting
    phase_funcs = list(all_results.keys())
    errors = [all_results[name]["avg_abs_error"] for name in phase_funcs]
    
    # Sort by error
    sorted_indices = np.argsort(errors)
    phase_funcs = [phase_funcs[i] for i in sorted_indices]
    errors = [errors[i] for i in sorted_indices]
    
    # Create figure
    plt.figure(figsize=(14, 8))
    
    # Plot bars
    bars = plt.bar(phase_funcs, errors, color='skyblue', alpha=0.7)
    
    # Add error values on top of bars
    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                 f"{errors[i]:.2f}%", 
                 ha='center', va='bottom', fontsize=10)
    
    # Add labels and title
    plt.xlabel('Phase Function', fontsize=12)
    plt.ylabel('Average Absolute Error (%)', fontsize=12)
    plt.title('TRAPPIST-1 System: Performance of Different Phase Functions in COM-HQS-LZ Model', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # Add model parameters text
    param_text = f"Model Parameters:\n" \
                 f"LZ = {LZ}\n" \
                 f"HQS = {HQS}\n" \
                 f"a₀ = {a0_TRAPPIST} AU"
    plt.figtext(0.02, 0.02, param_text, fontsize=12, 
                bbox=dict(facecolor='white', alpha=0.8))
    
    # Save figure
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()
    
    print(f"Phase function comparison plot saved to {os.path.join(output_dir, filename)}")

def plot_best_phase_function_results(best_results, output_dir="figures", filename="trappist1_best_phase_function.png"):
    """
    Create a bar chart comparing observed and predicted values for the best phase function.
    
    Parameters:
    - best_results: Dictionary with results for the best phase function
    - output_dir: Directory to save the figure
    - filename: Name of the output file
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Extract data for plotting
    planet_names = best_results["planet_names"]
    observed = best_results["observed"]
    predicted = best_results["predicted"]
    residuals = best_results["residuals"]
    phase_func_name = best_results["phase_func_name"]
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot bars
    x = np.arange(len(planet_names))
    width = 0.35
    
    plt.bar(x - width/2, observed, width, label='Observed', color='blue', alpha=0.7)
    plt.bar(x + width/2, predicted, width, label=f'COM-HQS-LZ Predicted with {phase_func_name}', color='green', alpha=0.7)
    
    # Add error labels
    for i in range(len(planet_names)):
        plt.text(i, max(observed[i], predicted[i]) + 0.002, 
                 f"Error: {residuals[i]:.2f}%", 
                 ha='center', va='bottom', fontsize=10)
    
    # Add labels and title
    plt.xlabel('Planet', fontsize=12)
    plt.ylabel('Semi-Major Axis (AU)', fontsize=12)
    plt.title(f'TRAPPIST-1 System: Observed vs. COM-HQS-LZ Predicted with {phase_func_name}', fontsize=14)
    plt.xticks(x, planet_names, rotation=45)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add average error text
    plt.figtext(0.5, 0.01, f"Average Absolute Error: {best_results['avg_abs_error']:.2f}%", 
                ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    # Save figure
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()
    
    print(f"Best phase function results plot saved to {os.path.join(output_dir, filename)}")

def plot_continuous_best_model(best_results, output_dir="figures", filename="trappist1_best_continuous.png"):
    """
    Create a plot of the continuous model with the best phase function.
    
    Parameters:
    - best_results: Dictionary with results for the best phase function
    - output_dir: Directory to save the figure
    - filename: Name of the output file
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Extract data for plotting
    planet_names = best_results["planet_names"]
    observed = best_results["observed"]
    predicted = best_results["predicted"]
    residuals = best_results["residuals"]
    layers = best_results["layers"]
    phase_func_name = best_results["phase_func_name"]
    phase_func = phase_functions[phase_func_name]
    
    # Generate continuous model
    n_values = np.linspace(0, len(observed) - 0.5, 1000)
    a_values = [com_hqs_lz_orbit(n, phase_func) for n in n_values]
    
    # Create figure
    plt.figure(figsize=(14, 8))
    
    # Plot continuous model
    plt.plot(n_values, a_values, '-', color='green', alpha=0.7, linewidth=2, 
             label=f'COM-HQS-LZ Model with {phase_func_name}: $a_n = a_0 \\cdot \\lambda^n \\cdot (1 + \\eta \\cdot f(n))$')
    
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
    plt.title(f'Continuous COM-HQS-LZ Model for TRAPPIST-1 System with {phase_func_name}', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add legend with smaller font size and multiple columns
    plt.legend(fontsize=10, ncol=3, loc='upper left')
    
    # Add model parameters text
    param_text = f"Model Parameters:\n" \
                 f"$\\lambda$ (LZ) = {LZ}\n" \
                 f"$\\eta$ (HQS) = {HQS}\n" \
                 f"$a_0$ = {a0_TRAPPIST} AU\n" \
                 f"Phase function = {phase_func_name}"
    plt.figtext(0.02, 0.02, param_text, fontsize=12, 
                bbox=dict(facecolor='white', alpha=0.8))
    
    # Add average error text
    plt.figtext(0.98, 0.02, f"Average Absolute Error: {best_results['avg_abs_error']:.2f}%", 
                ha='right', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    # Save figure
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()
    
    print(f"Best continuous model plot saved to {os.path.join(output_dir, filename)}")

def plot_phase_functions(output_dir="figures", filename="trappist1_phase_functions.png"):
    """
    Create a plot showing the different phase functions.
    
    Parameters:
    - output_dir: Directory to save the figure
    - filename: Name of the output file
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Generate x values
    x = np.linspace(0, 6, 1000)
    
    # Create figure
    plt.figure(figsize=(14, 8))
    
    # Plot each phase function
    for name, func in phase_functions.items():
        y = [func(xi) for xi in x]
        plt.plot(x, y, label=name)
    
    # Add labels and title
    plt.xlabel('n (Octave Layer)', fontsize=12)
    plt.ylabel('f(n) (Phase Function Value)', fontsize=12)
    plt.title('Different Phase Functions for COM-HQS-LZ Model', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10, ncol=2)
    
    # Add horizontal lines at 0, 1, and -1
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.axhline(y=1, color='black', linestyle='--', alpha=0.3)
    plt.axhline(y=-1, color='black', linestyle='--', alpha=0.3)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()
    
    print(f"Phase functions plot saved to {os.path.join(output_dir, filename)}")

def main():
    """Main function to analyze TRAPPIST-1 system with different phase functions."""
    print("=" * 60)
    print("COM-HQS-LZ Analysis of TRAPPIST-1 System with Different Phase Functions")
    print("=" * 60)
    
    # Plot the different phase functions
    plot_phase_functions()
    
    # Test all phase functions
    all_results, best_results = test_all_phase_functions()
    
    # Generate visualizations
    plot_phase_function_comparison(all_results)
    plot_best_phase_function_results(best_results)
    plot_continuous_best_model(best_results)
    
    print("\n" + "=" * 60)
    print("Model parameters:")
    print(f"LZ (λ) = {LZ}")
    print(f"HQS (η) = {HQS}")
    print(f"a0 = {a0_TRAPPIST} AU")
    print(f"Best phase function = {best_results['phase_func_name']}")
    print(f"Average absolute error = {best_results['avg_abs_error']:.2f}%")
    print("=" * 60)
    
    # Return results for potential further analysis
    return all_results, best_results

if __name__ == "__main__":
    main()
