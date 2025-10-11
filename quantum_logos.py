import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
"""
LOGOS THEORY
Author: Martin Doina 
"""

class SpiralGeometry:
    def __init__(self, initial_condition=0.893466392242225182):
        self.psi0 = initial_condition
        self.fixed_point = None
        
    def recursive_wave(self, n_iterations=15):
        """Compute the recursive wave equation ψ(n) = sin(ψ(n-1)) + exp(-ψ(n-1))"""
        psi_values = np.zeros(n_iterations)
        psi_values[0] = self.psi0
        
        for i in range(1, n_iterations):
            psi_values[i] = np.sin(psi_values[i-1]) + np.exp(-psi_values[i-1])
            
        return psi_values
    
    def compute_hqs(self, psi_values):
        """Compute HQS values: exp(-ψ)/ψ"""
        return np.exp(-psi_values) / psi_values
    
    def find_fixed_point(self):
        """Mathematically find the fixed point where ψ = sin(ψ) + exp(-ψ)"""
        def equation(psi):
            return psi - (np.sin(psi) + np.exp(-psi))
        
        self.fixed_point = fsolve(equation, self.psi0)[0]
        return self.fixed_point
    
    def analyze_convergence(self, max_iterations=20):
        """Analyze how the system converges with different recursion levels"""
        iterations = range(1, max_iterations + 1)
        errors = []
        path_lengths = []
        
        # Find the true fixed point
        true_fixed = self.find_fixed_point()
        
        for n in iterations:
            psi_vals = self.recursive_wave(n)
            final_value = psi_vals[-1]
            error = abs(final_value - true_fixed)
            errors.append(error)
            
            # Calculate path length (total distance traveled)
            path_length = np.sum(np.abs(np.diff(psi_vals)))
            path_lengths.append(path_length)
            
        return iterations, errors, path_lengths, true_fixed
    
    def spiral_curvature(self, psi_values):
        """Calculate the curvature of the spiral path in phase space"""
        hqs_values = self.compute_hqs(psi_values)
        
        # First derivatives (tangent vectors)
        dx_dt = np.gradient(psi_values)
        dy_dt = np.gradient(hqs_values)
        
        # Second derivatives (curvature)
        d2x_dt2 = np.gradient(dx_dt)
        d2y_dt2 = np.gradient(dy_dt)
        
        # Curvature formula: |x'y'' - y'x''| / (x'² + y'²)^(3/2)
        curvature = np.abs(dx_dt * d2y_dt2 - dy_dt * d2x_dt2) / (dx_dt**2 + dy_dt**2)**1.5
        
        return curvature

# =============================================================================
# MATHEMATICAL VALIDATION TESTS
# =============================================================================

def test_initial_conditions():
    """Test how different starting points affect the spiral convergence"""
    print("=== TEST 1: INITIAL CONDITION SENSITIVITY ===")
    
    initial_conditions = [0.1, 0.5, 0.893466392242225182, 1.5, 2.0]
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    plt.figure(figsize=(15, 10))
    
    for i, ic in enumerate(initial_conditions):
        sg = SpiralGeometry(ic)
        
        # Plot convergence
        iterations, errors, path_lengths, fixed_point = sg.analyze_convergence()
        
        plt.subplot(2, 2, 1)
        plt.semilogy(iterations, errors, 'o-', color=colors[i], 
                    label=f'ψ₀={ic:.3f}, Fixed={fixed_point:.6f}')
        
        plt.subplot(2, 2, 2)
        plt.plot(iterations, path_lengths, 's-', color=colors[i],
                label=f'ψ₀={ic:.3f}')
        
        # Plot spiral trajectories
        plt.subplot(2, 2, 3)
        psi_vals = sg.recursive_wave(20)
        hqs_vals = sg.compute_hqs(psi_vals)
        plt.plot(psi_vals, hqs_vals, '.-', color=colors[i], alpha=0.7,
                label=f'ψ₀={ic:.3f}')
    
    plt.subplot(2, 2, 1)
    plt.xlabel('Recursion Level (n)')
    plt.ylabel('Error |ψ(n) - ψ_fixed|')
    plt.title('A) Convergence to Fixed Point')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.xlabel('Recursion Level (n)')
    plt.ylabel('Total Path Length')
    plt.title('B) Elastic Distance Growth')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.xlabel('Ψ(n)')
    plt.ylabel('HQS(n)')
    plt.title('C) Spiral Trajectories in Phase Space')
    plt.legend()
    plt.grid(True)
    
    # Test fixed point uniqueness
    plt.subplot(2, 2, 4)
    psi_range = np.linspace(0.1, 2.0, 1000)
    equation_values = psi_range - (np.sin(psi_range) + np.exp(-psi_range))
    plt.plot(psi_range, equation_values, 'b-', linewidth=2)
    plt.axhline(0, color='red', linestyle='--', alpha=0.5)
    plt.xlabel('ψ')
    plt.ylabel('ψ - f(ψ)')
    plt.title('D) Fixed Point Equation: Roots = Fixed Points')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def test_recursion_elasticity():
    """Test the elastic distance concept by varying recursion strategies"""
    print("\n=== TEST 2: RECURSION ELASTICITY ===")
    
    sg = SpiralGeometry()
    
    # Different recursion strategies
    strategies = {
        'Linear (n=5-50)': range(5, 51, 5),
        'Exponential (2^n)': [2**n for n in range(2, 6)],
        'Fibonacci': [5, 8, 13, 21, 34]
    }
    
    plt.figure(figsize=(15, 5))
    
    for idx, (strategy_name, iterations_list) in enumerate(strategies.items()):
        path_lengths = []
        final_errors = []
        final_precision = []
        
        for n in iterations_list:
            psi_vals = sg.recursive_wave(n)
            path_length = np.sum(np.abs(np.diff(psi_vals)))
            final_error = abs(psi_vals[-1] - sg.find_fixed_point())
            
            path_lengths.append(path_length)
            final_errors.append(final_error)
            final_precision.append(-np.log10(final_error))  # Decimal precision
            
        plt.subplot(1, 3, 1)
        plt.plot(iterations_list, path_lengths, 'o-', label=strategy_name)
        
        plt.subplot(1, 3, 2)
        plt.semilogy(iterations_list, final_errors, 's-', label=strategy_name)
        
        plt.subplot(1, 3, 3)
        plt.plot(path_lengths, final_precision, '^-', label=strategy_name)
    
    plt.subplot(1, 3, 1)
    plt.xlabel('Number of Recursions (n)')
    plt.ylabel('Total Path Length')
    plt.title('Path Length vs Recursions')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.xlabel('Number of Recursions (n)')
    plt.ylabel('Final Error')
    plt.title('Precision vs Recursions')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.xlabel('Total Path Length')
    plt.ylabel('Decimal Precision (-log10(error))')
    plt.title('Elastic Distance: Precision Cost')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def test_curvature_analysis():
    """Analyze the spiral curvature and its evolution"""
    print("\n=== TEST 3: SPIRAL CURVATURE ANALYSIS ===")
    
    sg = SpiralGeometry()
    psi_vals = sg.recursive_wave(25)
    curvature = sg.spiral_curvature(psi_vals)
    hqs_vals = sg.compute_hqs(psi_vals)
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(psi_vals, hqs_vals, 'b.-', alpha=0.7)
    plt.xlabel('Ψ(n)')
    plt.ylabel('HQS(n)')
    plt.title('A) Spiral Trajectory')
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.semilogy(range(len(curvature)), curvature, 'ro-')
    plt.xlabel('Recursion Level (n)')
    plt.ylabel('Curvature κ(n)')
    plt.title('B) Spiral Curvature Evolution')
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    # Color points by curvature
    scatter = plt.scatter(psi_vals, hqs_vals, c=curvature, cmap='viridis', s=50)
    plt.colorbar(scatter, label='Curvature')
    plt.xlabel('Ψ(n)')
    plt.ylabel('HQS(n)')
    plt.title('C) Curvature Heatmap on Spiral')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Initial curvature: {curvature[0]:.6f}")
    print(f"Final curvature: {curvature[-1]:.6f}")
    print(f"Curvature decay ratio: {curvature[-1]/curvature[0]:.6f}")

# =============================================================================
# RUN ALL VALIDATION TESTS
# =============================================================================

if __name__ == "__main__":
    # Test 1: Initial conditions
    test_initial_conditions()
    
    # Test 2: Recursion elasticity  
    test_recursion_elasticity()
    
    # Test 3: Curvature analysis
    test_curvature_analysis()
    
    # Final mathematical summary
    print("\n" + "="*60)
    print("MATHEMATICAL SUMMARY")
    print("="*60)
    
    sg = SpiralGeometry()
    fixed_point = sg.find_fixed_point()
    print(f"Fixed point equation: ψ = sin(ψ) + exp(-ψ)")
    print(f"Mathematical fixed point: ψ* = {fixed_point:.15f}")
    
    # Verify fixed point
    verification = fixed_point - (np.sin(fixed_point) + np.exp(-fixed_point))
    print(f"Fixed point verification: {verification:.2e} (should be ~0)")
    
    # Show convergence for standard case
    psi_vals = sg.recursive_wave(10)
    print(f"\nConvergence example (Logos):")
    print(f"ψ(0) = {psi_vals[0]:.15f}")
    print(f"ψ(9) = {psi_vals[9]:.15f}")
    print(f"Error: {abs(psi_vals[9] - fixed_point):.2e}")
