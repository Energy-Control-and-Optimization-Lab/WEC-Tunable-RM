"""
Design of Experiments (DOE) generator for parametric studies
Author: Pablo Antonio Matamala Carvajal
Date: 2025-01-21
"""

import numpy as np
import pandas as pd
from itertools import combinations
import warnings

def generate_doe_vectors(parameter_ranges, method='Box-Behnken', n_center_points=None, seed=None, parameter_names=None):
    """
    Generate Design of Experiments vectors for parametric analysis.
    
    Parameters:
    -----------
    parameter_ranges : list or dict
        Parameter ranges. Can be:
        - List of [min, max] pairs: [[5,15], [0,2], [6,13]]
        - Dict with ranges: {'D1': [5,15], 'D2': [0,2], 'D3': [6,13]}
    method : str
        DOE method to use. Options: 'Box-Behnken', 'CCD', 'Full-Factorial', 'LHS'
    n_center_points : int, optional
        Number of center points (auto-calculated if None)
    seed : int, optional
        Random seed for reproducible results
    parameter_names : list, optional
        Custom names for parameters (default: ['D1', 'D2', ...])
    
    Returns:
    --------
    dict : DOE results containing:
        - 'design_matrix': Array of experiment vectors
        - 'parameter_names': List of parameter names
        - 'parameter_ranges': Dictionary of parameter ranges
        - 'n_experiments': Number of experiments
        - 'method': DOE method used
        - 'metadata': Additional information
    """
    
    if seed is not None:
        np.random.seed(seed)
    
    # Parse parameter ranges
    if isinstance(parameter_ranges, dict):
        param_names = list(parameter_ranges.keys())
        ranges = [parameter_ranges[name] for name in param_names]
    else:
        ranges = parameter_ranges
        if parameter_names is not None:
            param_names = parameter_names
        else:
            param_names = [f'D{i+1}' for i in range(len(ranges))]
    
    n_params = len(ranges)
    
    # Validate inputs
    if n_params < 2:
        raise ValueError("At least 2 parameters required for DOE")
    
    if any(len(r) != 2 for r in ranges):
        raise ValueError("Each parameter range must be [min, max]")
    
    if any(r[0] >= r[1] for r in ranges):
        raise ValueError("Invalid range: min must be < max")
    
    print(f"🎯 Generating {method} design for {n_params} parameters:")
    for i, (name, r) in enumerate(zip(param_names, ranges)):
        print(f"   {name}: [{r[0]}, {r[1]}]")
    
    # Generate normalized design matrix (-1, 0, +1)
    if method.upper() == 'BOX-BEHNKEN':
        normalized_design = _generate_box_behnken(n_params, n_center_points)
    elif method.upper() == 'CCD':
        normalized_design = _generate_ccd(n_params, n_center_points)
    elif method.upper() == 'FULL-FACTORIAL':
        normalized_design = _generate_full_factorial(n_params)
    elif method.upper() == 'LHS':
        n_samples = n_center_points if n_center_points else max(50, 10*n_params)
        normalized_design = _generate_lhs(n_params, n_samples)
    else:
        raise ValueError(f"Method '{method}' not supported. Use: 'Box-Behnken', 'CCD', 'Full-Factorial', 'LHS'")
    
    # Scale to physical ranges
    design_matrix = _scale_to_physical_ranges(normalized_design, ranges)
    
    # Create results dictionary
    results = {
        'design_matrix': design_matrix,
        'parameter_names': param_names,
        'parameter_ranges': dict(zip(param_names, ranges)),
        'n_experiments': len(design_matrix),
        'method': method,
        'metadata': {
            'n_parameters': n_params,
            'normalized_design': normalized_design,
            'center_points': np.sum(np.all(normalized_design == 0, axis=1)),
            'factorial_points': np.sum(np.all(np.abs(normalized_design) == 1, axis=1)),
            'axial_points': len(normalized_design) - np.sum(np.all(normalized_design == 0, axis=1)) - np.sum(np.all(np.abs(normalized_design) == 1, axis=1))
        }
    }
    
    print(f"✅ Generated {results['n_experiments']} experiments:")
    print(f"   - Center points: {results['metadata']['center_points']}")
    print(f"   - Factorial points: {results['metadata']['factorial_points']}")
    print(f"   - Axial points: {results['metadata']['axial_points']}")
    
    return results


def _generate_box_behnken(n_params, n_center_points=None):
    """Generate Box-Behnken design matrix"""
    if n_params < 3:
        raise ValueError("Box-Behnken requires at least 3 parameters")
    
    # Auto-calculate center points if not specified
    if n_center_points is None:
        n_center_points = max(3, n_params // 2)
    
    design_points = []
    
    # Generate factorial points for each pair of variables
    for i, j in combinations(range(n_params), 2):
        # Four combinations for each pair: (-1,-1), (-1,+1), (+1,-1), (+1,+1)
        for level_i in [-1, 1]:
            for level_j in [-1, 1]:
                point = np.zeros(n_params)
                point[i] = level_i
                point[j] = level_j
                design_points.append(point)
    
    # Add center points
    center_point = np.zeros(n_params)
    for _ in range(n_center_points):
        design_points.append(center_point.copy())
    
    return np.array(design_points)


def _generate_ccd(n_params, n_center_points=None):
    """Generate Central Composite Design matrix"""
    if n_center_points is None:
        n_center_points = max(3, n_params // 2)
    
    design_points = []
    
    # Factorial points (2^k)
    for i in range(2**n_params):
        point = []
        for j in range(n_params):
            if (i >> j) & 1:
                point.append(1)
            else:
                point.append(-1)
        design_points.append(point)
    
    # Axial points (2*k)
    alpha = (2**n_params)**(1/4)  # Standard alpha for rotatability
    for i in range(n_params):
        # +alpha point
        point_pos = np.zeros(n_params)
        point_pos[i] = alpha
        design_points.append(point_pos)
        
        # -alpha point  
        point_neg = np.zeros(n_params)
        point_neg[i] = -alpha
        design_points.append(point_neg)
    
    # Center points
    center_point = np.zeros(n_params)
    for _ in range(n_center_points):
        design_points.append(center_point.copy())
    
    return np.array(design_points)


def _generate_full_factorial(n_params):
    """Generate Full Factorial design matrix"""
    if n_params > 6:
        warnings.warn(f"Full factorial with {n_params} parameters = {2**n_params} experiments. Consider other methods.")
    
    design_points = []
    for i in range(2**n_params):
        point = []
        for j in range(n_params):
            if (i >> j) & 1:
                point.append(1)
            else:
                point.append(-1)
        design_points.append(point)
    
    return np.array(design_points)


def _generate_lhs(n_params, n_samples):
    """Generate Latin Hypercube Sampling design matrix"""
    try:
        from scipy.stats import qmc
        sampler = qmc.LatinHypercube(d=n_params)
        lhs_samples = sampler.random(n=n_samples)
        # Convert from [0,1] to [-1,1]
        return 2 * lhs_samples - 1
    except ImportError:
        # Fallback to simple random sampling if scipy not available
        return 2 * np.random.random((n_samples, n_params)) - 1


def _scale_to_physical_ranges(normalized_design, ranges):
    """Scale normalized design matrix to physical parameter ranges"""
    n_experiments, n_params = normalized_design.shape
    physical_design = np.zeros_like(normalized_design)
    
    for i, (min_val, max_val) in enumerate(ranges):
        # Scale from [-1, 1] to [min_val, max_val]
        physical_design[:, i] = min_val + (max_val - min_val) * (normalized_design[:, i] + 1) / 2
    
    return physical_design


def print_design_summary(doe_results):
    """Print a summary of the DOE design"""
    print("\n" + "="*60)
    print(f"📊 DOE DESIGN SUMMARY - {doe_results['method']}")
    print("="*60)
    print(f"Parameters: {doe_results['n_experiments']} experiments × {len(doe_results['parameter_names'])} variables")
    print(f"Method: {doe_results['method']}")
    print(f"Total experiments: {doe_results['n_experiments']}")
    
    print(f"\nParameter ranges:")
    for name, range_vals in doe_results['parameter_ranges'].items():
        print(f"  {name}: [{range_vals[0]:8.3f}, {range_vals[1]:8.3f}]")
    
    print(f"\nDesign composition:")
    print(f"  - Center points: {doe_results['metadata']['center_points']}")
    print(f"  - Factorial points: {doe_results['metadata']['factorial_points']}")
    print(f"  - Axial points: {doe_results['metadata']['axial_points']}")
    
    print(f"\nFirst 5 experiment vectors:")
    for i in range(min(5, doe_results['n_experiments'])):
        vector_str = ", ".join([f"{val:7.3f}" for val in doe_results['design_matrix'][i]])
        print(f"  Exp {i+1:2d}: [{vector_str}]")
    
    if doe_results['n_experiments'] > 5:
        print(f"  ... (showing first 5 of {doe_results['n_experiments']} experiments)")


if __name__ == "__main__":
    # Example usage
    print("🧪 Testing DOE function with different cases:")
    
    # Test Case 1: Simple 3 parameters
    print("\n" + "="*50)
    print("TEST 1: Simple 3 parameters")
    simple_ranges = [[5, 15], [0, 2], [6, 13]]
    simple_doe = generate_doe_vectors(simple_ranges, method='Box-Behnken')
    print_design_summary(simple_doe)
    
    # Test Case 2: Your WEC case (6 parameters)
    print("\n" + "="*50) 
    print("TEST 2: WEC case (6 parameters)")
    wec_ranges = {
        'D1': [5, 15],      # profundidad_base_float
        'D2': [0, 2],       # offset_adicional  
        'D3': [6, 13],      # radio_float
        'D4': [25, 35],     # profundidad_spar
        'D5': [10, 20],     # radio_spar_superior
        'D6': [5, 12]       # altura_spar_superior
    }
    wec_doe = generate_doe_vectors(wec_ranges, method='Box-Behnken', seed=42)
    print_design_summary(wec_doe)
    
    # Test Case 3: CCD comparison
    print("\n" + "="*50)
    print("TEST 3: CCD method (4 parameters)")
    ccd_ranges = [[1, 10], [2, 8], [0.5, 2.5], [100, 500]]
    ccd_doe = generate_doe_vectors(ccd_ranges, method='CCD', n_center_points=5)
    print_design_summary(ccd_doe)
    
    print("\n🎉 All tests completed successfully!")
