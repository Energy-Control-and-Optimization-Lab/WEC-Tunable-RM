"""
St6A_OptimizationSetup.py - Multi-Objective Optimization with Metamodels
Author: Pablo Antonio Matamala Carvajal
Date: 2025-12-03
Description:
- Loads metamodel for specified function (P_efficiency by default)
- Displays metamodel quality and coefficients
- Runs multiple genetic algorithm optimizations for robustness
- Extracts top-N optimal designs
- Saves results in multiple formats for Capytaine validation
"""

import os
import sys
import numpy as np
import pickle
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from scipy.io import savemat
import time
import json
import warnings
warnings.filterwarnings('ignore')

#%%============================================================================
# CONFIGURATION - MODIFY THESE VALUES AS NEEDED
#%%============================================================================

# ===== MAIN CONFIGURATION =====
FUNCTION_NAME = "P_efficiency"           # Function to optimize
N_OPTIMAL_DESIGNS = 60                    # Number of optimal designs desired
OBJECTIVE_TYPE = "maximize"              # "maximize" or "minimize"
# NOTE: If using "minimize", the algorithm converges more precisely to the same minimum
# point, requiring adaptive tolerance in extract_top_designs() for diversity
MULTIPLE_RUNS = 5                        # Number of independent runs for robustness (matches number of seeds)

# ===== GENETIC ALGORITHM PARAMETERS =====
POPULATION_SIZE = 500                    # Population size (increased for robustness)
GENERATIONS = 300                        # Number of generations (increased)
CROSSOVER_PROB = 0.8                     # Crossover probability
MUTATION_PROB = 0.15                     # Mutation probability (increased for exploration)
#RANDOM_SEEDS = [20, 300, 320, 500, 852]          # Seeds for multiple runs
#RANDOM_SEEDS = [20, 300, 320, 500, 852]          # Seeds for multiple runs
#RANDOM_SEEDS = [147, 263, 419, 578, 691, 734, 825, 916, 1052, 1183]
"""
RANDOM_SEEDS = [147, 263, 419, 578, 691, 734, 825, 916, 1052, 1183,
                1245, 1367, 1489, 1521, 1678, 1742, 1896, 1934, 2057, 2145,
                2289, 2367, 2456, 2578, 2634, 2789, 2845, 2967, 3124, 3289]
"""
RANDOM_SEEDS = [147, 263, 419, 578, 691, 734, 825, 916, 1052, 1183,
                1245, 1367, 1489, 1521, 1678, 1742, 1896, 1934, 2057, 2145,
                2289, 2367, 2456, 2578, 2634, 2789, 2845, 2967, 3124, 3289,
                4147, 4263, 4419, 4578, 4691, 4734, 4825, 4916, 5052, 5183,
                5245, 5367, 5489, 5521, 5678, 5742, 5896, 5934, 6057, 6145,
                6289, 6367, 6456, 6578, 6634, 6789, 6845, 6967, 7124, 7289]


# ===== DOE PARAMETER RANGES (slightly re0duced to avoid boundary solutions) =====
'''
PARAMETER_RANGES = {
    'D1': [0.7, 0.7],      # Float diameter [m]
    'D2': [0.05, 0.15],    # Float draft [m]  
    'D3': [5, 5],         # Float angle [°]
    'D4': [1.3, 1.3],      # Spar draft [m]
    'D5': [1.0, 1.0],      # Spar plate diameter [m]
    'D6': [300, 700]      # PTO damping [kg/s]         # PTO damping [kg/s] (reduced from [600, 1000])
}
'''

PARAMETER_RANGES = {
    'D1': [0.6, 1.0],      # Float diameter [m]
    'D2': [0.05, 0.15],    # Float draft [m]  
    'D3': [0, 20],         # Float angle [°]
    'D4': [1.2, 1.6],      # Spar draft [m]
    'D5': [0.6, 1.0],      # Spar plate diameter [m]
    'D6': [300, 700]      # PTO damping [kg/s]         # PTO damping [kg/s] (reduced from [600, 1000])
}

PARAMETER_NAMES = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6']

# ===== PATHS =====
METAMODEL_FOLDER = "EcoData/MetaModel/Analysis/individual_enhanced_metamodels"
OUTPUT_FOLDER = "EcoData/Optimization"

#%%============================================================================
# UTILITY FUNCTIONS
#%%============================================================================

def create_output_folders():
    """Create output directory structure"""
    folders = [
        OUTPUT_FOLDER,
        os.path.join(OUTPUT_FOLDER, "final_results"),
        os.path.join(OUTPUT_FOLDER, "logs"),
        os.path.join(OUTPUT_FOLDER, "raw_results")
    ]
    
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
    
    print(f"📁 Output folders created:")
    print(f"   {OUTPUT_FOLDER}/")
    print(f"   ├── final_results/")
    print(f"   ├── logs/")
    print(f"   └── raw_results/")

def get_coefficient_name(feature_name, parameter_mapping):
    """Convert sklearn feature name to mathematical coefficient name"""
    
    # Intercept (this shouldn't be called for intercept, but included for completeness)
    if feature_name == '1':
        return 'β₀', 'Intercept'
    
    # Linear terms: D1 -> β₁, D2 -> β₂, etc.
    for i, param in enumerate(parameter_mapping):
        if feature_name == param:
            return f'β{i+1}', f'{param}'
    
    # Quadratic terms: D1^2 -> β₁₁, D2^2 -> β₂₂, etc.
    for i, param in enumerate(parameter_mapping):
        if feature_name == f'{param}^2':
            return f'β{i+1}{i+1}', f'{param}²'
    
    # Interaction terms: D1 D2 -> β₁₂, D1 D3 -> β₁₃, etc.
    if ' ' in feature_name:
        params = feature_name.split(' ')
        if len(params) == 2:
            try:
                i = parameter_mapping.index(params[0]) + 1
                j = parameter_mapping.index(params[1]) + 1
                return f'β{i}{j}', f'{params[0]}×{params[1]}'
            except ValueError:
                pass
    
    return f'β_?', feature_name

#%%============================================================================
# METAMODEL LOADING AND ANALYSIS
#%%============================================================================

def load_metamodel(function_name):
    """Load metamodel from file"""
    
    print(f"\n{'='*70}")
    print(f"LOADING METAMODEL: {function_name}")
    print(f"{'='*70}")
    
    # Construct file path
    metamodel_file = os.path.join(METAMODEL_FOLDER, f"{function_name}_enhanced_metamodel.pkl")
    
    # Check if file exists
    if not os.path.exists(metamodel_file):
        print(f"❌ ERROR: Metamodel file not found: {metamodel_file}")
        print(f"📁 Available metamodels in folder:")
        
        try:
            available_files = [f for f in os.listdir(METAMODEL_FOLDER) if f.endswith('_enhanced_metamodel.pkl')]
            for i, file in enumerate(available_files, 1):
                function_name = file.replace('_enhanced_metamodel.pkl', '')
                print(f"   {i:2d}. {function_name}")
        except FileNotFoundError:
            print(f"   ❌ Folder not found: {METAMODEL_FOLDER}")
        
        return None
    
    # Load metamodel
    try:
        with open(metamodel_file, 'rb') as f:
            metamodel_data = pickle.load(f)
        
        print(f"✅ Metamodel loaded successfully from: {metamodel_file}")
        return metamodel_data
        
    except Exception as e:
        print(f"❌ ERROR loading metamodel: {e}")
        return None

def display_metamodel_info(metamodel_data):
    """Display metamodel quality and coefficients"""
    
    print(f"\n📊 METAMODEL QUALITY:")
    print(f"{'='*50}")
    
    # Extract statistics
    stats = metamodel_data['statistics']
    r2 = stats['r2']
    mape = stats['mape']
    model_adequate = stats['model_adequate']
    
    # Quality classification
    if r2 > 0.9 and mape < 5:
        quality = "🟢 Excellent"
    elif r2 > 0.85 and mape < 10:
        quality = "🟡 Good"
    else:
        quality = "🔴 Poor"
    
    print(f"   R² = {r2:.3f}")
    print(f"   MAPE = {mape:.1f}%")
    print(f"   Model adequate: {'✅ YES' if model_adequate else '❌ NO'}")
    print(f"   Quality: {quality}")
    
    # Extract model information
    model = metamodel_data['model']
    poly_features = metamodel_data['polynomial_features']
    parameter_names = metamodel_data['parameter_names']
    
    # Get feature names and coefficients
    feature_names = poly_features.get_feature_names_out(parameter_names)
    coefficients = model.coef_
    intercept = model.intercept_
    
    print(f"\n📋 COEFFICIENTS (28 total: 1 intercept + 27 features):")
    print(f"{'='*70}")
    
    # Display intercept (β₀ is stored separately in sklearn)
    print(f"\nINTERCEPT:")
    print(f"   β₀ = {intercept:.6f}")
    print(f"   ℹ️  Intercept is stored separately from coefficients in sklearn")
    
    # Group coefficients by type
    # Note: model.coef_[0] is ≈0 (placeholder for '1' feature), so we use model.coef_[1:]
    # feature_names[1:] has 27 elements (D1, D2, ..., D6²)
    # model.coef_[1:] also has 27 elements (β₁ through β₂₇)
    linear_coeffs = []
    interaction_coeffs = []
    quadratic_coeffs = []
    
    print(f"\n📊 COEFFICIENT MAPPING (CORRECTED):")
    print(f"   ⚠️  model.coef_[0] ≈ 0 (placeholder for '1' feature when include_bias=True)")
    print(f"   ✅ model.coef_[1] = β₁ (for {parameter_names[0]})")
    print(f"   ✅ model.coef_[2] = β₂ (for {parameter_names[1]})")
    print(f"   ✅ ...and so on for all remaining coefficients")
    print(f"   📝 Using model.coef_[1:] to skip the bias placeholder")
    
    for i, (feature_name, coeff) in enumerate(zip(feature_names[1:], model.coef_[1:])):  # Skip intercept feature AND first coefficient
        coeff_name, readable_name = get_coefficient_name(feature_name, parameter_names)
        
        if '^2' in feature_name:
            quadratic_coeffs.append((coeff_name, readable_name, coeff))
        elif ' ' in feature_name:
            interaction_coeffs.append((coeff_name, readable_name, coeff))
        else:
            linear_coeffs.append((coeff_name, readable_name, coeff))
    
    # Display linear effects
    print(f"\nLINEAR EFFECTS ({len(linear_coeffs)}):")
    for coeff_name, readable_name, coeff in linear_coeffs:
        print(f"   {coeff_name:8s} ({readable_name:2s}) = {coeff:10.6f}")
    
    # Display interaction effects
    print(f"\nINTERACTION EFFECTS ({len(interaction_coeffs)}):")
    for coeff_name, readable_name, coeff in interaction_coeffs:
        highlight = "  ⭐ HIGH IMPACT" if abs(coeff) > np.std([c[2] for c in interaction_coeffs]) else ""
        print(f"   {coeff_name:8s} ({readable_name:8s}) = {coeff:10.6f}{highlight}")
    
    # Display quadratic effects
    print(f"\nQUADRATIC EFFECTS ({len(quadratic_coeffs)}):")
    for coeff_name, readable_name, coeff in quadratic_coeffs:
        highlight = "  ⭐ DOMINANT" if abs(coeff) == max([abs(c[2]) for c in quadratic_coeffs]) else ""
        print(f"   {coeff_name:8s} ({readable_name:4s}) = {coeff:10.6f}{highlight}")
    
    return {
        'model': model,
        'poly_features': poly_features,
        'parameter_names': parameter_names,
        'statistics': stats,
        'feature_names': feature_names,
        'coefficients': coefficients,
        'intercept': intercept
    }

#%%============================================================================
# OPTIMIZATION FUNCTIONS
#%%============================================================================

def create_objective_function(model, poly_features, objective_type):
    """Create objective function for optimization"""
    
    def objective_function(x):
        """Evaluate metamodel at point x"""
        # Reshape input for polynomial features
        x_reshaped = np.array(x).reshape(1, -1)
        
        # Transform to polynomial features
        x_poly = poly_features.transform(x_reshaped)
        
        # Predict using metamodel
        prediction = model.predict(x_poly)[0]
        
        # CRITICAL FIX: Add penalty for unrealistic extrapolation
        # Based on training data range: P_efficiency max ≈ 4.9
        MAX_REALISTIC_VALUE = 7.0  # Slightly above training max
        
        if prediction > MAX_REALISTIC_VALUE:
            # Heavy penalty for extrapolation beyond realistic range
            penalty = 1000 * (prediction - MAX_REALISTIC_VALUE)**2
            prediction = MAX_REALISTIC_VALUE - penalty
        
        # Additional penalty for being too close to parameter bounds
        # Encourage interior solutions, not boundary solutions
        boundary_penalty = 0
        bounds = [PARAMETER_RANGES[param] for param in PARAMETER_NAMES]
        
        for i, (value, (min_val, max_val)) in enumerate(zip(x, bounds)):
            range_size = max_val - min_val
            tolerance = 0.05 * range_size  # 5% tolerance from boundaries
            
            if value < (min_val + tolerance) or value > (max_val - tolerance):
                boundary_penalty += 0.1  # Small penalty for boundary solutions
        
        prediction -= boundary_penalty
        
        # Return negative for maximization (minimizer algorithms)
        if objective_type == "maximize":
            return -prediction
        else:
            return prediction
    
    return objective_function

def run_single_optimization(objective_function, bounds, seed, run_id):
    """Run single optimization with differential evolution"""
    
    print(f"\n🚀 RUN {run_id}/{len(RANDOM_SEEDS)} (seed={seed})")
    print(f"   Algorithm: Differential Evolution")
    print(f"   Population: {POPULATION_SIZE}")
    print(f"   Generations: {GENERATIONS}")
    
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Track all evaluations for raw_results
    all_evaluations = []
    convergence_history = []
    generation_counter = [0]  # Use list to modify in callback
    
    def evaluation_callback(x, f_val):
        """Track every function evaluation"""
        all_evaluations.append({
            'x': x.copy(),
            'f_val': f_val,
            'generation': generation_counter[0],
            'evaluation_id': len(all_evaluations) + 1
        })
    
    # Enhanced objective function wrapper to track evaluations
    def tracked_objective_function(x):
        f_val = objective_function(x)
        evaluation_callback(x, f_val)
        return f_val
    
    # Progress callback for convergence tracking
    def callback(x, convergence):
        generation_counter[0] += 1
        current_best = -tracked_objective_function(x) if OBJECTIVE_TYPE == "maximize" else tracked_objective_function(x)
        convergence_history.append(current_best)
        return False
    
    # Record start time
    start_time = time.time()
    
    # Run optimization
    result = differential_evolution(
        tracked_objective_function,
        bounds=bounds,
        seed=seed,
        maxiter=GENERATIONS,
        popsize=int(POPULATION_SIZE/len(bounds)),
        mutation=(0.5, 1),
        recombination=CROSSOVER_PROB,
        disp=False,
        callback=callback
    )
    
    # Record execution time
    execution_time = time.time() - start_time
    
    # Extract results
    optimal_x = result.x
    optimal_value = result.fun
    
    # Convert back to maximization if needed
    if OBJECTIVE_TYPE == "maximize":
        optimal_value = -optimal_value
    
    print(f"   ✅ Completed: Best = {optimal_value:.6f}")
    print(f"   📊 Evaluations: {len(all_evaluations)}, Time: {execution_time:.1f}s")
    
    # Prepare detailed results for raw_results
    detailed_result = {
        'run_id': run_id,
        'seed': seed,
        'algorithm_config': {
            'population_size': POPULATION_SIZE,
            'generations': GENERATIONS,
            'bounds': bounds,
            'mutation': (0.5, 1),
            'recombination': CROSSOVER_PROB
        },
        'optimal_x': optimal_x,
        'optimal_value': optimal_value,
        'scipy_result': {
            'x': result.x,
            'fun': result.fun,
            'nfev': result.nfev,
            'nit': result.nit,
            'success': result.success,
            'message': result.message
        },
        'all_evaluations': all_evaluations,
        'convergence_history': convergence_history,
        'execution_time': execution_time,
        'function_evaluations': len(all_evaluations),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    return {
        'optimal_x': optimal_x,
        'optimal_value': optimal_value,
        'success': result.success,
        'nfev': result.nfev,
        'result': result,
        'run_id': run_id,
        'seed': seed,
        'detailed_result': detailed_result  # NEW: Raw results data
    }

def run_multiple_optimizations(metamodel_info):
    """Run multiple optimizations for robustness"""
    
    print(f"\n{'='*70}")
    print(f"OPTIMIZATION CONFIGURATION")
    print(f"{'='*70}")
    
    # Setup optimization
    model = metamodel_info['model']
    poly_features = metamodel_info['poly_features']
    
    # Create objective function
    objective_function = create_objective_function(model, poly_features, OBJECTIVE_TYPE)
    
    # Setup bounds
    bounds = [PARAMETER_RANGES[param] for param in PARAMETER_NAMES]
    
    print(f"📊 Parameter ranges (from DOE):")
    for param_name, bound in zip(PARAMETER_NAMES, bounds):
        print(f"   {param_name}: [{bound[0]:8.2f}, {bound[1]:8.2f}]")
    
    print(f"\n⚙️  Optimization settings:")
    print(f"   Objective: {OBJECTIVE_TYPE.upper()} {FUNCTION_NAME}")
    print(f"   Algorithm: Differential Evolution")
    print(f"   Multiple runs: {MULTIPLE_RUNS}")
    print(f"   Seeds: {RANDOM_SEEDS}")
    
    # Run multiple optimizations
    print(f"\n🚀 RUNNING {MULTIPLE_RUNS} OPTIMIZATION RUNS...")
    print(f"{'='*70}")
    
    all_results = []
    for run_id, seed in enumerate(RANDOM_SEEDS, 1):
        result = run_single_optimization(objective_function, bounds, seed, run_id)
        all_results.append(result)
    
    return all_results

def analyze_multiple_results(all_results, metamodel_info):
    """Analyze results from multiple runs"""
    
    print(f"\n{'='*70}")
    print(f"ANALYZING MULTIPLE RUNS")
    print(f"{'='*70}")
    
    # Extract best values from each run
    best_values = [result['optimal_value'] for result in all_results]
    best_designs = [result['optimal_x'] for result in all_results]
    
    # Calculate statistics
    mean_value = np.mean(best_values)
    std_value = np.std(best_values)
    min_value = np.min(best_values)
    max_value = np.max(best_values)
    
    print(f"📊 CONSISTENCY ANALYSIS:")
    print(f"   Best values: {best_values}")
    print(f"   Mean: {mean_value:.6f}")
    print(f"   Std:  {std_value:.6f}")
    print(f"   Range: [{min_value:.6f}, {max_value:.6f}]")
    print(f"   Spread: {max_value - min_value:.6f}")
    
    # Calculate consistency score
    consistency_score = 1.0 - (std_value / abs(mean_value) if mean_value != 0 else 1.0)
    
    if consistency_score > 0.95:
        consistency_rating = "🟢 HIGH"
    elif consistency_score > 0.85:
        consistency_rating = "🟡 MEDIUM"
    else:
        consistency_rating = "🔴 LOW"
    
    print(f"   Consistency score: {consistency_score:.3f} ({consistency_rating})")
    
    # Generate population of all good designs
    model = metamodel_info['model']
    poly_features = metamodel_info['poly_features']
    
    # Evaluate all designs with metamodel to get precise values
    all_designs = []
    all_values = []
    
    for result in all_results:
        x = result['optimal_x']
        
        # Evaluate with metamodel for consistency
        x_reshaped = np.array(x).reshape(1, -1)
        x_poly = poly_features.transform(x_reshaped)
        value = model.predict(x_poly)[0]
        
        all_designs.append(x)
        all_values.append(value)
    
    return {
        'all_designs': all_designs,
        'all_values': all_values,
        'consistency_score': consistency_score,
        'consistency_rating': consistency_rating,
        'stats': {
            'mean': mean_value,
            'std': std_value,
            'min': min_value,
            'max': max_value
        }
    }

#%%============================================================================
# POST-PROCESSING AND SAVING
#%%============================================================================

def extract_top_designs(analysis_results, n_designs):
    """Extract top N unique designs with adaptive tolerance for minimize vs maximize"""
    
    designs = np.array(analysis_results['all_designs'])
    values = np.array(analysis_results['all_values'])
    
    # DEBUG: Print all results from multiple runs
    print(f"\n🔍 DEBUG - All results from {len(designs)} runs:")
    for i, (design, value) in enumerate(zip(designs, values)):
        print(f"   Run {i+1}: P_efficiency = {value:.6f}")
        print(f"           D1={design[0]:.3f}, D2={design[1]:.3f}, D3={design[2]:.3f}")
        print(f"           D4={design[3]:.3f}, D5={design[4]:.3f}, D6={design[5]:.3f}")
    
    # Sort by objective value (descending for maximization, ascending for minimization)
    if OBJECTIVE_TYPE == "maximize":
        sorted_indices = np.argsort(values)[::-1]
    else:
        sorted_indices = np.argsort(values)
    
    # Extract top designs
    top_designs = designs[sorted_indices]
    top_values = values[sorted_indices]
    
    print(f"\n🔍 DEBUG - After sorting:")
    for i, (design, value) in enumerate(zip(top_designs, top_values)):
        print(f"   Rank {i+1}: P_efficiency = {value:.6f}")
    
    # ADAPTIVE TOLERANCE: More lenient for minimize, normal for maximize
    if OBJECTIVE_TYPE == "minimize":
        # For minimize: algorithms converge to same minimum more precisely
        # Use larger tolerance to allow more diversity
        tolerance = 1e-2
        print(f"   🎯 Using MINIMIZE tolerance: {tolerance} (more lenient)")
    else:
        # For maximize: natural diversity in optimal region
        tolerance = 1e-3
        print(f"   🎯 Using MAXIMIZE tolerance: {tolerance} (standard)")
    
    # Remove duplicates (designs that are very similar)
    unique_designs = []
    unique_values = []
    
    for i, (design, value) in enumerate(zip(top_designs, top_values)):
        is_unique = True
        for existing_design in unique_designs:
            if np.allclose(design, existing_design, atol=tolerance):
                is_unique = False
                break
        
        if is_unique:
            unique_designs.append(design)
            unique_values.append(value)
        else:
            print(f"   🔄 Design {i+1} considered duplicate (tolerance={tolerance})")
            
        # Stop when we have enough unique designs
        if len(unique_designs) >= n_designs:
            break
    
    print(f"\n🔍 DEBUG - Final unique designs: {len(unique_designs)}")
    
    # If still not enough unique designs, relax tolerance further
    if len(unique_designs) < n_designs and OBJECTIVE_TYPE == "minimize":
        print(f"   ⚠️  Only {len(unique_designs)} unique designs found, relaxing tolerance...")
        
        # Try with even more relaxed tolerance
        tolerance = 5e-2  # 0.05
        unique_designs = []
        unique_values = []
        
        for i, (design, value) in enumerate(zip(top_designs, top_values)):
            is_unique = True
            for existing_design in unique_designs:
                if np.allclose(design, existing_design, atol=tolerance):
                    is_unique = False
                    break
            
            if is_unique:
                unique_designs.append(design)
                unique_values.append(value)
                
            if len(unique_designs) >= n_designs:
                break
        
        print(f"   📈 Relaxed tolerance to {tolerance}: Found {len(unique_designs)} designs")
    
    # Convert to arrays
    unique_designs = np.array(unique_designs)
    unique_values = np.array(unique_values)
    
    return unique_designs[:n_designs], unique_values[:n_designs]

def save_results(top_designs, top_values, metamodel_info, analysis_results, all_raw_results):
    """Save results in multiple formats including raw results"""
    
    print(f"\n{'='*70}")
    print(f"SAVING RESULTS")
    print(f"{'='*70}")
    
    # Prepare data structure
    results_data = {
        'top_designs': top_designs,
        'function_values': top_values,
        'parameter_names': PARAMETER_NAMES,
        'function_name': FUNCTION_NAME,
        'optimization_config': {
            'population_size': POPULATION_SIZE,
            'generations': GENERATIONS,
            'multiple_runs': MULTIPLE_RUNS,
            'seeds': RANDOM_SEEDS,
            'objective_type': OBJECTIVE_TYPE
        },
        'metamodel_quality': metamodel_info['statistics'],
        'consistency_analysis': {
            'score': analysis_results['consistency_score'],
            'rating': analysis_results['consistency_rating'],
            'stats': analysis_results['stats']
        }
    }
    
    # Create base filename
    base_filename = f"{FUNCTION_NAME}_optimal_designs"
    
    # Save as PKL
    pkl_path = os.path.join(OUTPUT_FOLDER, "final_results", f"{base_filename}.pkl")
    with open(pkl_path, 'wb') as f:
        pickle.dump(results_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"✅ PKL saved: {pkl_path}")
    
    # Save as CSV
    csv_data = []
    for i, (design, value) in enumerate(zip(top_designs, top_values)):
        row = {'Rank': i+1}
        for j, param_name in enumerate(PARAMETER_NAMES):
            row[param_name] = design[j]
        row[f'{FUNCTION_NAME}_predicted'] = value
        csv_data.append(row)
    
    csv_df = pd.DataFrame(csv_data)
    csv_path = os.path.join(OUTPUT_FOLDER, "final_results", f"{base_filename}.csv")
    csv_df.to_csv(csv_path, index=False)
    print(f"✅ CSV saved: {csv_path}")
    
    # Save as MAT
    try:
        mat_data = {
            'top_designs': top_designs,
            'function_values': top_values,
            'parameter_names': PARAMETER_NAMES,
            'function_name': FUNCTION_NAME
        }
        mat_path = os.path.join(OUTPUT_FOLDER, "final_results", f"{base_filename}.mat")
        savemat(mat_path, mat_data, do_compression=True)
        print(f"✅ MAT saved: {mat_path}")
    except Exception as e:
        print(f"⚠️  Could not save MAT: {e}")
    
    # ==========================================
    # NEW: Save RAW RESULTS for detailed analysis
    # ==========================================
    
    print(f"\n📊 Saving RAW RESULTS for detailed analysis...")
    
    # 1. Save individual run details
    for result in all_raw_results:
        detailed_data = result['detailed_result']
        run_file = os.path.join(OUTPUT_FOLDER, "raw_results", 
                               f"{FUNCTION_NAME}_run_{result['run_id']:03d}_detailed.pkl")
        with open(run_file, 'wb') as f:
            pickle.dump(detailed_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"   📁 Run {result['run_id']} details: {os.path.basename(run_file)}")
    
    # 2. Save convergence data across all runs
    convergence_data = {
        'all_runs_convergence': {},
        'convergence_summary': {},
        'cross_run_statistics': {}
    }
    
    all_convergence = []
    for result in all_raw_results:
        run_id = result['run_id']
        conv_history = result['detailed_result']['convergence_history']
        convergence_data['all_runs_convergence'][f'run_{run_id}'] = conv_history
        all_convergence.append(conv_history)
    
    # Calculate convergence statistics
    if all_convergence:
        max_length = max(len(conv) for conv in all_convergence)
        # Pad shorter convergence histories with their last values
        padded_convergence = []
        for conv in all_convergence:
            padded = conv + [conv[-1]] * (max_length - len(conv))
            padded_convergence.append(padded)
        
        padded_convergence = np.array(padded_convergence)
        convergence_data['cross_run_statistics'] = {
            'mean_convergence': np.mean(padded_convergence, axis=0).tolist(),
            'std_convergence': np.std(padded_convergence, axis=0).tolist(),
            'min_convergence': np.min(padded_convergence, axis=0).tolist(),
            'max_convergence': np.max(padded_convergence, axis=0).tolist()
        }
    
    convergence_file = os.path.join(OUTPUT_FOLDER, "raw_results", 
                                   f"{FUNCTION_NAME}_convergence_data.pkl")
    with open(convergence_file, 'wb') as f:
        pickle.dump(convergence_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"   📈 Convergence data: {os.path.basename(convergence_file)}")
    
    # 3. Save ALL evaluations as CSV for easy analysis
    all_evaluations_data = []
    for result in all_raw_results:
        run_id = result['run_id']
        evaluations = result['detailed_result']['all_evaluations']
        
        for eval_data in evaluations:
            row = {
                'run_id': run_id,
                'evaluation_id': eval_data['evaluation_id'],
                'generation': eval_data['generation']
            }
            # Add parameter values
            for i, param_name in enumerate(PARAMETER_NAMES):
                row[param_name] = eval_data['x'][i]
            
            # Add function value (convert back to maximization)
            f_val = eval_data['f_val']
            if OBJECTIVE_TYPE == "maximize":
                f_val = -f_val
            row[f'{FUNCTION_NAME}_predicted'] = f_val
            
            all_evaluations_data.append(row)
    
    evaluations_df = pd.DataFrame(all_evaluations_data)
    evaluations_csv = os.path.join(OUTPUT_FOLDER, "raw_results", 
                                  f"{FUNCTION_NAME}_all_evaluations.csv")
    evaluations_df.to_csv(evaluations_csv, index=False)
    print(f"   📋 All evaluations: {os.path.basename(evaluations_csv)} ({len(all_evaluations_data)} points)")
    
    # 4. Save optimization statistics
    optimization_stats = {
        'total_runs': len(all_raw_results),
        'total_evaluations': len(all_evaluations_data),
        'algorithm': 'Differential Evolution',
        'optimization_config': {
            'population_size': POPULATION_SIZE,
            'generations': GENERATIONS,
            'seeds': RANDOM_SEEDS
        },
        'execution_summary': {
            'successful_runs': sum(1 for r in all_raw_results if r['detailed_result']['scipy_result']['success']),
            'total_execution_time': sum(r['detailed_result']['execution_time'] for r in all_raw_results),
            'average_evaluations_per_run': len(all_evaluations_data) / len(all_raw_results)
        },
        'best_results_per_run': [
            {
                'run_id': r['run_id'],
                'seed': r['seed'],
                'best_value': r['optimal_value'],
                'execution_time': r['detailed_result']['execution_time'],
                'evaluations': r['detailed_result']['function_evaluations']
            }
            for r in all_raw_results
        ]
    }
    
    stats_file = os.path.join(OUTPUT_FOLDER, "raw_results", 
                             f"{FUNCTION_NAME}_optimization_stats.json")
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(optimization_stats, f, indent=2, ensure_ascii=False)
    print(f"   📊 Optimization stats: {os.path.basename(stats_file)}")
    
    # Save optimization log (existing functionality)
    log_path = os.path.join(OUTPUT_FOLDER, "logs", f"{FUNCTION_NAME}_optimization_log.txt")
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(f"{'='*80}\n")
        f.write(f"St6A OPTIMIZATION LOG - {FUNCTION_NAME}\n")
        f.write(f"{'='*80}\n\n")
        
        f.write(f"METAMODEL INFO:\n")
        f.write(f"✅ Function: {FUNCTION_NAME}\n")
        f.write(f"✅ R² = {metamodel_info['statistics']['r2']:.3f}\n")
        f.write(f"✅ MAPE = {metamodel_info['statistics']['mape']:.1f}%\n")
        f.write(f"✅ Model adequate: {'YES' if metamodel_info['statistics']['model_adequate'] else 'NO'}\n\n")
        
        f.write(f"OPTIMIZATION CONFIG:\n")
        f.write(f"✅ Population: {POPULATION_SIZE}\n")
        f.write(f"✅ Generations: {GENERATIONS}\n")
        f.write(f"✅ Multiple runs: {MULTIPLE_RUNS}\n")
        f.write(f"✅ Seeds: {RANDOM_SEEDS}\n\n")
        
        f.write(f"RAW RESULTS SAVED:\n")
        f.write(f"✅ Individual runs: {len(all_raw_results)} files\n")
        f.write(f"✅ Total evaluations: {len(all_evaluations_data)}\n")
        f.write(f"✅ Convergence data: 1 file\n")
        f.write(f"✅ Statistics: 1 JSON file\n\n")
        
        f.write(f"RESULTS:\n")
        f.write(f"✅ Consistency score: {analysis_results['consistency_score']:.3f}\n")
        f.write(f"✅ Rating: {analysis_results['consistency_rating']}\n\n")
        
        f.write(f"TOP {len(top_designs)} DESIGNS:\n")
        for i, (design, value) in enumerate(zip(top_designs, top_values)):
            f.write(f"Rank {i+1}: {FUNCTION_NAME} = {value:.6f}\n")
            for j, param_name in enumerate(PARAMETER_NAMES):
                f.write(f"  {param_name} = {design[j]:.6f}\n")
            f.write(f"\n")
    
    print(f"✅ Log saved: {log_path}")
    
    print(f"\n📁 RAW RESULTS SUMMARY:")
    print(f"   📊 {len(all_raw_results)} detailed run files")
    print(f"   📈 1 convergence analysis file")  
    print(f"   📋 1 CSV with {len(all_evaluations_data)} evaluations")
    print(f"   📊 1 JSON with statistics")
    
    return [pkl_path, csv_path, log_path]

def display_final_results(top_designs, top_values):
    """Display final results on screen"""
    
    print(f"\n{'='*70}")
    print(f"🏆 TOP {len(top_designs)} OPTIMAL DESIGNS")
    print(f"{'='*70}")
    
    for i, (design, value) in enumerate(zip(top_designs, top_values)):
        print(f"\nRANK {i+1}: {FUNCTION_NAME} = {value:.6f}")
        
        # Display parameters in two columns
        for j in range(0, len(PARAMETER_NAMES), 2):
            line = f"   {PARAMETER_NAMES[j]} = {design[j]:8.3f}"
            if j+1 < len(PARAMETER_NAMES):
                line += f"      {PARAMETER_NAMES[j+1]} = {design[j+1]:8.3f}"
            print(line)

#%%============================================================================
# MAIN EXECUTION
#%%============================================================================

def main():
    """Main execution function"""
    
    print("="*70)
    print("St6A - OPTIMIZATION SETUP AND EXECUTION")
    print("="*70)
    
    # Create output folders
    create_output_folders()
    
    # Load metamodel
    metamodel_data = load_metamodel(FUNCTION_NAME)
    if metamodel_data is None:
        print("❌ Failed to load metamodel. Exiting.")
        return
    
    # Display metamodel information
    metamodel_info = display_metamodel_info(metamodel_data)
    
    # Run multiple optimizations
    all_results = run_multiple_optimizations(metamodel_info)
    
    # Analyze results
    analysis_results = analyze_multiple_results(all_results, metamodel_info)
    
    # Extract top designs
    top_designs, top_values = extract_top_designs(analysis_results, N_OPTIMAL_DESIGNS)
    
    # Save results (including raw results)
    saved_files = save_results(top_designs, top_values, metamodel_info, analysis_results, all_results)
    
    # Display final results
    display_final_results(top_designs, top_values)
    
    # Final summary
    print(f"\n{'='*70}")
    print(f"🎉 OPTIMIZATION COMPLETED SUCCESSFULLY")
    print(f"{'='*70}")
    print(f"📊 Best {FUNCTION_NAME} achieved: {top_values[0]:.6f}")
    print(f"📊 Consistency rating: {analysis_results['consistency_rating']}")
    print(f"💾 Files saved: {len(saved_files)}")
    for file_path in saved_files:
        print(f"   📁 {file_path}")
    
    # Count raw results files
    raw_results_count = len(all_results) + 3  # individual runs + convergence + evaluations + stats
    print(f"📊 Raw results files: {raw_results_count}")
    print(f"   📁 EcoData/Optimization/raw_results/ (detailed analysis data)")
    
    print(f"\n🎯 NEXT STEPS:")
    print(f"   1. Use optimal designs for Capytaine validation")
    print(f"   2. Compare predicted vs real {FUNCTION_NAME}")
    print(f"   3. Analyze raw_results/ for convergence patterns")
    print(f"   4. Identify true optimal design")
    
    print(f"\n{'='*70}")

if __name__ == "__main__":
    main()
