"""
St1_DoeValues.py - Design of Experiments Generator
Generates Box-Behnken design and saves DOE values for WEC parametric analysis

Author: Pablo Antonio Matamala Carvajal
Date: 2025-01-21
"""

import numpy as np
import pickle
import os
import sys

# Add EcoFunctions to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'EcoFunctions'))
from EcoFunctions.Eco_DOE import generate_doe_vectors

#%%============================================================================
# CONFIGURATION
#==============================================================================

# Output configuration
OUTPUT_FOLDER = "EcoData/St1_DOEvalues"
OUTPUT_FILENAME = "DOE_values"  # Without extension

# WEC Design Parameters
wec_parameters = {
    'D1': [0.6, 1],      # Floater diameter [m]
    'D2': [0.05, 0.15],       # Floater draft [m]
    'D3': [0, 20],          # Floater angle [deg]
    'D4': [1.2, 1.6],     # Spar draft [m]
    'D5': [0.6, 1],       # Spar plate diameter [m]
    'D6': [300, 700]         # PTO damping [kg/s]
}

# DOE Configuration
METHOD = 'Box-Behnken'
N_CENTER_POINTS = 6
SEED = 42

#%%============================================================================
# GENERATE DOE DESIGN
#==============================================================================

print("="*70)
print("DOE DESIGN GENERATOR - Box-Behnken")
print("="*70)

print(f"\n📋 Parameters defined:")
for param, range_vals in wec_parameters.items():
    print(f"   {param}: [{range_vals[0]:6.1f}, {range_vals[1]:6.1f}]")

print(f"\n⚙️  DOE Configuration:")
print(f"   Method: {METHOD}")
print(f"   Center points: {N_CENTER_POINTS}")
print(f"   Random seed: {SEED}")

# Generate Box-Behnken design
print(f"\n🧪 Generating {METHOD} design...")
doe_results = generate_doe_vectors(
    parameter_ranges=wec_parameters,
    method=METHOD,
    n_center_points=N_CENTER_POINTS,
    seed=SEED
)

# Extract design matrix as NumPy array
design_matrix = doe_results['design_matrix']  # [n_experiments × n_parameters]
n_experiments, n_parameters = design_matrix.shape

print(f"\n✅ Design generated successfully!")
print(f"   Experiments: {n_experiments}")
print(f"   Parameters: {n_parameters}")
print(f"   Design matrix shape: {design_matrix.shape}")

#%%============================================================================
# CREATE OUTPUT FOLDER
#==============================================================================

print(f"\n📁 Creating output folder...")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
print(f"   ✅ Folder created/verified: {OUTPUT_FOLDER}/")

#%%============================================================================
# SAVE DOE VALUES
#==============================================================================

print(f"\n💾 Saving DOE values...")

# Prepare data structure
doe_data = {
    'design_matrix': design_matrix,              # [n_experiments × n_parameters] NumPy array
    'parameter_names': doe_results['parameter_names'],
    'parameter_ranges': doe_results['parameter_ranges'],
    'n_experiments': n_experiments,
    'n_parameters': n_parameters,
    'method': METHOD,
    'n_center_points': N_CENTER_POINTS,
    'seed': SEED,
    'metadata': doe_results['metadata']
}

# Save as PKL (Python - easy to read)
pkl_path = os.path.join(OUTPUT_FOLDER, f"{OUTPUT_FILENAME}.pkl")
with open(pkl_path, 'wb') as f:
    pickle.dump(doe_data, f, protocol=pickle.HIGHEST_PROTOCOL)
print(f"   ✅ Saved: {pkl_path}")

# Save as MAT (MATLAB compatible)
try:
    from scipy.io import savemat
    
    mat_path = os.path.join(OUTPUT_FOLDER, f"{OUTPUT_FILENAME}.mat")
    
    # Prepare MATLAB-compatible structure
    mat_data = {
        'design_matrix': design_matrix,
        'parameter_names': np.array(doe_results['parameter_names'], dtype=object),
        'n_experiments': n_experiments,
        'n_parameters': n_parameters,
        'method': METHOD,
        'n_center_points': N_CENTER_POINTS,
        'seed': SEED
    }
    
    # Add parameter ranges
    for param, range_vals in doe_results['parameter_ranges'].items():
        mat_data[f'{param}_range'] = np.array(range_vals)
    
    savemat(mat_path, mat_data, do_compression=True)
    print(f"   ✅ Saved: {mat_path}")
    
except ImportError:
    print(f"   ⚠️  scipy not available - MAT file not saved")
except Exception as e:
    print(f"   ⚠️  Error saving MAT file: {e}")

#%%============================================================================
# VERIFICATION
#==============================================================================

print(f"\n🔍 Verification:")
print(f"   Design matrix type: {type(design_matrix)}")
print(f"   Design matrix dtype: {design_matrix.dtype}")
print(f"   Design matrix shape: {design_matrix.shape}")

print(f"\n📊 First 5 experiments:")
for i in range(min(5, n_experiments)):
    exp_vector = design_matrix[i, :]
    vector_str = ", ".join([f"{val:7.3f}" for val in exp_vector])
    print(f"   Exp {i+1:2d}: [{vector_str}]")

if n_experiments > 5:
    print(f"   ... ({n_experiments - 5} more experiments)")

#%%============================================================================
# USAGE INSTRUCTIONS
#==============================================================================

print(f"\n" + "="*70)
print("✅ DOE VALUES GENERATED SUCCESSFULLY")
print("="*70)

print(f"\n📖 How to use in Python:")
print(f"""
import pickle
import numpy as np

# Load DOE data
with open('{OUTPUT_FOLDER}/{OUTPUT_FILENAME}.pkl', 'rb') as f:
    doe_data = pickle.load(f)

# Access design matrix
design_matrix = doe_data['design_matrix']  # NumPy array [{n_experiments} × {n_parameters}]

# Access specific experiment
exp_1 = design_matrix[0, :]  # First experiment [D1, D2, D3, D4, D5, D6]

# Access specific parameter column
D1_values = design_matrix[:, 0]  # All D1 values

# Loop through experiments
for i, experiment in enumerate(design_matrix):
    D1, D2, D3, D4, D5, D6 = experiment
    print(f"Experiment {{i+1}}: D1={{D1:.2f}}, D2={{D2:.2f}}, ...")
""")

print(f"\n📖 How to use in MATLAB:")
print(f"""
% Load DOE data
load('{OUTPUT_FOLDER}/{OUTPUT_FILENAME}.mat')

% Access design matrix
design_matrix  % [{n_experiments} × {n_parameters}] matrix

% Access specific experiment
exp_1 = design_matrix(1, :);  % First row

% Access specific parameter
D1_values = design_matrix(:, 1);  % First column

% Loop through experiments
for i = 1:size(design_matrix, 1)
    D1 = design_matrix(i, 1);
    D2 = design_matrix(i, 2);
    % ... process experiment
end
""")

print(f"\n📁 Files created in '{OUTPUT_FOLDER}/':")
print(f"   - {OUTPUT_FILENAME}.pkl  (Python - Pickle format)")
print(f"   - {OUTPUT_FILENAME}.mat  (MATLAB format)")

print(f"\n🎯 Next steps:")
print(f"   1. Use these DOE values in subsequent analysis scripts")
print(f"   2. Generate geometries based on design_matrix")
print(f"   3. Run hydrodynamic analysis for each experiment")
print(f"   4. Collect results and build response surface")

print(f"\n" + "="*70)
