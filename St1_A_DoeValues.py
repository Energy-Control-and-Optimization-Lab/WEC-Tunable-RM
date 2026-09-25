"""
St1_B_DoeValues.py - Design of Experiments Generator (7 parameters)
Generates Box-Behnken design and saves DOE values for WEC parametric analysis.
Full 7-parameter study: float geometry (D1, D2, D3), spar geometry (D4, D5),
PTO damping (D6), and PTO stiffness (D7).

Author: Pablo Antonio Matamala Carvajal
Date: 2025-06-18
Updated: 2025-09-11 - Added D7 (PTO stiffness K_PTO, [0, 50000] N/m) as a
                      7th free DOE variable, acting on the float-spar
                      relative motion for resonance tuning. Same treatment
                      as D6 throughout the pipeline: not used in the BEM
                      step (St2), enters the impedance matrix in St3_B.

NOTE: This is the 7-parameter companion to St1_A_DoeValues.py (3-parameter,
      float-only DOE). Both studies run independently in separate folders.
      St1_A: 15 experiments, float only, spar and B_PTO fixed.
      St1_B: N experiments (Box-Behnken, 7 factors), all 7 variables free.
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
OUTPUT_FILENAME = "DOE_values"

# WEC Design Parameters — all 7 varying
wec_parameters = {
    'D1': [2.1, 2.8],      # Float diameter [m]
    'D2': [0.3, 0.5],      # Float external draft [m]
    'D3': [0.0, 0.20],      # Float internal draft offset [m]
    'D4': [3.9, 5.2],      # Spar draft [m]
    'D5': [3.0, 4.0],      # Spar plate diameter [m]
    'D6': [50000, 100000],  # PTO damping B_PTO [kg/s]
    'D7': [0, 50000],       # PTO stiffness K_PTO [N/m] (relative motion, resonance tuning)
}

# DOE Configuration
METHOD = 'Box-Behnken'
N_CENTER_POINTS = 3   # Deterministic BEM simulation → 3 centers sufficient
SEED = 42             # No effect on Box-Behnken (deterministic design);
                      # kept for consistency and future LHS comparison

#%%============================================================================
# GENERATE DOE DESIGN
#==============================================================================

print("="*70)
print("DOE DESIGN GENERATOR (7 parameters) - Box-Behnken")
print("="*70)

print(f"\n📋 Parameters defined:")
for param, range_vals in wec_parameters.items():
    print(f"   {param}: [{range_vals[0]:>10.2f}, {range_vals[1]:>10.2f}]")

print(f"\n⚙️  DOE Configuration:")
print(f"   Method:        {METHOD}")
print(f"   Center points: {N_CENTER_POINTS}")
print(f"   Random seed:   {SEED}")

# Generate Box-Behnken design
print(f"\n🧪 Generating {METHOD} design...")
doe_results = generate_doe_vectors(
    parameter_ranges=wec_parameters,
    method=METHOD,
    n_center_points=N_CENTER_POINTS,
    seed=SEED
)

design_matrix = doe_results['design_matrix']   # [n_experiments × 6]
n_experiments, n_parameters = design_matrix.shape

print(f"\n✅ Design generated successfully!")
print(f"   Experiments: {n_experiments}")
print(f"   Parameters:  {n_parameters}")
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

doe_data = {
    'design_matrix':    design_matrix,
    'parameter_names':  doe_results['parameter_names'],
    'parameter_ranges': doe_results['parameter_ranges'],
    'n_experiments':    n_experiments,
    'n_parameters':     n_parameters,
    'method':           METHOD,
    'n_center_points':  N_CENTER_POINTS,
    'seed':             SEED,
    'metadata':         doe_results['metadata']
}

# Save as PKL
pkl_path = os.path.join(OUTPUT_FOLDER, f"{OUTPUT_FILENAME}.pkl")
with open(pkl_path, 'wb') as f:
    pickle.dump(doe_data, f, protocol=pickle.HIGHEST_PROTOCOL)
print(f"   ✅ Saved: {pkl_path}")

# Save as MAT
try:
    from scipy.io import savemat

    mat_path = os.path.join(OUTPUT_FOLDER, f"{OUTPUT_FILENAME}.mat")
    mat_data = {
        'design_matrix':  design_matrix,
        'parameter_names': np.array(doe_results['parameter_names'], dtype=object),
        'n_experiments':  n_experiments,
        'n_parameters':   n_parameters,
        'method':         METHOD,
        'n_center_points': N_CENTER_POINTS,
        'seed':           SEED
    }
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
print(f"   Design matrix dtype:  {design_matrix.dtype}")
print(f"   Design matrix shape:  {design_matrix.shape}")

print(f"\n📊 First 5 experiment vectors [D1, D2, D3, D4, D5, D6, D7]:")
for i in range(min(5, n_experiments)):
    exp_vector = design_matrix[i, :]
    vector_str = ", ".join([f"{val:9.2f}" for val in exp_vector])
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

with open('{OUTPUT_FOLDER}/{OUTPUT_FILENAME}.pkl', 'rb') as f:
    doe_data = pickle.load(f)

design_matrix = doe_data['design_matrix']   # [{n_experiments} × {n_parameters}]

for i, experiment in enumerate(design_matrix):
    D1, D2, D3, D4, D5, D6, D7 = experiment
    print(f"Exp {{i+1}}: D1={{D1:.3f}}, D2={{D2:.3f}}, D3={{D3:.3f}}, "
          f"D4={{D4:.3f}}, D5={{D5:.3f}}, D6={{D6:.0f}}, D7={{D7:.0f}}")
""")

print(f"\n📖 How to use in MATLAB:")
print(f"""
load('{OUTPUT_FOLDER}/{OUTPUT_FILENAME}.mat')
% design_matrix: [{n_experiments} × {n_parameters}]  columns: D1 D2 D3 D4 D5 D6 D7
for i = 1:size(design_matrix,1)
    D1=design_matrix(i,1); D2=design_matrix(i,2); D3=design_matrix(i,3);
    D4=design_matrix(i,4); D5=design_matrix(i,5); D6=design_matrix(i,6);
    D7=design_matrix(i,7);
end
""")

print(f"\n📁 Files created in '{OUTPUT_FOLDER}/':")
print(f"   - {OUTPUT_FILENAME}.pkl  (Python)")
print(f"   - {OUTPUT_FILENAME}.mat  (MATLAB)")

print(f"\n🎯 Next steps:")
print(f"   1. Run St2_B_Hydro.py  → geometry generation + BEM for all {n_experiments} experiments")
print(f"   2. Run St3_B_RAO.py    → RAO calculation")
print(f"   3. Run St3_B_Power.py  → power absorption")
print(f"   4. Run St4_B_Results.py → build response vectors for metamodel")

print(f"\n" + "="*70)
