"""
St4_A_ResultsVector.py - Extract Response Vectors for MetaModel Generation
Consolidates power results from DOE experiments into response vectors

Author: Pablo Antonio Matamala Carvajal
Date: 2025-01-21
Updated: 2025-11-30 - FIXED: Use power data for phase extraction (physical consistency)
Description:
- Reads P_omega from each DOE experiment (EcoData/Power/)
- Reads phase from POWER files for consistency (FIXED - was using RAO files)
- Extracts P_avg (average power in specified frequency range)
- Extracts P at specific frequencies (user-defined)
- Extracts phase at specific frequencies (CORRECTED - now from power data with PTO)
- Calculates P_efficiency = P_avg / total_mass
- Includes design matrix (D1-D6) from DOE
- Saves all vectors in EcoData/MetaModel/VectorValues.*

CRITICAL FIX: Phase vectors now extracted from power data (WITH PTO) instead of 
              RAO data (WITHOUT PTO) to ensure physical consistency between 
              power and phase vectors in metamodel generation.
"""

import os
import sys
import numpy as np
import pickle

#%%============================================================================
# CONFIGURATION
#==============================================================================

# Input paths
POWER_FOLDER = "EcoData/Power"
DOE_FILE = "EcoData/DOE_values.pkl"

# Output path
OUTPUT_FOLDER = "EcoData/MetaModel"
OUTPUT_FILENAME = "VectorValues"

# Frequency range for P_avg [rad/s]
F_MIN_AVG = 1.2
F_MAX_AVG = 3.2

# Specific frequencies for P and phase extraction [rad/s] - MODIFIABLE
SPECIFIC_FREQUENCIES = [1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25]

# Wave parameters
WAVE_HEIGHT = 2.0  # H [m] - current P_omega is for this height
WAVE_AMPLITUDE = 1.0  # A [m] - corresponding amplitude

#%%============================================================================
# LOAD DOE INFORMATION
#==============================================================================

print("="*70)
print("EXTRACT RESPONSE VECTORS FOR METAMODEL (FIXED VERSION)")
print("="*70)

print(f"\n📂 Loading DOE information from: {DOE_FILE}")

try:
    with open(DOE_FILE, 'rb') as f:
        doe_data = pickle.load(f)
    
    n_experiments = doe_data['n_experiments']
    design_matrix = doe_data['design_matrix']  # [66 × 6] with D1-D6
    parameter_names = doe_data['parameter_names']
    parameter_ranges = doe_data['parameter_ranges']
    
    print(f"✅ DOE data loaded successfully!")
    print(f"   Total experiments: {n_experiments}")
    print(f"   Parameters: {parameter_names}")
    print(f"   Design matrix shape: {design_matrix.shape}")
    
except FileNotFoundError:
    print(f"❌ ERROR: DOE file not found: {DOE_FILE}")
    print(f"   Please run St1_DoeValues.py first")
    sys.exit(1)
except Exception as e:
    print(f"❌ ERROR loading DOE file: {e}")
    sys.exit(1)

#%%============================================================================
# EXTRACT P_OMEGA FROM ALL EXPERIMENTS
#==============================================================================

print(f"\n{'='*70}")
print("STEP 1: EXTRACT P(ω) FROM POWER FILES")
print(f"{'='*70}")

print(f"🚀 CRITICAL FIX: Using enhanced power files with phase data")
print(f"   ✅ Physical consistency: P_at_* and phase_at_* from SAME system (with PTO)")
print(f"   ❌ Previous error: phase_at_* from RAO (without PTO) ≠ P_at_* from Power (with PTO)")

# Storage for P_omega matrix
P_matrix_list = []
frequencies = None
experiment_ids = []
successful_experiments = []
failed_experiments = []

for exp_id in range(1, n_experiments + 1):
    power_file = os.path.join(POWER_FOLDER, f"DOE_Exp_{exp_id:03d}_Power.pkl")
    
    if not os.path.exists(power_file):
        print(f"⚠️ Exp {exp_id:03d}: Power file not found")
        failed_experiments.append(exp_id)
        continue
    
    try:
        # Load power data
        with open(power_file, 'rb') as f:
            power_data = pickle.load(f)
        
        # Extract P_omega
        P_omega = power_data['P_omega']  # [n_frequencies]
        
        # Get frequencies (only once from first experiment)
        if frequencies is None:
            frequencies = power_data['frequencies']
            n_freq = len(frequencies)
            print(f"✅ Frequency array loaded: {n_freq} points [{frequencies[0]:.2f}, {frequencies[-1]:.2f}] rad/s")
        
        # Append to matrix
        P_matrix_list.append(P_omega)
        experiment_ids.append(exp_id)
        successful_experiments.append(exp_id)
        
        if exp_id % 10 == 0:
            print(f"   Processing: Exp {exp_id:03d}/{n_experiments}")
        
    except Exception as e:
        print(f"❌ Exp {exp_id:03d}: Error loading - {e}")
        failed_experiments.append(exp_id)
        continue

# Convert to numpy array
P_matrix = np.array(P_matrix_list)  # [n_successful × n_frequencies]
experiment_ids = np.array(experiment_ids)
n_successful = len(successful_experiments)

print(f"\n✅ Extraction completed:")
print(f"   Successful: {n_successful} experiments")
print(f"   Failed: {len(failed_experiments)} experiments")
print(f"   P_matrix shape: {P_matrix.shape}")

if len(failed_experiments) > 0:
    print(f"\n⚠️ Failed experiments: {failed_experiments}")
    print(f"⚠️ Continuing with {n_successful} successful experiments")

# Filter design matrix for successful experiments only
design_matrix_filtered = design_matrix[np.array(successful_experiments) - 1, :]

#%%============================================================================
# EXTRACT PHASE AND MASS FROM POWER FILES (FIXED)
#==============================================================================

print(f"\n{'='*70}")
print("STEP 2: EXTRACT PHASE FROM POWER FILES (FIXED)")
print(f"{'='*70}")

print(f"🔧 CRITICAL FIX ACTIVE:")
print(f"   ✅ Source: St3_B_Power.py (WITH PTO + viscous damping)")
print(f"   ✅ Phase: Calculated from RAO_with_PTO_relative")
print(f"   ✅ Consistency: Same damping configuration as P_omega")

# Storage for phase matrix and mass matrix
phase_matrix_list = []
mass_matrix_list = []

print(f"📂 Loading enhanced power data for phase extraction...")

for i, exp_id in enumerate(successful_experiments):
    power_file = os.path.join(POWER_FOLDER, f"DOE_Exp_{exp_id:03d}_Power.pkl")
    
    if not os.path.exists(power_file):
        print(f"⚠️ Exp {exp_id:03d}: Power file not found - excluding from analysis")
        # Remove this experiment from all lists
        P_matrix_list[i] = None
        experiment_ids[i] = None
        continue
    
    try:
        # Load POWER data (FIXED - was loading RAO data)
        with open(power_file, 'rb') as f:
            power_data = pickle.load(f)
        
        # Check if enhanced features are available
        if 'phase_relative_deg' not in power_data:
            print(f"❌ Exp {exp_id:03d}: Enhanced power file missing phase data")
            print(f"   Please run enhanced St3_B_Power.py first")
            P_matrix_list[i] = None
            experiment_ids[i] = None
            continue
        
        # Extract phase vector (FIXED - now from power data with PTO)
        phase_relative_deg = power_data['phase_relative_deg']  # [n_frequencies] in degrees
        
        # For mass, we need to load from original RAO data (still needed for efficiency)
        rao_file = os.path.join("EcoData/RAO", f"DOE_Exp_{exp_id:03d}_RAO.pkl")
        if os.path.exists(rao_file):
            with open(rao_file, 'rb') as f:
                rao_data = pickle.load(f)
            M_heave = rao_data['M_heave']  # [2, 2]
        else:
            # Alternative: estimate mass from power data if available
            print(f"⚠️ Exp {exp_id:03d}: RAO file not found for mass, using default")
            M_heave = np.array([[1000, 0], [0, 500]])  # Default masses [kg]
        
        total_mass = M_heave[0, 0] + M_heave[1, 1]  # Float + Spar heave mass
        
        # Append to matrices
        phase_matrix_list.append(phase_relative_deg)
        mass_matrix_list.append(total_mass)
        
        if exp_id % 10 == 0:
            print(f"   Processing Power: Exp {exp_id:03d}/{n_experiments} ✅ Enhanced")
        
    except Exception as e:
        print(f"❌ Exp {exp_id:03d}: Error loading power data - {e}")
        # Remove this experiment from all lists
        P_matrix_list[i] = None
        experiment_ids[i] = None
        continue

# Clean up None entries
valid_indices = [i for i, p in enumerate(P_matrix_list) if p is not None]
P_matrix = np.array([P_matrix_list[i] for i in valid_indices])
phase_matrix = np.array([phase_matrix_list[i] for i in valid_indices])
mass_array = np.array([mass_matrix_list[i] for i in valid_indices])
experiment_ids = np.array([experiment_ids[i] for i in valid_indices])
successful_experiments = [successful_experiments[i] for i in valid_indices]
n_successful = len(successful_experiments)

# Update design matrix for final successful experiments
design_matrix_filtered = design_matrix[np.array(successful_experiments) - 1, :]

print(f"\n✅ Phase extraction completed (FIXED):")
print(f"   Final successful experiments: {n_successful}")
print(f"   Phase matrix shape: {phase_matrix.shape}")
print(f"   Mass array shape: {mass_array.shape}")
print(f"   🎯 Expected fix: phase_at_1_50 MAPE 411.5% → <5%")

#%%============================================================================
# CALCULATE P_AVG (AVERAGE IN FREQUENCY RANGE)
#==============================================================================

print(f"\n{'='*70}")
print(f"STEP 3: CALCULATE P_avg (AVERAGE POWER)")
print(f"{'='*70}")

print(f"   Frequency range: [{F_MIN_AVG}, {F_MAX_AVG}] rad/s")

# Create mask for frequency range
mask_avg = (frequencies >= F_MIN_AVG) & (frequencies <= F_MAX_AVG)
n_freq_in_range = np.sum(mask_avg)

print(f"   Frequencies in range: {n_freq_in_range}")

# Calculate P_avg for each experiment
P_avg = np.mean(P_matrix[:, mask_avg], axis=1)  # [n_successful]

print(f"✅ P_avg calculated:")
print(f"   Shape: {P_avg.shape}")
print(f"   Min: {np.min(P_avg):.2f} W")
print(f"   Max: {np.max(P_avg):.2f} W")
print(f"   Mean: {np.mean(P_avg):.2f} W")

#%%============================================================================
# CALCULATE P_EFFICIENCY
#==============================================================================

print(f"\n{'='*70}")
print(f"STEP 4: CALCULATE P_EFFICIENCY")
print(f"{'='*70}")

# Calculate efficiency: P_efficiency = P_avg / total_mass [W/kg]
P_efficiency = P_avg / mass_array

print(f"✅ P_efficiency calculated:")
print(f"   Shape: {P_efficiency.shape}")
print(f"   Min: {np.min(P_efficiency):.4f} W/kg")
print(f"   Max: {np.max(P_efficiency):.4f} W/kg")
print(f"   Mean: {np.mean(P_efficiency):.4f} W/kg")
print(f"   Mass range: [{np.min(mass_array):.0f}, {np.max(mass_array):.0f}] kg")

#%%============================================================================
# EXTRACT P AT SPECIFIC FREQUENCIES
#==============================================================================

print(f"\n{'='*70}")
print(f"STEP 5: EXTRACT P AT SPECIFIC FREQUENCIES")
print(f"{'='*70}")

print(f"   Target frequencies: {SPECIFIC_FREQUENCIES}")

# Dictionary to store P at each specific frequency
P_at_specific = {}
response_names = ['P_avg', 'P_efficiency']  # Start with P_avg and efficiency

for omega_target in SPECIFIC_FREQUENCIES:
    # Find nearest frequency in the array
    idx = np.argmin(np.abs(frequencies - omega_target))
    freq_actual = frequencies[idx]
    
    # Extract P at this frequency for all experiments
    P_at_omega = P_matrix[:, idx]  # [n_successful]
    
    # Create variable name (replace decimal point with underscore)
    var_name = f"P_at_{omega_target:.2f}".replace('.', '_')
    P_at_specific[var_name] = P_at_omega
    response_names.append(var_name)
    
    print(f"   {var_name}: ω_target={omega_target:.2f} → ω_actual={freq_actual:.2f} rad/s, "
          f"P_range=[{np.min(P_at_omega):.2f}, {np.max(P_at_omega):.2f}] W")

print(f"\n✅ Extracted {len(SPECIFIC_FREQUENCIES)} specific frequency points for power")

#%%============================================================================
# EXTRACT PHASE AT SPECIFIC FREQUENCIES (FIXED)
#==============================================================================

print(f"\n{'='*70}")
print(f"STEP 6: EXTRACT PHASE AT SPECIFIC FREQUENCIES (FIXED)")
print(f"{'='*70}")

print(f"   Target frequencies: {SPECIFIC_FREQUENCIES}")
print(f"   🔧 Source: St3_B_Power.py (WITH PTO) - PHYSICALLY CONSISTENT")

# Dictionary to store phase at each specific frequency
phase_at_specific = {}

for omega_target in SPECIFIC_FREQUENCIES:
    # Find nearest frequency in the array
    idx = np.argmin(np.abs(frequencies - omega_target))
    freq_actual = frequencies[idx]
    
    # Extract phase at this frequency for all experiments
    phase_at_omega = phase_matrix[:, idx]  # [n_successful]
    
    # Create variable name (replace decimal point with underscore)
    var_name = f"phase_at_{omega_target:.2f}".replace('.', '_')
    phase_at_specific[var_name] = phase_at_omega
    response_names.append(var_name)
    
    print(f"   {var_name}: ω_target={omega_target:.2f} → ω_actual={freq_actual:.2f} rad/s, "
          f"phase_range=[{np.min(phase_at_omega):.1f}°, {np.max(phase_at_omega):.1f}°]")

print(f"\n✅ Extracted {len(SPECIFIC_FREQUENCIES)} specific frequency points for phase (FIXED)")
print(f"🎯 Critical improvement expected at phase_at_1_50 (ω=1.5 rad/s)")

#%%============================================================================
# PREPARE OUTPUT DATA
#==============================================================================

print(f"\n{'='*70}")
print(f"STEP 7: PREPARE OUTPUT DATA")
print(f"{'='*70}")

# Prepare results dictionary
results = {
    # Response vectors [n_successful × 1]
    'P_avg': P_avg,
    'P_efficiency': P_efficiency,
    
    # Design matrix [n_successful × 6]
    'design_matrix': design_matrix_filtered,
    
    # Frequency information
    'frequencies_full': frequencies,
    'specific_frequencies': np.array(SPECIFIC_FREQUENCIES),
    'f_range_avg': np.array([F_MIN_AVG, F_MAX_AVG]),
    
    # Parameter information
    'parameter_names': parameter_names,
    'parameter_ranges': parameter_ranges,
    
    # Response information
    'response_names': response_names,
    
    # Experiment information
    'experiment_ids': experiment_ids,
    'n_experiments': n_successful,
    'n_parameters': len(parameter_names),
    'failed_experiments': failed_experiments,
    
    # Wave parameters
    'wave_height': WAVE_HEIGHT,
    'wave_amplitude': WAVE_AMPLITUDE,
    
    # Metadata
    'metadata': {
        'description': 'Response vectors for quadratic metamodel generation (FIXED VERSION)',
        'doe_method': doe_data.get('method', 'Box-Behnken'),
        'n_center_points': doe_data.get('n_center_points', 'Unknown'),
        'seed': doe_data.get('seed', 'Unknown'),
        'fix_description': 'Phase vectors now extracted from power data (WITH PTO) for physical consistency',
        'phase_source': 'St3_B_Power.py (enhanced with PTO damping)',
        'power_source': 'St3_B_Power.py (same source for consistency)',
        'expected_improvement': 'phase_at_1_50 MAPE: 411.5% → <5%',
        'units': {
            'P_vectors': 'W (Watts)',
            'phase_vectors': 'degrees',
            'P_efficiency': 'W/kg',
            'frequencies': 'rad/s',
            'wave_height': 'm',
            'wave_amplitude': 'm'
        }
    }
}

# Add P at specific frequencies to results
for var_name, P_values in P_at_specific.items():
    results[var_name] = P_values

# Add phase at specific frequencies to results (FIXED)
for var_name, phase_values in phase_at_specific.items():
    results[var_name] = phase_values

print(f"✅ Data prepared (FIXED):")
print(f"   Response vectors: {len(response_names)}")
print(f"   Design matrix: {design_matrix_filtered.shape}")
print(f"   🔧 Fix applied: Consistent source for power and phase vectors")

#%%============================================================================
# SAVE RESULTS
#==============================================================================

print(f"\n{'='*70}")
print(f"STEP 8: SAVE RESULTS")
print(f"{'='*70}")

# Create output folder
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Save as NPZ
try:
    npz_path = os.path.join(OUTPUT_FOLDER, f"{OUTPUT_FILENAME}.npz")
    
    # Prepare data for NPZ (numpy arrays only)
    npz_data = {
        'P_avg': P_avg,
        'P_efficiency': P_efficiency,
        'design_matrix': design_matrix_filtered,
        'frequencies_full': frequencies,
        'specific_frequencies': np.array(SPECIFIC_FREQUENCIES),
        'f_range_avg': np.array([F_MIN_AVG, F_MAX_AVG]),
        'parameter_names': np.array(parameter_names, dtype=object),
        'response_names': np.array(response_names, dtype=object),
        'experiment_ids': experiment_ids,
        'n_experiments': n_successful,
        'n_parameters': len(parameter_names),
        'failed_experiments': np.array(failed_experiments),
        'wave_height': WAVE_HEIGHT,
        'wave_amplitude': WAVE_AMPLITUDE
    }
    
    # Add P at specific frequencies
    for var_name, P_values in P_at_specific.items():
        npz_data[var_name] = P_values
    
    # Add phase at specific frequencies (FIXED)
    for var_name, phase_values in phase_at_specific.items():
        npz_data[var_name] = phase_values
    
    np.savez_compressed(npz_path, **npz_data)
    print(f"✅ NPZ saved: {OUTPUT_FILENAME}.npz")
except Exception as e:
    print(f"⚠️ Error saving NPZ: {e}")

# Save as PKL (includes all data with metadata)
try:
    pkl_path = os.path.join(OUTPUT_FOLDER, f"{OUTPUT_FILENAME}.pkl")
    with open(pkl_path, 'wb') as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"✅ PKL saved: {OUTPUT_FILENAME}.pkl")
except Exception as e:
    print(f"⚠️ Error saving PKL: {e}")

# Save as MAT
try:
    from scipy.io import savemat
    mat_path = os.path.join(OUTPUT_FOLDER, f"{OUTPUT_FILENAME}.mat")
    
    # Prepare data for MAT
    mat_data = {
        'P_avg': P_avg,
        'P_efficiency': P_efficiency,
        'design_matrix': design_matrix_filtered,
        'frequencies_full': frequencies,
        'specific_frequencies': np.array(SPECIFIC_FREQUENCIES),
        'f_range_avg': np.array([F_MIN_AVG, F_MAX_AVG]),
        'parameter_names': np.array(parameter_names, dtype=object),
        'response_names': np.array(response_names, dtype=object),
        'experiment_ids': experiment_ids,
        'n_experiments': n_successful,
        'n_parameters': len(parameter_names),
        'failed_experiments': np.array(failed_experiments),
        'wave_height': WAVE_HEIGHT,
        'wave_amplitude': WAVE_AMPLITUDE
    }
    
    # Add P at specific frequencies
    for var_name, P_values in P_at_specific.items():
        mat_data[var_name] = P_values
    
    # Add phase at specific frequencies (FIXED)
    for var_name, phase_values in phase_at_specific.items():
        mat_data[var_name] = phase_values
    
    savemat(mat_path, mat_data, do_compression=True)
    print(f"✅ MAT saved: {OUTPUT_FILENAME}.mat")
except Exception as e:
    print(f"⚠️ Error saving MAT: {e}")

#%%============================================================================
# SUMMARY
#==============================================================================

print(f"\n{'='*70}")
print("🎉 RESPONSE VECTORS EXTRACTION COMPLETED (FIXED)")
print(f"{'='*70}")

print(f"\n📁 Results saved in: {OUTPUT_FOLDER}/")
print(f"   - {OUTPUT_FILENAME}.npz")
print(f"   - {OUTPUT_FILENAME}.pkl (includes metadata)")
print(f"   - {OUTPUT_FILENAME}.mat")

print(f"\n📊 Response Vectors Generated:")
print(f"   1. P_avg [{n_successful}×1] - Average power in [{F_MIN_AVG}, {F_MAX_AVG}] rad/s")
print(f"   2. P_efficiency [{n_successful}×1] - Power per unit mass [W/kg]")
for i, omega in enumerate(SPECIFIC_FREQUENCIES, start=3):
    var_name = f"P_at_{omega:.2f}".replace('.', '_')
    print(f"   {i}. {var_name} [{n_successful}×1] - Power at ω={omega} rad/s")
for i, omega in enumerate(SPECIFIC_FREQUENCIES, start=3+len(SPECIFIC_FREQUENCIES)):
    var_name = f"phase_at_{omega:.2f}".replace('.', '_')
    print(f"   {i}. {var_name} [{n_successful}×1] - Phase at ω={omega} rad/s (FIXED)")

print(f"\n📋 Design Matrix:")
print(f"   Shape: {design_matrix_filtered.shape} [{n_successful} experiments × {len(parameter_names)} parameters]")
print(f"   Parameters: {parameter_names}")

print(f"\n🔧 CRITICAL FIX SUMMARY:")
print(f"   ❌ Before: phase_at_* from St3_A_RAO.py (WITHOUT PTO)")
print(f"   ✅ After:  phase_at_* from St3_B_Power.py (WITH PTO)")
print(f"   🎯 Result: Physical consistency between power and phase vectors")
print(f"   📈 Expected: phase_at_1_50 MAPE 411.5% → <5%")

print(f"\n🎯 Ready for Metamodel Generation:")
print(f"   - Total response vectors: {len(response_names)}")
print(f"   - Each vector: [{n_successful}×1]")
print(f"   - Design matrix: [{n_successful}×{len(parameter_names)}]")
print(f"   - Physical consistency: ✅ FIXED")

print(f"\n💡 Next Steps:")
print(f"   1. Run St4_B_MetaModel.py to generate quadratic metamodels")
print(f"   2. Verify phase_at_1_50 metamodel improvement")
print(f"   3. Compare MAPE before/after fix")
print(f"   4. Use physically consistent response vectors for optimization")

print(f"\n💡 Usage Example (Python):")
print(f"   import numpy as np")
print(f"   data = np.load('EcoData/MetaModel/VectorValues.npz')")
print(f"   P_avg = data['P_avg']  # Power response vector")
print(f"   phase_at_1_50 = data['phase_at_1_50']  # FIXED phase vector")
print(f"   X = data['design_matrix']  # Input parameters D1-D6")
print(f"   # Both power and phase now from SAME physical system (with PTO)")

print(f"\n🎉 Physical consistency FIX applied successfully!")
print(f"{'='*70}")
