"""
St4_A_ResultsVector.py - Extract Response Vectors for MetaModel Generation
Consolidates power results from DOE experiments into response vectors

Author: Pablo Antonio Matamala Carvajal
Date: 2025-01-21
Updated: 2025-11-30 - FIXED: Use power data for phase extraction (physical consistency)
Updated: 2025-03-09 - FIXED: Alignment bug in Step 2 — phase_matrix_list and
                      mass_matrix_list now use None placeholders on failure,
                      matching P_matrix_list indexing. Previous version produced
                      silently misaligned mass/phase vectors when any experiment
                      failed in Step 2, corrupting P_efficiency for all subsequent
                      experiments.

Description:
- Reads P_omega from each DOE experiment (EcoData/St3_Power/)
- Reads phase from POWER files for consistency (FIXED - was using RAO files)
- Extracts P_avg (average power in specified frequency range)
- Extracts P at specific frequencies (user-defined)
- Extracts phase at specific frequencies (CORRECTED - now from power data with PTO)
- Calculates P_efficiency = P_avg / total_mass
- Includes design matrix (D1-D6) from DOE
- Saves all vectors in EcoData/St4_ResultVector/VectorValues.*

CRITICAL FIX 1: Phase vectors now extracted from power data (WITH PTO) instead of
               RAO data (WITHOUT PTO) to ensure physical consistency between
               power and phase vectors in metamodel generation.

CRITICAL FIX 2: Alignment bug — phase_matrix_list and mass_matrix_list now append
               None on failure so that valid_indices filtering produces correctly
               aligned arrays. Without this fix, a single failed experiment in
               Step 2 shifts all subsequent mass/phase values by one position,
               silently corrupting P_efficiency for the entire dataset.
"""

import os
import sys
import numpy as np
import pickle

#%%============================================================================
# CONFIGURATION
#==============================================================================

# Input paths
POWER_FOLDER = "EcoData/St3_Power"
DOE_FILE     = "EcoData/St1_DOEvalues/DOE_values.pkl"
RAO_FOLDER   = "EcoData/St3_RAO"

# Output path
OUTPUT_FOLDER   = "EcoData/St4_ResultVector"
OUTPUT_FILENAME = "VectorValues"

# Frequency range for P_avg [rad/s]
F_MIN_AVG = 1.6
F_MAX_AVG = 4.0

# Specific frequencies for P and phase extraction [rad/s]
SPECIFIC_FREQUENCIES = [1.6, 2.0, 2.4, 2.8, 3.2, 3.6, 4.0]

# Wave parameters
WAVE_HEIGHT    = 2.0   # H [m]
WAVE_AMPLITUDE = 1.0   # A [m]

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

    n_experiments    = doe_data['n_experiments']
    design_matrix    = doe_data['design_matrix']   # [N × 6]
    parameter_names  = doe_data['parameter_names']
    parameter_ranges = doe_data['parameter_ranges']

    print(f"✅ DOE data loaded successfully!")
    print(f"   Total experiments: {n_experiments}")
    print(f"   Parameters: {parameter_names}")
    print(f"   Design matrix shape: {design_matrix.shape}")

except FileNotFoundError:
    print(f"❌ ERROR: DOE file not found: {DOE_FILE}")
    sys.exit(1)
except Exception as e:
    print(f"❌ ERROR loading DOE file: {e}")
    sys.exit(1)

#%%============================================================================
# STEP 1 — EXTRACT P(ω) FROM POWER FILES
#==============================================================================

print(f"\n{'='*70}")
print("STEP 1: EXTRACT P(ω) FROM POWER FILES")
print(f"{'='*70}")

# All three lists are kept in strict 1-to-1 correspondence.
# Failures → append None so that valid_indices filtering stays aligned.
P_matrix_list    = []   # None on failure
phase_matrix_list = []  # None on failure  ← FIX: was populated only on success
mass_matrix_list       = []  # None on failure  ← FIX: was populated only on success
mass_float_list        = []  # None on failure — float heave mass separately
mass_spar_list         = []  # None on failure — spar heave mass separately
experiment_ids    = []
successful_experiments = []
failed_experiments     = []
frequencies = None

for exp_id in range(1, n_experiments + 1):
    power_file = os.path.join(POWER_FOLDER, f"DOE_Exp_{exp_id:03d}_Power.pkl")

    if not os.path.exists(power_file):
        print(f"⚠️ Exp {exp_id:03d}: Power file not found")
        failed_experiments.append(exp_id)
        P_matrix_list.append(None)
        phase_matrix_list.append(None)
        mass_matrix_list.append(None)
        mass_float_list.append(None)
        mass_spar_list.append(None)
        experiment_ids.append(None)
        continue

    try:
        with open(power_file, 'rb') as f:
            power_data = pickle.load(f)

        P_omega = power_data['P_omega']

        if frequencies is None:
            frequencies = power_data['frequencies']
            n_freq = len(frequencies)
            print(f"✅ Frequency array: {n_freq} pts "
                  f"[{frequencies[0]:.2f}, {frequencies[-1]:.2f}] rad/s")

        # Check enhanced phase data is present
        if 'phase_relative_deg' not in power_data:
            print(f"❌ Exp {exp_id:03d}: Missing phase_relative_deg — "
                  f"run enhanced St3_B_Power.py first")
            failed_experiments.append(exp_id)
            P_matrix_list.append(None)
            phase_matrix_list.append(None)
            mass_matrix_list.append(None)
            mass_float_list.append(None)
            mass_spar_list.append(None)
            experiment_ids.append(None)
            continue

        phase_relative_deg = power_data['phase_relative_deg']

        # Load mass from RAO file
        rao_file = os.path.join(RAO_FOLDER, f"DOE_Exp_{exp_id:03d}_RAO.pkl")
        if os.path.exists(rao_file):
            with open(rao_file, 'rb') as f:
                rao_data = pickle.load(f)
            M_heave = rao_data['M_heave']   # [2, 2]
        else:
            print(f"⚠️ Exp {exp_id:03d}: RAO file not found — using default mass")
            M_heave = np.array([[1000, 0], [0, 500]])

        total_mass  = M_heave[0, 0] + M_heave[1, 1]
        float_mass  = M_heave[0, 0]   # float heave mass [kg]
        spar_mass   = M_heave[1, 1]   # spar  heave mass [kg]

        # ── All lists get a value at the same position ─────────────────────
        P_matrix_list.append(P_omega)
        phase_matrix_list.append(phase_relative_deg)
        mass_matrix_list.append(total_mass)
        mass_float_list.append(float_mass)
        mass_spar_list.append(spar_mass)
        experiment_ids.append(exp_id)
        successful_experiments.append(exp_id)

        if exp_id % 10 == 0:
            print(f"   Processed: Exp {exp_id:03d}/{n_experiments}")

    except Exception as e:
        print(f"❌ Exp {exp_id:03d}: Error — {e}")
        failed_experiments.append(exp_id)
        P_matrix_list.append(None)
        phase_matrix_list.append(None)
        mass_matrix_list.append(None)
        mass_float_list.append(None)
        mass_spar_list.append(None)
        experiment_ids.append(None)
        continue

# ── Filter: keep only positions where ALL three lists have valid data ──────
valid_indices = [
    i for i in range(len(P_matrix_list))
    if P_matrix_list[i] is not None
    and phase_matrix_list[i] is not None
    and mass_matrix_list[i] is not None
]

P_matrix          = np.array([P_matrix_list[i]     for i in valid_indices])   # [n × n_freq]
phase_matrix      = np.array([phase_matrix_list[i]  for i in valid_indices])  # [n × n_freq]
mass_array        = np.array([mass_matrix_list[i]   for i in valid_indices])  # [n]
mass_float_array  = np.array([mass_float_list[i]    for i in valid_indices])  # [n]
mass_spar_array   = np.array([mass_spar_list[i]     for i in valid_indices])  # [n]
experiment_ids_clean = np.array([experiment_ids[i] for i in valid_indices])

# Rebuild successful list from valid indices
successful_experiments_clean = [experiment_ids[i] for i in valid_indices]
n_successful = len(successful_experiments_clean)

# Design matrix aligned to valid experiments
design_matrix_filtered = design_matrix[np.array(successful_experiments_clean) - 1, :]

print(f"\n✅ Extraction completed:")
print(f"   Successful: {n_successful} experiments")
print(f"   Failed:     {len(failed_experiments)} experiments")
print(f"   P_matrix shape:     {P_matrix.shape}")
print(f"   phase_matrix shape: {phase_matrix.shape}")
print(f"   mass_array shape:   {mass_array.shape}")
print(f"   ✅ All three arrays are aligned (FIX applied)")

if failed_experiments:
    print(f"\n⚠️ Failed experiments: {failed_experiments}")

#%%============================================================================
# STEP 2 — CALCULATE P_AVG
#==============================================================================

print(f"\n{'='*70}")
print(f"STEP 2: CALCULATE P_avg  [{F_MIN_AVG}, {F_MAX_AVG}] rad/s")
print(f"{'='*70}")

mask_avg         = (frequencies >= F_MIN_AVG) & (frequencies <= F_MAX_AVG)
n_freq_in_range  = np.sum(mask_avg)
print(f"   Frequencies in range: {n_freq_in_range}")

P_avg = np.mean(P_matrix[:, mask_avg], axis=1)

print(f"✅ P_avg: min={np.min(P_avg):.2f} W, max={np.max(P_avg):.2f} W, "
      f"mean={np.mean(P_avg):.2f} W")

#%%============================================================================
# STEP 3 — CALCULATE P_EFFICIENCY
#==============================================================================

print(f"\n{'='*70}")
print(f"STEP 3: CALCULATE P_EFFICIENCY")
print(f"{'='*70}")

P_efficiency = P_avg / mass_array

print(f"✅ P_efficiency: min={np.min(P_efficiency):.4f}, "
      f"max={np.max(P_efficiency):.4f}, mean={np.mean(P_efficiency):.4f} W/kg")
print(f"   Mass range: [{np.min(mass_array):.1f}, {np.max(mass_array):.1f}] kg")
print(f"   Float mass range: [{np.min(mass_float_array):.1f}, {np.max(mass_float_array):.1f}] kg")
print(f"   Spar  mass range: [{np.min(mass_spar_array):.1f}, {np.max(mass_spar_array):.1f}] kg")

#%%============================================================================
# STEP 4 — EXTRACT P AND PHASE AT SPECIFIC FREQUENCIES
#==============================================================================

print(f"\n{'='*70}")
print(f"STEP 4: EXTRACT P AND PHASE AT SPECIFIC FREQUENCIES")
print(f"{'='*70}")

response_names  = ['P_avg', 'P_efficiency']
P_at_specific   = {}
phase_at_specific = {}

for omega_target in SPECIFIC_FREQUENCIES:
    idx         = np.argmin(np.abs(frequencies - omega_target))
    freq_actual = frequencies[idx]

    # Power
    P_at_omega  = P_matrix[:, idx]
    p_name      = f"P_at_{omega_target:.2f}".replace('.', '_')
    P_at_specific[p_name] = P_at_omega
    response_names.append(p_name)

    # Phase
    ph_at_omega = phase_matrix[:, idx]
    ph_name     = f"phase_at_{omega_target:.2f}".replace('.', '_')
    phase_at_specific[ph_name] = ph_at_omega
    response_names.append(ph_name)

    print(f"   ω={omega_target:.2f}: P=[{np.min(P_at_omega):.2f}, "
          f"{np.max(P_at_omega):.2f}] W  |  "
          f"phase=[{np.min(ph_at_omega):.1f}°, {np.max(ph_at_omega):.1f}°]")

print(f"\n✅ Extracted {len(SPECIFIC_FREQUENCIES)} frequency points "
      f"({len(SPECIFIC_FREQUENCIES)*2} response vectors)")

#%%============================================================================
# STEP 5 — SAVE RESULTS
#==============================================================================

print(f"\n{'='*70}")
print(f"STEP 5: SAVE RESULTS")
print(f"{'='*70}")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

results = {
    'P_avg':                 P_avg,
    'P_efficiency':          P_efficiency,
    'mass_total':            mass_array,        # float + spar heave mass [kg]
    'mass_float':            mass_float_array,  # float heave mass only   [kg]
    'mass_spar':             mass_spar_array,   # spar  heave mass only   [kg]
    'design_matrix':         design_matrix_filtered,
    'frequencies_full':      frequencies,
    'specific_frequencies':  np.array(SPECIFIC_FREQUENCIES),
    'f_range_avg':           np.array([F_MIN_AVG, F_MAX_AVG]),
    'parameter_names':       parameter_names,
    'parameter_ranges':      parameter_ranges,
    'response_names':        response_names,
    'experiment_ids':        experiment_ids_clean,
    'n_experiments':         n_successful,
    'n_parameters':          len(parameter_names),
    'failed_experiments':    failed_experiments,
    'wave_height':           WAVE_HEIGHT,
    'wave_amplitude':        WAVE_AMPLITUDE,
    'metadata': {
        'description':   'Response vectors for quadratic metamodel (alignment-fixed)',
        'doe_method':    doe_data.get('method', 'Box-Behnken'),
        'n_center_points': doe_data.get('n_center_points', 'Unknown'),
        'seed':          doe_data.get('seed', 'Unknown'),
        'fix_1':         'Phase from power data (WITH PTO) — physical consistency',
        'fix_2':         'None placeholders in all lists — alignment guarantee',
        'units': {
            'P_vectors':   'W',
            'phase_vectors': 'degrees',
            'P_efficiency': 'W/kg',
            'mass_total':   'kg',
            'mass_float':   'kg',
            'mass_spar':    'kg',
            'frequencies': 'rad/s'
        }
    }
}

for var_name, vals in P_at_specific.items():
    results[var_name] = vals
for var_name, vals in phase_at_specific.items():
    results[var_name] = vals

# PKL
try:
    pkl_path = os.path.join(OUTPUT_FOLDER, f"{OUTPUT_FILENAME}.pkl")
    with open(pkl_path, 'wb') as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"✅ PKL saved: {OUTPUT_FILENAME}.pkl")
except Exception as e:
    print(f"⚠️ Error saving PKL: {e}")

# NPZ
try:
    npz_path = os.path.join(OUTPUT_FOLDER, f"{OUTPUT_FILENAME}.npz")
    npz_data = {k: v for k, v in results.items()
                if isinstance(v, (np.ndarray, int, float, list))}
    npz_data['parameter_names'] = np.array(parameter_names, dtype=object)
    npz_data['response_names']  = np.array(response_names,  dtype=object)
    np.savez_compressed(npz_path, **npz_data)
    print(f"✅ NPZ saved: {OUTPUT_FILENAME}.npz")
except Exception as e:
    print(f"⚠️ Error saving NPZ: {e}")

# MAT
try:
    from scipy.io import savemat
    mat_path = os.path.join(OUTPUT_FOLDER, f"{OUTPUT_FILENAME}.mat")
    mat_data = {k: v for k, v in results.items()
                if k != 'metadata' and not isinstance(v, dict)}
    mat_data['parameter_names'] = np.array(parameter_names, dtype=object)
    mat_data['response_names']  = np.array(response_names,  dtype=object)
    savemat(mat_path, mat_data, do_compression=True)
    print(f"✅ MAT saved: {OUTPUT_FILENAME}.mat")
except Exception as e:
    print(f"⚠️ Error saving MAT: {e}")

#%%============================================================================
# FINAL SUMMARY
#==============================================================================

print(f"\n{'='*70}")
print("🎉 RESPONSE VECTORS EXTRACTION COMPLETED")
print(f"{'='*70}")

print(f"\n📁 Results saved in: {OUTPUT_FOLDER}/")
print(f"   - {OUTPUT_FILENAME}.pkl")
print(f"   - {OUTPUT_FILENAME}.npz")
print(f"   - {OUTPUT_FILENAME}.mat")

print(f"\n📊 Response Vectors [{n_successful} experiments]:")
print(f"   1. P_avg          — average power [{F_MIN_AVG}, {F_MAX_AVG}] rad/s [W]")
print(f"   2. P_efficiency   — P_avg / mass [W/kg]")
for i, omega in enumerate(SPECIFIC_FREQUENCIES, start=3):
    print(f"   {i:2d}. P_at_{omega:.2f}    — power at ω={omega} rad/s [W]"
          .replace('.', '_', 1))
for i, omega in enumerate(SPECIFIC_FREQUENCIES,
                           start=3 + len(SPECIFIC_FREQUENCIES)):
    print(f"   {i:2d}. phase_at_{omega:.2f} — phase at ω={omega} rad/s [°] (PTO-consistent)"
          .replace('.', '_', 1))

print(f"\n🔧 FIXES APPLIED:")
print(f"   ✅ FIX 1 — Phase source: St3_B_Power.py (WITH PTO, physically consistent)")
print(f"   ✅ FIX 2 — Alignment:    None placeholders ensure P / phase / mass are "
      f"always at the same list index")

print(f"\n🎯 Next step: Run St4_B_MetaModel.py")
print("="*70)
