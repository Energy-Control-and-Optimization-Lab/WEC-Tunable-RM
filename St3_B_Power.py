"""
St3_B_Power.py - Power Calculation for 7-parameter DOE
Reads hydrodynamic coefficients from St2_B_EcoBatch and computes:
  1. Heave RAOs with PTO damping + stiffness (float, spar, relative)
  2. Power absorption curve P(ω)
  3. Peak power and corresponding frequency
  4. Float and spar masses

B_PTO (D6) and K_PTO (D7) are read per-experiment from the DOE design
matrix — both vary across experiments.

No wave spectrum convolution — all outputs are functions of frequency only.
P_avg (spectral) is computed in a separate script.

Author: Pablo Antonio Matamala Carvajal
Date: 2025-06-18
Updated: 2025-09-11 - Added D7 (PTO stiffness K_PTO, [0, 50000] N/m).
                      K_PTO_matrix (same relative-motion topology as
                      B_PTO_matrix) is added to C_heave in the impedance.

Input:  EcoData/St2_B_EcoBatch/DOE_Exp_XXX/hydroData/HydCoeff.{npz,pkl}
        EcoData/St2_B_EcoBatch/DOE_Exp_XXX/experiment_summary.json
Output: EcoData/St3_B_Power/DOE_Exp_XXX/St3_B_Exp_XXX.{npz,pkl,mat}
        EcoData/St3_B_Power/DOE_Exp_XXX/Plots/
"""

import os
import sys
import json
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.io import savemat

#%%============================================================================
# CONFIGURATION
#==============================================================================

# --- Input folder (St2 output) ---
ST2_FOLDER = "EcoData/St2_EcoBatch"

# --- Output folder ---
ST3_FOLDER = "EcoData/St3_Power"

# --- DOE file (source of B_PTO = D6 per experiment) ---
DOE_FILE   = "EcoData/St1_DOEvalues/DOE_values.pkl"

# --- No viscous damping (handled separately if needed) ---
B_VISC_FLOAT = 0.0   # [kg/s]
B_VISC_SPAR  = 0.0   # [kg/s]

# --- Resume ---
START_FROM_EXPERIMENT = 1

# --- Heave DOF indices in the 12x12 matrix (0-based) ---
IDX_FLOAT_HEAVE = 2   # DOF 3 → Float heave
IDX_SPAR_HEAVE  = 8   # DOF 9 → Spar heave
HEAVE_IDX = [IDX_FLOAT_HEAVE, IDX_SPAR_HEAVE]

#%%============================================================================
# SETUP
#==============================================================================

print("=" * 70)
print("POWER CALCULATION - DOE-B (7 parameters, B_PTO=D6, K_PTO=D7 per experiment)")
print("=" * 70)
print(f"\n📂 Reading from: {ST2_FOLDER}/")
print(f"📁 Writing to:   {ST3_FOLDER}/")
print(f"🔧 B_PTO:        read from DOE design matrix (D6, varies per experiment)")
print(f"🔧 K_PTO:        read from DOE design matrix (D7, varies per experiment)")

os.makedirs(ST3_FOLDER, exist_ok=True)

# Detect number of experiments from folder structure
exp_folders = sorted([
    d for d in os.listdir(ST2_FOLDER)
    if d.startswith("DOE_Exp_") and os.path.isdir(os.path.join(ST2_FOLDER, d))
]) if os.path.exists(ST2_FOLDER) else []

n_experiments = len(exp_folders)
print(f"📊 Experiments found: {n_experiments}")

if n_experiments == 0:
    print(f"❌ No experiment folders found in {ST2_FOLDER}")
    print(f"   Run St2_B_Hydro.py first.")
    sys.exit(1)

print("=" * 70)

# ------------------------------------------------------------------
# Load DOE design matrix — extract D6 (B_PTO) and D7 (K_PTO) per experiment
# ------------------------------------------------------------------
print(f"\n📂 Loading DOE design matrix from: {DOE_FILE}")
try:
    with open(DOE_FILE, 'rb') as f:
        doe_data = pickle.load(f)
    doe_design_matrix = doe_data['design_matrix']   # [n_exp x 7]
    doe_param_names   = doe_data['parameter_names']  # ['D1',...,'D6','D7']
    idx_D6 = doe_param_names.index('D6')
    idx_D7 = doe_param_names.index('D7')
    print(f"✅ DOE loaded — {len(doe_design_matrix)} experiments, "
          f"D6 at column {idx_D6}, D7 at column {idx_D7}")
except Exception as e:
    print(f"❌ Error loading DOE file: {e}")
    sys.exit(1)

#%%============================================================================
# EXPERIMENT LOOP
#==============================================================================

successful = 0
failed     = 0

for exp_folder in exp_folders:
    case_counter = int(exp_folder.split("_")[-1])

    if case_counter < START_FROM_EXPERIMENT:
        continue

    print(f"\n{'=' * 70}")
    print(f"  Experiment {case_counter}/{n_experiments}  —  {exp_folder}")
    print(f"{'=' * 70}")

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    hydro_npz    = os.path.join(ST2_FOLDER, exp_folder, "hydroData", "HydCoeff.npz")
    hydro_pkl    = os.path.join(ST2_FOLDER, exp_folder, "hydroData", "HydCoeff.pkl")
    summary_path = os.path.join(ST2_FOLDER, exp_folder, "experiment_summary.json")
    output_dir   = os.path.join(ST3_FOLDER, exp_folder)
    plots_dir    = os.path.join(output_dir, "Plots")
    os.makedirs(plots_dir, exist_ok=True)

    stem = os.path.join(output_dir, f"St3_B_Exp_{case_counter:03d}")

    # ------------------------------------------------------------------
    # Read B_PTO (D6) and K_PTO (D7) from DOE design matrix
    # ------------------------------------------------------------------
    try:
        B_PTO = float(doe_design_matrix[case_counter - 1, idx_D6])
        K_PTO = float(doe_design_matrix[case_counter - 1, idx_D7])
        print(f"   B_PTO (D6) = {B_PTO:.0f} kg/s  (from DOE design matrix, exp {case_counter})")
        print(f"   K_PTO (D7) = {K_PTO:.0f} N/m   (from DOE design matrix, exp {case_counter})")
    except Exception as e:
        print(f"❌ Error reading D6/D7 from DOE matrix: {e}")
        failed += 1
        continue

    # ------------------------------------------------------------------
    # Load hydrodynamic coefficients
    # ------------------------------------------------------------------
    if not os.path.exists(hydro_npz):
        print(f"❌ HydCoeff.npz not found: {hydro_npz}")
        failed += 1
        continue

    try:
        data    = np.load(hydro_npz, allow_pickle=True)
        A_full  = data['A']    # [n_freq, 12, 12]
        B_full  = data['B']    # [n_freq, 12, 12]
        Fe_full = data['Fe']   # [12, n_freq] complex
        w       = data['w']    # [n_freq]
        n_freq  = len(w)

        with open(hydro_pkl, 'rb') as f:
            pkl_data = pickle.load(f)
        M_full = np.array(pkl_data['M'])   # [12, 12]
        C_full = np.array(pkl_data['C'])   # [12, 12]

        print(f"✅ Loaded HydCoeff  ({n_freq} frequencies, "
              f"{w[0]:.2f}–{w[-1]:.2f} rad/s)")
    except Exception as e:
        print(f"❌ Error loading HydCoeff: {e}")
        failed += 1
        continue

    # ------------------------------------------------------------------
    # Extract 2×2 heave submatrices
    # ------------------------------------------------------------------
    ix = HEAVE_IDX

    A_heave  = A_full[:, ix, :][:, :, ix]      # [n_freq, 2, 2]
    B_heave  = B_full[:, ix, :][:, :, ix]      # [n_freq, 2, 2]
    M_heave  = M_full[np.ix_(ix, ix)]          # [2, 2]
    C_heave  = C_full[np.ix_(ix, ix)]          # [2, 2]
    Fe_heave = Fe_full[ix, :].T                # [n_freq, 2] complex

    M_float = M_heave[0, 0]
    M_spar  = M_heave[1, 1]

    print(f"   M_float = {M_float:.4f} kg")
    print(f"   M_spar  = {M_spar:.4f} kg")

    # ------------------------------------------------------------------
    # Build total damping and stiffness matrices
    # ------------------------------------------------------------------
    B_PTO_matrix = np.array([[ B_PTO, -B_PTO],
                              [-B_PTO,  B_PTO]])

    K_PTO_matrix = np.array([[ K_PTO, -K_PTO],
                              [-K_PTO,  K_PTO]])

    B_viscous = np.array([[B_VISC_FLOAT,      0      ],
                          [     0,       B_VISC_SPAR]])

    # ------------------------------------------------------------------
    # RAOs with PTO + Power calculation
    # ------------------------------------------------------------------
    RAO_float    = np.zeros(n_freq, dtype=complex)
    RAO_spar     = np.zeros(n_freq, dtype=complex)
    RAO_relative = np.zeros(n_freq, dtype=complex)
    P_omega      = np.zeros(n_freq)

    for i, omega in enumerate(w):
        A       = A_heave[i]
        B_hydro = B_heave[i]
        Fe      = Fe_heave[i]

        B_total = B_hydro + B_PTO_matrix + B_viscous
        C_total = C_heave + K_PTO_matrix

        # Impedance matrix: Z = -ω²(M+A) + iωB_total + C_total
        Z = -omega**2 * (M_heave + A) + 1j * omega * B_total + C_total

        try:
            X = np.linalg.solve(Z, Fe)   # unit wave amplitude ζ_a = 1

            RAO_float[i]    = X[0]
            RAO_spar[i]     = X[1]
            RAO_relative[i] = X[0] - X[1]

            # P(ω) = ½ · B_PTO · ω² · |ζ_rel|²
            v_rel = omega * (X[0] - X[1])
            P_omega[i] = 0.5 * B_PTO * np.abs(v_rel)**2

        except np.linalg.LinAlgError:
            print(f"   ⚠️  Singular matrix at ω={omega:.3f} rad/s → outputs=0")

    # Absolute RAOs
    RAO_float_abs    = np.abs(RAO_float)
    RAO_spar_abs     = np.abs(RAO_spar)
    RAO_relative_abs = np.abs(RAO_relative)

    # Peak power
    idx_peak     = np.argmax(P_omega)
    P_peak       = P_omega[idx_peak]
    omega_P_peak = w[idx_peak]

    print(f"   P_peak = {P_peak:.2f} W  at ω = {omega_P_peak:.3f} rad/s")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    results = {
        # Scalars
        'P_peak':           P_peak,
        'omega_P_peak':     omega_P_peak,
        'M_float':          M_float,
        'M_spar':           M_spar,
        # Curves
        'P_omega':          P_omega,
        'RAO_float_abs':    RAO_float_abs,
        'RAO_spar_abs':     RAO_spar_abs,
        'RAO_relative_abs': RAO_relative_abs,
        # Metadata
        'frequencies':      w,
        'B_PTO':            B_PTO,
        'K_PTO':            K_PTO,
        'experiment_id':    case_counter,
    }

    # NPZ
    try:
        np.savez_compressed(stem + ".npz", **results)
        print(f"   ✅ NPZ: St3_B_Exp_{case_counter:03d}.npz")
    except Exception as e:
        print(f"   ⚠️  NPZ error: {e}")

    # PKL
    try:
        with open(stem + ".pkl", 'wb') as f:
            pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"   ✅ PKL: St3_B_Exp_{case_counter:03d}.pkl")
    except Exception as e:
        print(f"   ⚠️  PKL error: {e}")

    # MAT
    try:
        savemat(stem + ".mat", results, do_compression=True)
        print(f"   ✅ MAT: St3_B_Exp_{case_counter:03d}.mat")
    except Exception as e:
        print(f"   ⚠️  MAT error: {e}")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    # Combined RAO + Power figure (2 subplots)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
    fig.suptitle(f'Exp {case_counter:03d} — B_PTO = {B_PTO:.0f} kg/s, K_PTO = {K_PTO:.0f} N/m',
                 fontsize=13, fontweight='bold')

    # Top: RAOs
    ax1.plot(w, RAO_float_abs,    'b-', linewidth=2, label='Float')
    ax1.plot(w, RAO_spar_abs,     'r-', linewidth=2, label='Spar')
    ax1.plot(w, RAO_relative_abs, 'g-', linewidth=2, label='Relative')
    ax1.set_ylabel('|RAO| [m/m]', fontsize=11)
    ax1.set_title('Heave RAOs with PTO', fontsize=11)
    ax1.legend(fontsize=10); ax1.grid(True, alpha=0.3)

    # Bottom: Power
    ax2.plot(w, P_omega, 'b-', linewidth=2, label='P(ω)')
    ax2.plot(omega_P_peak, P_peak, 'ro', markersize=8,
             label=f'Peak: {P_peak:.2f} W @ ω={omega_P_peak:.3f} rad/s')
    ax2.set_xlabel('ω [rad/s]', fontsize=11)
    ax2.set_ylabel('P(ω) [W]', fontsize=11)
    ax2.set_title('Power Absorption', fontsize=11)
    ax2.legend(fontsize=10); ax2.grid(True, alpha=0.3)

    ax1.set_xlim([w[0], w[-1]])
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, 'RAO_Power.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"   ✅ Plot saved: RAO_Power.png")
    successful += 1

#%%============================================================================
# FINAL SUMMARY
#==============================================================================

print("\n" + "=" * 70)
print("St3_B_Power COMPLETED")
print("=" * 70)
print(f"Successful: {successful}  |  Failed: {failed}")
print(f"B_PTO:      varied per experiment (D6 from DOE, 50000–100000 kg/s)")
print(f"K_PTO:      varied per experiment (D7 from DOE, 0–50000 N/m)")
print(f"Results in: {ST3_FOLDER}/")

print(f"\n📁 Folder status:")
for exp_folder in exp_folders:
    case_counter = int(exp_folder.split("_")[-1])
    mat_file = os.path.join(ST3_FOLDER, exp_folder,
                            f"St3_B_Exp_{case_counter:03d}.mat")
    if os.path.exists(mat_file):
        print(f"   ✅ {exp_folder}")
    else:
        print(f"   ❌ {exp_folder}  (output missing)")

print(f"\n📦 Variables saved per experiment:")
print(f"   Scalars : P_peak, omega_P_peak, M_float, M_spar, B_PTO, K_PTO")
print(f"   Curves  : P_omega, RAO_float_abs, RAO_spar_abs, RAO_relative_abs")
print(f"   Metadata: frequencies, experiment_id")

print(f"\n🎯 Next step: St4_B_Results.py → build response vectors for metamodel")
print("\n" + "=" * 70)
