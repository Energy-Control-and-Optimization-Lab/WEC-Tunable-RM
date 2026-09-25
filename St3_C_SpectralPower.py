"""
St3_C_SpectralPower.py - Spectral Power Calculation
For each experiment, convolves P(ω) with each JONSWAP sea state spectrum
to obtain the power spectral density S_P(ω) and the mean power P_sea
for all 181 sea states (1 Mean_2024 reference + 180 climatological
monthly states, 12 months x 15 years).

Frequency grids:
  - BEM (HydCoeff.npz):  w from St2_A/B_Hydro  [rad/s]  ← master grid
  - Spectra (Spectra.pkl): own grid (typically coarser) ← interpolated to BEM grid

Formula:
  S_P(ω) = P(ω) · S(ω)          [W · m²·s/rad]
  P_sea   = ∫ S_P(ω) dω         [W]

Author: Pablo Antonio Matamala Carvajal
Date: 2025-06-18
Updated: 2026-09-22 - Spectra.pkl now holds 181 states instead of 13
                      (see St3_A_Spectra.py). The per-experiment
                      calculation loop is unchanged (generic over
                      n_states/labels), only the plotting section was
                      updated: instead of 1 combined plot with all
                      curves (illegible at 181 lines), each experiment
                      now gets 16 plots — 1 per climatological year
                      (12 monthly S_P(ω) curves overlaid) + 1 for the
                      Mean_2024 reference — same scheme as St3_A.

Input:  EcoData/St2_EcoBatch/DOE_Exp_XXX/hydroData/HydCoeff.npz  (w, for grid)
        EcoData/St3_Power/DOE_Exp_XXX/St3_B_Exp_XXX.npz          (P_omega)
        EcoData/St3_Spectra/Spectra.pkl                            (S_omega, labels)
Output: EcoData/St3_C_SpectralPower/DOE_Exp_XXX/St3_C_Exp_XXX.{npz,pkl,mat}
        EcoData/St3_C_SpectralPower/DOE_Exp_XXX/Plots/SP_Mean_2024.png
        EcoData/St3_C_SpectralPower/DOE_Exp_XXX/Plots/SP_y01.png ... y15.png
        (16 plots per experiment, same folder-per-DOE structure as before)
"""

import os
import sys
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.io import savemat

# NumPy compatibility: trapezoid was added in 2.0, trapz removed in 2.0
_trapz = getattr(np, 'trapezoid', None) or getattr(np, 'trapz')

#%%============================================================================
# CONFIGURATION
#==============================================================================

# --- Input: hydrodynamic data (for frequency grid) ---
ST2_FOLDER = "EcoData/St2_EcoBatch"

# --- Input: power curves ---
ST3_POWER_FOLDER = "EcoData/St3_Power"

# --- Input: JONSWAP spectra ---
SPECTRA_FILE = "EcoData/St3_Spectra/Spectra.pkl"

# --- Output folder ---
ST3C_FOLDER = "EcoData/St3_SpectralPower"

# --- Resume ---
START_FROM_EXPERIMENT = 1

#%%============================================================================
# SETUP — Load spectra (common to all experiments)
#==============================================================================

print("=" * 70)
print("SPECTRAL POWER CALCULATION")
print("=" * 70)
print(f"\n📂 Hydro grid from: {ST2_FOLDER}/")
print(f"📂 Power from:      {ST3_POWER_FOLDER}/")
print(f"📂 Spectra from:    {SPECTRA_FILE}")
print(f"📁 Writing to:      {ST3C_FOLDER}/")

# Load JONSWAP spectra
if not os.path.exists(SPECTRA_FILE):
    print(f"❌ Spectra file not found: {SPECTRA_FILE}")
    print(f"   Run St3_A_Spectra.py first.")
    sys.exit(1)

with open(SPECTRA_FILE, 'rb') as f:
    spectra_data = pickle.load(f)

S_omega_all   = spectra_data['S_omega']       # [181, n_freq_spectra]
w_spectra     = spectra_data['frequencies']   # [n_freq_spectra]
labels        = spectra_data['labels']        # list of 181 strings
n_states      = spectra_data['n_states']      # 181

print(f"\n✅ Spectra loaded: {n_states} sea states")
print(f"   Spectra grid: {w_spectra[0]:.2f}–{w_spectra[-1]:.2f} rad/s  "
      f"({len(w_spectra)} points)")
print(f"   Sea states: {labels}")

os.makedirs(ST3C_FOLDER, exist_ok=True)

# Detect experiments
exp_folders = sorted([
    d for d in os.listdir(ST2_FOLDER)
    if d.startswith("DOE_Exp_") and os.path.isdir(os.path.join(ST2_FOLDER, d))
]) if os.path.exists(ST2_FOLDER) else []

n_experiments = len(exp_folders)
print(f"\n📊 Experiments found: {n_experiments}")

if n_experiments == 0:
    print(f"❌ No experiment folders found in {ST2_FOLDER}")
    sys.exit(1)

print("=" * 70)

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
    hydro_npz   = os.path.join(ST2_FOLDER, exp_folder, "hydroData", "HydCoeff.npz")
    output_dir  = os.path.join(ST3C_FOLDER, exp_folder)
    plots_dir   = os.path.join(output_dir, "Plots")
    os.makedirs(plots_dir, exist_ok=True)
    stem = os.path.join(output_dir, f"St3_C_Exp_{case_counter:03d}")

    # ------------------------------------------------------------------
    # Load frequency grid from HydCoeff (master grid)
    # ------------------------------------------------------------------
    if not os.path.exists(hydro_npz):
        print(f"❌ HydCoeff.npz not found: {hydro_npz}")
        failed += 1
        continue

    try:
        hydro_data = np.load(hydro_npz, allow_pickle=True)
        w = hydro_data['w']   # [n_freq] — master frequency grid
        n_freq = len(w)
        print(f"✅ Frequency grid: {w[0]:.2f}–{w[-1]:.2f} rad/s  ({n_freq} points)")
    except Exception as e:
        print(f"❌ Error loading HydCoeff.npz: {e}")
        failed += 1
        continue

    # ------------------------------------------------------------------
    # Load P(ω) — find the NPZ automatically in the experiment power folder
    # ------------------------------------------------------------------
    power_exp_dir = os.path.join(ST3_POWER_FOLDER, exp_folder)
    power_npz_files = sorted([
        f for f in os.listdir(power_exp_dir) if f.endswith('.npz')
    ]) if os.path.exists(power_exp_dir) else []

    if not power_npz_files:
        print(f"❌ No NPZ found in: {power_exp_dir}")
        print(f"   Run St3_B_Power.py (or St3_A_Power.py) first.")
        failed += 1
        continue

    power_npz = os.path.join(power_exp_dir, power_npz_files[0])

    try:
        power_data = np.load(power_npz, allow_pickle=True)
        P_omega = power_data['P_omega']   # [n_freq]
        B_PTO   = float(power_data['B_PTO'])
        K_PTO   = float(power_data['K_PTO'])
        print(f"✅ P(ω) loaded  (B_PTO={B_PTO:.0f} kg/s, K_PTO={K_PTO:.0f} N/m)")
    except Exception as e:
        print(f"❌ Error loading Power npz: {e}")
        failed += 1
        continue

    # ------------------------------------------------------------------
    # Interpolate each spectrum to BEM frequency grid
    # ------------------------------------------------------------------
    # S_omega_all is defined on w_spectra; P_omega is on w (BEM grid).
    # Interpolate S to w — use zero outside the spectra range.
    S_on_w = np.zeros((n_states, n_freq))   # [13, n_freq]

    for k in range(n_states):
        interp_fn = interp1d(w_spectra, S_omega_all[k], kind='linear',
                             bounds_error=False, fill_value=0.0)
        S_on_w[k] = interp_fn(w)

    # ------------------------------------------------------------------
    # Compute S_P(ω) and P_sea for each sea state
    # ------------------------------------------------------------------
    S_P_all = np.zeros((n_states, n_freq))   # [13, n_freq]  power spectral density
    P_sea   = np.zeros(n_states)             # [13]          mean power per sea state

    for k in range(n_states):
        S_P_all[k] = 2.0 * P_omega * S_on_w[k]           # [W · m²·s/rad]
        P_sea[k]   = _trapz(S_P_all[k], w)       # [W]

    # Annual mean power (index 0 = Mean_2024)
    P_avg_annual = P_sea[0]

    print(f"\n   Sea state power results:")
    for k, lbl in enumerate(labels):
        print(f"   {lbl:12s}  P_sea = {P_sea[k]:10.4f} W")
    print(f"\n   ⭐ P_avg_annual (Mean_2024) = {P_avg_annual:.4f} W")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    results = {
        # Power spectral density curves [13 x n_freq]
        'S_P_omega':      S_P_all,
        # Mean power per sea state [13]
        'P_sea':          P_sea,
        # Annual mean power (scalar)
        'P_avg_annual':   P_avg_annual,
        # Metadata
        'frequencies':    w,
        'B_PTO':          B_PTO,
        'K_PTO':          K_PTO,
        'experiment_id':  case_counter,
        'n_states':       n_states,
    }

    # NPZ
    try:
        np.savez_compressed(stem + ".npz", **results)
        print(f"\n   ✅ NPZ: St3_C_Exp_{case_counter:03d}.npz")
    except Exception as e:
        print(f"   ⚠️  NPZ error: {e}")

    # PKL — also store labels for convenience
    try:
        results_pkl = dict(results)
        results_pkl['labels'] = labels
        with open(stem + ".pkl", 'wb') as f:
            pickle.dump(results_pkl, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"   ✅ PKL: St3_C_Exp_{case_counter:03d}.pkl")
    except Exception as e:
        print(f"   ⚠️  PKL error: {e}")

    # MAT — labels as individual strings
    try:
        mat_data = dict(results)
        for k, lbl in enumerate(labels):
            mat_data[f'label_{k:02d}'] = lbl
        savemat(stem + ".mat", mat_data, do_compression=True)
        print(f"   ✅ MAT: St3_C_Exp_{case_counter:03d}.mat")
    except Exception as e:
        print(f"   ⚠️  MAT error: {e}")

    # ------------------------------------------------------------------
    # Plots — 16 images per experiment: 1 per climatological year (12
    # monthly S_P(ω) curves overlaid) + 1 standalone for Mean_2024.
    # Plotting all 181 states in one figure (or one file per state) would
    # be illegible / produce too many files — same scheme as St3_A_Spectra.
    # ------------------------------------------------------------------
    month_colors = plt.cm.viridis(np.linspace(0, 1, 12))
    month_order  = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

    idx_mean = labels.index('Mean_2024')

    # --- Mean_2024 (standalone) ---
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(w, S_P_all[idx_mean], color='#d1242f', linewidth=2.2,
            label=f"Mean 2024  P={P_sea[idx_mean]:.3f} W")
    ax.set_xlabel('ω [rad/s]', fontsize=12)
    ax.set_ylabel('S_P(ω)  [W·m²·s/rad]', fontsize=12)
    ax.set_title(f'Power Spectral Density — Exp {case_counter:03d} — Mean 2024 (AWAC reference)\n'
                 f'B_PTO={B_PTO:.0f} kg/s, K_PTO={K_PTO:.0f} N/m',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([w[0], w[-1]])
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, 'SP_Mean_2024.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # --- One figure per climatological year (y01..y15), 12 monthly curves each ---
    for yi in range(1, 16):
        year_prefix = f"y{yi:02d}_"
        year_label  = 2009 + yi   # y01 -> 2010, ..., y15 -> 2024

        fig, ax = plt.subplots(figsize=(10, 5.5))
        for mi, mon in enumerate(month_order):
            lbl = f"{year_prefix}{mon}"
            if lbl not in labels:
                continue
            k = labels.index(lbl)
            ax.plot(w, S_P_all[k], color=month_colors[mi], linewidth=1.4,
                    label=f"{mon}  P={P_sea[k]:.3f} W")

        ax.set_xlabel('ω [rad/s]', fontsize=12)
        ax.set_ylabel('S_P(ω)  [W·m²·s/rad]', fontsize=12)
        ax.set_title(f'Power Spectral Density — Exp {case_counter:03d} — Year {yi:02d} ({year_label})\n'
                     f'B_PTO={B_PTO:.0f} kg/s, K_PTO={K_PTO:.0f} N/m  |  climatological monthly states',
                     fontsize=12, fontweight='bold')
        ax.legend(fontsize=7.5, loc='upper right', ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([w[0], w[-1]])
        ax.set_ylim(bottom=0)

        plt.tight_layout()
        fig.savefig(os.path.join(plots_dir, f'SP_y{yi:02d}.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)

    print(f"   ✅ Plots saved: 16 images (SP_Mean_2024.png + SP_y01..y15.png)")

    successful += 1

#%%============================================================================
# FINAL SUMMARY
#==============================================================================

print("\n" + "=" * 70)
print("St3_C_SpectralPower COMPLETED")
print("=" * 70)
print(f"Successful: {successful}  |  Failed: {failed}")
print(f"Results in: {ST3C_FOLDER}/")

print(f"\n📁 Folder status:")
for exp_folder in exp_folders:
    case_counter = int(exp_folder.split("_")[-1])
    mat_file = os.path.join(ST3C_FOLDER, exp_folder,
                            f"St3_C_Exp_{case_counter:03d}.mat")
    if os.path.exists(mat_file):
        print(f"   ✅ {exp_folder}")
    else:
        print(f"   ❌ {exp_folder}  (output missing)")

print(f"\n📦 Variables saved per experiment:")
print(f"   S_P_omega     → [181 x n_freq]  power spectral density [W·m²·s/rad]")
print(f"   P_sea         → [181]            mean power per sea state [W]")
print(f"   P_avg_annual  → scalar           mean annual power (Mean_2024) [W]")
print(f"   frequencies   → [n_freq]         BEM frequency grid [rad/s]")
print(f"   B_PTO         → scalar           PTO damping used [kg/s]")
print(f"   K_PTO         → scalar           PTO stiffness used [N/m]")

print(f"\n🎯 Next step: St4 → build response vectors for metamodel")
print(f"   Key scalar for RSM: P_avg_annual")
print(f"   Key vector for RSM: P_sea (181 sea states: Mean_2024 + 180 climatological)")
print("\n" + "=" * 70)
