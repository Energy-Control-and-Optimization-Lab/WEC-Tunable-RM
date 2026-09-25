"""
St4_A_ResultsVector.py - Response Vector Builder for Metamodel (RSM/Surrogate)
Joins the DOE design matrix (D1..D7) with:
  - P_avg_annual, P_sea (181 sea states: Mean_2024 + 180 climatological)
    from St3_SpectralPower (St3_C)
  - M_float, M_spar, B_PTO, K_PTO          from St3_Power         (St3_B)
into a single aggregated dataset ready for St4_B_MetaModel.py.

Also computes:
  - M_total = M_float + M_spar
  - PMR     = P_avg_annual / (M_total / 1000)   [W/t]  (Power-to-Mass Ratio)

PTO efficiency (ETA_PTO) is applied ONCE here to P_avg_annual and P_sea.
Do NOT re-apply it downstream (St4_B, St5, etc.) — it is already baked
into every power value saved by this script.

Author: Pablo Antonio Matamala Carvajal
Date: 2025-06-18
Updated: 2025-09-11 - Added K_PTO (D7) pass-through from St3_Power,
                      alongside B_PTO. Does not affect mass or PMR.

Input:  EcoData/St1_DOEvalues/DOE_values.pkl                    (design_matrix, D1..D7)
        EcoData/St3_SpectralPower/DOE_Exp_XXX/*.pkl              (P_avg_annual, P_sea, labels)
        EcoData/St3_Power/DOE_Exp_XXX/*.pkl                      (M_float, M_spar, B_PTO, K_PTO)
Output: EcoData/St4_ResultVector/VectorValues.{npz,pkl,mat}
"""

import os
import sys
import pickle
import numpy as np
from scipy.io import savemat

#%%============================================================================
# CONFIGURATION
#==============================================================================

# --- Input: DOE design matrix ---
DOE_FILE = "EcoData/St1_DOEvalues/DOE_values.pkl"

# --- Input: spectral power results (P_avg_annual, P_sea) ---
SPECTRAL_POWER_FOLDER = "EcoData/St3_SpectralPower"

# --- Input: power / mass results (M_float, M_spar, B_PTO) ---
POWER_FOLDER = "EcoData/St3_Power"

# --- Output folder ---
# NOTE: folder name is "St4_ResultVector" (no 's') to match what
# St4_B_MetaModel.py expects — do not rename without updating both scripts.
OUTPUT_FOLDER   = "EcoData/St4_ResultVector"
OUTPUT_FILENAME = "VectorValues"

# --- PTO efficiency factor — applied ONCE, here ---
ETA_PTO = 0.50   # 50% — multiplies P_avg_annual and P_sea

#%%============================================================================
# SETUP
#==============================================================================

print("=" * 70)
print("ST4_A - RESPONSE VECTOR BUILDER FOR METAMODEL")
print("=" * 70)
print(f"\n📂 DOE design matrix from:   {DOE_FILE}")
print(f"📂 Spectral power from:      {SPECTRAL_POWER_FOLDER}/")
print(f"📂 Power / mass from:        {POWER_FOLDER}/")
print(f"📁 Writing to:               {OUTPUT_FOLDER}/")
print(f"⚙️  PTO efficiency (η):       {ETA_PTO:.0%}")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ------------------------------------------------------------------
# Load DOE design matrix
# ------------------------------------------------------------------
if not os.path.exists(DOE_FILE):
    print(f"❌ DOE file not found: {DOE_FILE}")
    print(f"   Run the DOE generation step first (Eco_DOE.py driver).")
    sys.exit(1)

with open(DOE_FILE, 'rb') as f:
    doe_data = pickle.load(f)

design_matrix_full = np.asarray(doe_data['design_matrix'])   # [n_doe, n_params]
parameter_names     = list(doe_data['parameter_names'])       # ['D1', ..., 'D6']
parameter_ranges    = doe_data.get('parameter_ranges', None)
n_doe                = design_matrix_full.shape[0]
n_params             = design_matrix_full.shape[1]

print(f"\n✅ DOE loaded: {n_doe} experiments × {n_params} parameters")
print(f"   Parameters: {parameter_names}")
print("=" * 70)

#%%============================================================================
# EXPERIMENT LOOP - join design vector with response vectors
#==============================================================================

experiment_ids     = []
P_avg_annual_list  = []
P_sea_list         = []
M_float_list       = []
M_spar_list        = []
B_PTO_list         = []
K_PTO_list         = []
successful         = []
failed             = []
sea_labels         = None   # filled from the first successfully-read experiment

for exp_id in range(1, n_doe + 1):

    exp_name = f"DOE_Exp_{exp_id:03d}"
    sp_exp_dir = os.path.join(SPECTRAL_POWER_FOLDER, exp_name)
    pw_exp_dir = os.path.join(POWER_FOLDER,          exp_name)

    # Auto-detect PKL files (name-independent)
    sp_pkl_files = sorted([f for f in os.listdir(sp_exp_dir) if f.endswith('.pkl')]
                          ) if os.path.exists(sp_exp_dir) else []
    pw_pkl_files = sorted([f for f in os.listdir(pw_exp_dir) if f.endswith('.pkl')]
                          ) if os.path.exists(pw_exp_dir) else []

    if not sp_pkl_files:
        print(f"⚠️  Exp {exp_id:03d}: SpectralPower PKL not found — skipping")
        failed.append(exp_id)
        continue

    if not pw_pkl_files:
        print(f"⚠️  Exp {exp_id:03d}: Power PKL not found — skipping")
        failed.append(exp_id)
        continue

    try:
        # --- Load P_avg_annual, P_sea from St3_SpectralPower ---
        with open(os.path.join(sp_exp_dir, sp_pkl_files[0]), 'rb') as f:
            sp_data = pickle.load(f)
        P_avg_annual = float(sp_data['P_avg_annual'])
        P_sea        = np.asarray(sp_data['P_sea'])     # [181]
        labels       = sp_data['labels']                 # list of 181 strings

        if sea_labels is None:
            sea_labels = labels
        elif labels != sea_labels:
            print(f"   ⚠️  Exp {exp_id:03d}: sea-state labels differ from previous "
                  f"experiments — check consistency of St3_Spectra runs")

        # --- Load M_float, M_spar, B_PTO, K_PTO from St3_Power ---
        with open(os.path.join(pw_exp_dir, pw_pkl_files[0]), 'rb') as f:
            pw_data = pickle.load(f)
        M_float = float(pw_data['M_float'])
        M_spar  = float(pw_data['M_spar'])
        B_PTO   = float(pw_data['B_PTO'])
        K_PTO   = float(pw_data['K_PTO'])

        experiment_ids.append(exp_id)
        P_avg_annual_list.append(P_avg_annual)
        P_sea_list.append(P_sea)
        M_float_list.append(M_float)
        M_spar_list.append(M_spar)
        B_PTO_list.append(B_PTO)
        K_PTO_list.append(K_PTO)
        successful.append(exp_id)

        if exp_id % 10 == 0 or exp_id == n_doe:
            print(f"   Processed: Exp {exp_id:03d}/{n_doe}  "
                  f"P_avg={P_avg_annual:.4f} W  "
                  f"M_float={M_float:.2f} kg  M_spar={M_spar:.2f} kg")

    except Exception as e:
        print(f"⚠️  Exp {exp_id:03d}: error reading PKL ({e}) — skipping")
        failed.append(exp_id)
        continue

#%%============================================================================
# ASSEMBLE VECTORS
#==============================================================================

print("\n" + "=" * 70)
print("ASSEMBLING RESPONSE VECTORS")
print("=" * 70)

n_successful = len(successful)

if n_successful == 0:
    print(f"❌ No experiments could be assembled — nothing to save.")
    sys.exit(1)

experiment_ids   = np.array(experiment_ids, dtype=int)
P_avg_annual_vec = np.array(P_avg_annual_list)               # [n_succ]
P_sea_matrix     = np.array(P_sea_list)                       # [n_succ, 181]
M_float_vec      = np.array(M_float_list)                     # [n_succ]
M_spar_vec       = np.array(M_spar_list)                      # [n_succ]
B_PTO_vec        = np.array(B_PTO_list)                       # [n_succ]
K_PTO_vec        = np.array(K_PTO_list)                       # [n_succ]
M_total_vec      = M_float_vec + M_spar_vec

# Design matrix filtered to only the successfully-assembled experiments
# (0-based row = exp_id - 1)
row_idx = experiment_ids - 1
design_matrix_filtered = design_matrix_full[row_idx, :]       # [n_succ, n_params]

# --- Apply PTO efficiency (ONCE, here) ---
P_avg_annual_vec = P_avg_annual_vec * ETA_PTO
P_sea_matrix     = P_sea_matrix * ETA_PTO
print(f"\n⚙️  PTO efficiency applied: η = {ETA_PTO:.0%}  "
      f"→  P_avg_annual × {ETA_PTO},  P_sea × {ETA_PTO}")

# --- PMR: Power-to-Mass Ratio [W/t] — mass converted from kg to tonnes ---
PMR_vec = P_avg_annual_vec / (M_total_vec / 1000.0)
print(f"⚙️  PMR = P_avg_annual / M_total  [W/t]")

print(f"\n✅ Extraction completed:")
print(f"   Successful: {n_successful}  |  Failed: {len(failed)}")
print(f"   P_avg_annual (post-η): min={np.min(P_avg_annual_vec):.4f}  "
      f"max={np.max(P_avg_annual_vec):.4f}  mean={np.mean(P_avg_annual_vec):.4f} W")
print(f"   PMR: min={np.min(PMR_vec):.4f}  max={np.max(PMR_vec):.4f}  "
      f"mean={np.mean(PMR_vec):.4f} W/t")
print(f"   M_float:  [{np.min(M_float_vec):.2f}, {np.max(M_float_vec):.2f}] kg")
print(f"   M_spar:   [{np.min(M_spar_vec):.2f}, {np.max(M_spar_vec):.2f}] kg")
print(f"   M_total:  [{np.min(M_total_vec):.2f}, {np.max(M_total_vec):.2f}] kg")
print(f"   P_sea:    {P_sea_matrix.shape}   (columns = {sea_labels})")

if failed:
    print(f"\n⚠️  Failed experiments: {failed}")

#%%============================================================================
# SAVE RESULTS
#==============================================================================

print(f"\n{'=' * 70}")
print("SAVING RESULTS")
print(f"{'=' * 70}")

results = {
    # Response vectors [n_successful]
    'P_avg_annual':       P_avg_annual_vec,       # Annual mean power [W]  (post-η)
    'PMR':                PMR_vec,                 # Power-to-Mass Ratio [W/t]
    'M_float':            M_float_vec,             # Float heave mass [kg]
    'M_spar':             M_spar_vec,               # Spar heave mass [kg]
    'M_total':            M_total_vec,             # Float + Spar mass [kg]
    'B_PTO':              B_PTO_vec,               # PTO damping used per experiment [kg/s]
    'K_PTO':              K_PTO_vec,               # PTO stiffness used per experiment [N/m]
    # Response matrix [n_successful x 181]
    'P_sea':              P_sea_matrix,            # Power per sea state [W]  (post-η)
    'sea_labels':         sea_labels,              # 181 sea-state names (columns of P_sea)
    # Design matrix [n_successful x n_parameters]
    'design_matrix':      design_matrix_filtered,
    # Metadata
    'experiment_ids':     experiment_ids,
    'parameter_names':    parameter_names,
    'parameter_ranges':   parameter_ranges,
    'n_experiments':      n_successful,
    'n_parameters':       n_params,
    'failed_experiments': np.array(failed) if failed else np.array([]),
    'metadata': {
        'eta_PTO': ETA_PTO,
        'units': {
            'P_avg_annual': 'W',
            'PMR':          'W/t',
            'M_float':      'kg',
            'M_spar':       'kg',
            'M_total':      'kg',
            'B_PTO':        'kg/s',
            'K_PTO':        'N/m',
            'P_sea':        'W',
        }
    }
}

# PKL
try:
    with open(os.path.join(OUTPUT_FOLDER, f"{OUTPUT_FILENAME}.pkl"), 'wb') as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"✅ PKL saved: {OUTPUT_FILENAME}.pkl")
except Exception as e:
    print(f"⚠️  Error saving PKL: {e}")

# NPZ (arrays / scalars only — metadata dict and sea_labels list dropped)
try:
    npz_data = {k: v for k, v in results.items()
                if isinstance(v, (np.ndarray, int, float))}
    npz_data['parameter_names'] = np.array(parameter_names, dtype=object)
    npz_data['sea_labels']      = np.array(sea_labels, dtype=object)
    np.savez_compressed(os.path.join(OUTPUT_FOLDER, f"{OUTPUT_FILENAME}.npz"), **npz_data)
    print(f"✅ NPZ saved: {OUTPUT_FILENAME}.npz")
except Exception as e:
    print(f"⚠️  Error saving NPZ: {e}")

# MAT — labels as individual strings (MATLAB has no native list-of-str field)
try:
    mat_data = {k: v for k, v in results.items() if k != 'metadata'}
    mat_data['parameter_names'] = np.array(parameter_names, dtype=object)
    for k, lbl in enumerate(sea_labels):
        mat_data[f'sea_label_{k:02d}'] = lbl
    del mat_data['sea_labels']
    mat_data['eta_PTO'] = ETA_PTO
    savemat(os.path.join(OUTPUT_FOLDER, f"{OUTPUT_FILENAME}.mat"), mat_data, do_compression=True)
    print(f"✅ MAT saved: {OUTPUT_FILENAME}.mat")
except Exception as e:
    print(f"⚠️  Error saving MAT: {e}")

#%%============================================================================
# FINAL SUMMARY
#==============================================================================

print("\n" + "=" * 70)
print("St4_A_ResultsVector COMPLETED")
print("=" * 70)
print(f"Included: {n_successful}   |   Skipped: {len(failed)}   |   DOE size: {n_doe}")
print(f"Output:   {OUTPUT_FOLDER}/")

if failed:
    print(f"\n⚠️  {len(failed)} DOE experiment(s) missing or unreadable — the metamodel")
    print(f"   in St4_B will be trained on {n_successful}/{n_doe} points only.")

print(f"\n📦 Variables saved:")
print(f"   design_matrix    → [{n_successful} x {n_params}]  D1..D7 (input X for metamodel)")
print(f"   P_avg_annual     → [{n_successful}]                annual mean power [W]  (post-η, scalar Y)")
print(f"   PMR              → [{n_successful}]                power-to-mass ratio [W/t]  (scalar Y)")
print(f"   M_float, M_spar, M_total → [{n_successful}] each   masses [kg]")
print(f"   B_PTO            → [{n_successful}]                PTO damping used [kg/s]")
print(f"   K_PTO            → [{n_successful}]                PTO stiffness used [N/m]")
print(f"   P_sea            → [{n_successful} x 181]            power per sea state [W]  (post-η, vector Y)")
print(f"   sea_labels       → 181 sea-state names (Mean_2024 + 180 climatological, columns of P_sea)")
print(f"   experiment_ids   → [{n_successful}]                DOE_Exp id per row")

print(f"\n🎯 Next step: St4_B_MetaModel.py")
print(f"   X = design_matrix")
print(f"   Y = P_avg_annual  |  PMR  |  P_sea[:, k]  (k=0..180)")
print(f"   ⚠️  Do NOT multiply by ETA_PTO again downstream — already applied here.")
print("\n" + "=" * 70)