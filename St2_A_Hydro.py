"""
St2_B_Hydro.py - Hydrodynamic Analysis using 7-parameter DOE values
Reads DOE design from St1_B_DoeValues and executes hydrodynamic analysis.
Full 7-parameter study: float geometry (D1, D2, D3), spar geometry (D4, D5),
PTO damping (D6), and PTO stiffness (D7).

Author: Pablo Antonio Matamala Carvajal
Date: 2025-06-18
Updated: 2025-09-11 - Added D7 (PTO stiffness K_PTO). Same treatment as D6:
                      NOT used in the BEM analysis, only carried through to
                      the experiment summary for St3_B_Power.py.

NOTE: This is the 7-parameter companion to St2_A_Hydro.py (3-parameter,
      float-only study). Results go to EcoData/St2_B_EcoBatch/ (separate
      from St2_A). D6 (B_PTO) and D7 (K_PTO) are read from the DOE but NOT
      used in the BEM analysis — they are saved in the experiment summary
      for use in St3_B_Power.py.
"""

import os
os.environ['MPLBACKEND'] = 'Agg'
import numpy as np
import pickle
import shutil
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'EcoFunctions'))

from EcoFunctions.Eco_StlRev import generate_revolution_solid_stl
from EcoFunctions.Eco_Cap2B import analyze_two_body_hydrodynamics

#%%============================================================================
# CONFIGURATION
#==============================================================================

# Input DOE file
DOE_FILE = "EcoData/St1_DOEvalues/DOE_values.pkl"

# Output configuration
BATCH_FOLDER = "EcoData/St2_EcoBatch"

# Resume configuration (for power outage recovery)
START_FROM_EXPERIMENT = 1

# --- Frequency range [rad/s] --- real site conditions
frequencies = np.arange(0.5, 2.5 + 0.025, 0.025)  # [rad/s]

# --- Water depth: real site ---
SEA_BOTTOM = -10.0  # [m]

# --- Fixed geometric constants (Scale_03 baseline, scaled x3 from tank values) ---
F1 = 0.6    # Inner radius reference [m]       (float/spar connection radius)
F2 = 0.12   # Spar bottom flange height [m]
F3 = 0.45    # Float freeboard [m]
F4 = 1.2    # Spar freeboard [m]

# STL Generation Parameters - Float
NUM_SEGMENTS_float    = 40
Z_SUBDIVISIONS_float  = 6
HEIGHT_THRESHOLD_float = 0.5   # mesh-only parameter, NOT a physical dimension
MIN_SUBDIVISIONS_float = 6

# STL Generation Parameters - Spar
NUM_SEGMENTS_spar      = 40
Z_SUBDIVISIONS_spar    = 30
HEIGHT_THRESHOLD_spar  = 0.3   # mesh-only parameter, NOT a physical dimension
MIN_SUBDIVISIONS_spar  = 5
RAD_SUBDIVISION        = 4

#%%============================================================================
# LOAD DOE VALUES
#==============================================================================

print("="*70)
print("HYDRODYNAMIC ANALYSIS - 6-PARAMETER DOE (real site conditions)")
print("="*70)

print(f"\n📂 Loading DOE values from: {DOE_FILE}")

try:
    with open(DOE_FILE, 'rb') as f:
        doe_data = pickle.load(f)

    design_matrix  = doe_data['design_matrix']
    parameter_names = doe_data['parameter_names']
    n_experiments  = doe_data['n_experiments']

    print(f"✅ DOE data loaded successfully!")
    print(f"   Experiments: {n_experiments}")
    print(f"   Parameters:  {len(parameter_names)}  {parameter_names}")
    print(f"   Design matrix shape: {design_matrix.shape}")

except FileNotFoundError:
    print(f"❌ ERROR: DOE file not found: {DOE_FILE}")
    print(f"   Please run St1_B_DoeValues.py first to generate DOE values")
    sys.exit(1)
except Exception as e:
    print(f"❌ ERROR loading DOE file: {e}")
    sys.exit(1)

# Sanity check: this script expects exactly 7 parameters [D1..D7]
if design_matrix.shape[1] != 7:
    print(f"❌ ERROR: expected 7 DOE parameters [D1..D7] but design_matrix "
          f"has {design_matrix.shape[1]} columns. Check St1_B_DoeValues.py.")
    sys.exit(1)

#%%============================================================================
# SETUP
#==============================================================================

os.makedirs(BATCH_FOLDER, exist_ok=True)
print(f"\n📁 Batch folder: {BATCH_FOLDER}/")

print(f"\n⚙️  Analysis Configuration:")
print(f"   Frequencies: {len(frequencies)} points  [{frequencies[0]:.2f}, {frequencies[-1]:.2f}] rad/s")
print(f"   Water depth: {abs(SEA_BOTTOM):.1f} m  (sea_bottom = {SEA_BOTTOM} m)")
print(f"   Float mesh:  {NUM_SEGMENTS_float} segments × {Z_SUBDIVISIONS_float} Z-subdivisions")
print(f"   Spar mesh:   {NUM_SEGMENTS_spar} segments × {Z_SUBDIVISIONS_spar} Z-subdivisions")
print(f"\n📌 Fixed geometric constants (Scale_04 baseline):")
print(f"   F1={F1} m, F2={F2} m, F3={F3} m, F4={F4} m")
print(f"   D4, D5 vary per experiment (from DOE)")
print(f"   D6 (B_PTO) read from DOE — stored in summary, not used in BEM")
print(f"   D7 (K_PTO) read from DOE — stored in summary, not used in BEM")

if START_FROM_EXPERIMENT > 1:
    print(f"\n⚡ RESUME MODE: starting from experiment {START_FROM_EXPERIMENT}")
else:
    print(f"\n▶️  Starting from beginning (experiment 1)")

print("="*70)

#%%============================================================================
# DOE HYDRODYNAMIC ANALYSIS LOOP
#==============================================================================

successful_experiments = 0
failed_experiments = 0
RS = RAD_SUBDIVISION

for experiment_id, experiment_vector in enumerate(design_matrix):
    case_counter = experiment_id + 1

    if case_counter < START_FROM_EXPERIMENT:
        continue

    D1, D2, D3, D4, D5, D6, D7 = experiment_vector

    folder_name = f"DOE_Exp_{case_counter:03d}"
    folder_path = os.path.join(BATCH_FOLDER, folder_name)

    print(f"\n{'='*70}")
    print(f"Processing DOE Experiment {case_counter}/{n_experiments}")
    print(f"{'='*70}")
    print(f"   D1={D1:.3f}  D2={D2:.3f}  D3={D3:.3f}  "
          f"D4={D4:.3f}  D5={D5:.3f}  D6={D6:.0f}  D7={D7:.0f}")
    print(f"   Folder: {folder_path}")

    # Remove existing folder to ensure a clean run
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        print(f"   Existing folder removed")

    geometry_path = os.path.join(folder_path, "geometry")
    os.makedirs(geometry_path, exist_ok=True)

    # ------------------------------------------------------------------
    # Geometry points
    # ------------------------------------------------------------------
    # FLOAT: D1 (diameter), D2 (external draft), D3 (internal draft offset)
    # Inner point sits at depth D2+D3 — always deeper than outer edge by D3.
    P_float = np.array([
        [F1/2, 0, -D2-D3],
        [D1/2, 0, -D2],
        [D1/2, 0,  F3],
        [F1/2, 0,  F3],
    ])

    # SPAR: D4 (draft) and D5 (plate diameter) now vary per experiment
    P_spar = np.array([
        [0,              0, -D4],
        *[[i*D5/(2*RS),  0, -D4]       for i in range(1, RS+1)],
        [RS*D5/(2*RS),   0, -D4+F2],
        *[[i*D5/(2*RS),  0, -D4+F2]    for i in range(RS-1, 0, -1)
          if i*D5/(2*RS) >= 1.8*F1/2],
        [F1/2,           0, -D4+F2],
        [F1/2,           0,  F4],
        [0,              0,  F4],
    ])

    print(f"\n   Geometry points FLOAT:")
    for k, pt in enumerate(P_float):
        print(f"     P{k+1}: [{pt[0]:7.4f}, {pt[1]:5.1f}, {pt[2]:7.4f}]")
    print(f"   Geometry points SPAR:")
    for k, pt in enumerate(P_spar):
        print(f"     P{k+1}: [{pt[0]:7.4f}, {pt[1]:5.1f}, {pt[2]:7.4f}]")

    # ------------------------------------------------------------------
    # STL generation
    # ------------------------------------------------------------------
    current_dir = os.getcwd()
    os.chdir(geometry_path)

    try:
        print(f"\n🔄 Generating FLOAT geometry...")
        result_float = generate_revolution_solid_stl(
            points=P_float,
            filename="float.stl",
            num_segments=NUM_SEGMENTS_float,
            z_subdivisions=Z_SUBDIVISIONS_float,
            visualize=False,
            save_plot_path=os.getcwd(),
            plot_filename="float_profile_plot.png",
            height_threshold=HEIGHT_THRESHOLD_float,
            min_subdivisions=MIN_SUBDIVISIONS_float
        )
        print(f"✅ FLOAT STL: {result_float['num_vertices']:,} vertices, "
              f"{result_float['num_triangles']:,} triangles")

        print(f"🔄 Generating SPAR geometry...")
        result_spar = generate_revolution_solid_stl(
            points=P_spar,
            filename="spar.stl",
            num_segments=NUM_SEGMENTS_spar,
            z_subdivisions=Z_SUBDIVISIONS_spar,
            visualize=False,
            save_plot_path=os.getcwd(),
            plot_filename="spar_profile_plot.png",
            height_threshold=HEIGHT_THRESHOLD_spar,
            min_subdivisions=MIN_SUBDIVISIONS_spar
        )
        print(f"✅ SPAR STL:  {result_spar['num_vertices']:,} vertices, "
              f"{result_spar['num_triangles']:,} triangles")

    except Exception as e:
        print(f"⚠️ Error generating STL files: {e}")
        os.chdir(current_dir)
        failed_experiments += 1
        continue

    os.chdir(current_dir)

    # ------------------------------------------------------------------
    # Hydrodynamic analysis
    # ------------------------------------------------------------------
    mesh1_path = os.path.join(folder_path, "geometry", "float.stl")
    mesh2_path = os.path.join(folder_path, "geometry", "spar.stl")

    if not os.path.exists(mesh1_path) or not os.path.exists(mesh2_path):
        print(f"⚠️ STL files missing after generation — skipping experiment {case_counter}")
        failed_experiments += 1
        continue

    try:
        print(f"\n🌊 Starting hydrodynamic analysis...")
        hydro_output_dir = os.path.join(folder_path, "hydroData")

        # D6 (B_PTO) is intentionally NOT passed to BEM — used in St3_B_Power.py
        results = analyze_two_body_hydrodynamics(
            mesh1_path=mesh1_path,
            mesh2_path=mesh2_path,
            frequency_range=frequencies,
            mesh1_position=[0.0, 0.0, 0.0],
            mesh2_position=[0.0, 0.0, 0.0],
            body_names=["Float", "Spar"],
            output_directory=hydro_output_dir,
            nc_filename="Eco2BPA.nc",
            plot_xlim=[-D5*0.6, D5*0.6],
            plot_ylim=[-(D4 + 0.5), F4 + 0.2],
            save_plots=True,
            show_plots=False,
            logging_level="CRITICAL",
            sea_bottom=SEA_BOTTOM
        )

        A  = results['added_mass']
        B  = results['radiation_damping']
        Fe = results['excitation_force']

        print(f"✅ BEM completed:")
        print(f"   Added mass shape:        {A.shape}")
        print(f"   Radiation damping shape: {B.shape}")
        print(f"   Excitation force shape:  {Fe.shape}")
        print(f"   Float panels: {results['Npan1']}   Spar panels: {results['Npan2']}")

        # Save experiment summary (includes D6, D7 for later use)
        import json
        doe_summary = {
            'experiment_id': case_counter,
            'varying_parameters': {
                'D1': float(D1), 'D2': float(D2), 'D3': float(D3),
                'D4': float(D4), 'D5': float(D5), 'D6': float(D6),
                'D7': float(D7)
            },
            'fixed_constants': {
                'F1': float(F1), 'F2': float(F2),
                'F3': float(F3), 'F4': float(F4)
            },
            'frequencies': frequencies.tolist(),
            'sea_bottom_m': float(SEA_BOTTOM),
            'water_depth_m': float(abs(SEA_BOTTOM)),
            'float_vertices': int(result_float['num_vertices']),
            'spar_vertices':  int(result_spar['num_vertices']),
            'float_panels':   int(results['Npan1']),
            'spar_panels':    int(results['Npan2']),
            'total_panels':   int(results['Npan1'] + results['Npan2'])
        }

        summary_path = os.path.join(folder_path, "experiment_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(doe_summary, f, indent=2)
        print(f"✅ Summary saved: experiment_summary.json  "
              f"(D6={D6:.0f}, D7={D7:.0f} stored for Power step)")

        # Verify output files
        hydro_files = ['HydCoeff.npz', 'HydCoeff.pkl', 'HydCoeff.mat']
        files_found = []
        for fname in hydro_files:
            fpath = os.path.join(hydro_output_dir, fname)
            if os.path.exists(fpath):
                files_found.append(fname)
                print(f"   ✅ {fname}: {os.path.getsize(fpath):,} bytes")
            else:
                print(f"   ⚠️  {fname}: NOT FOUND")

        if files_found:
            successful_experiments += 1
        else:
            print(f"⚠️ No output files created for experiment {case_counter}")
            failed_experiments += 1

    except Exception as e:
        print(f"⚠️ CRITICAL ERROR in BEM for experiment {case_counter}")
        print(f"   {type(e).__name__}: {e}")
        print(f"   Continuing with next experiment...")
        failed_experiments += 1
        continue

#%%============================================================================
# FINAL SUMMARY
#==============================================================================

print("\n" + "="*70)
print("🎉 6-PARAMETER DOE BATCH COMPLETED")
print("="*70)
print(f"Total experiments: {n_experiments}")
print(f"Successful:        {successful_experiments}")
print(f"Failed:            {failed_experiments}")
print(f"Results saved in:  {BATCH_FOLDER}/")

print(f"\n📁 Folder status:")
for i in range(n_experiments):
    folder_path = os.path.join(BATCH_FOLDER, f"DOE_Exp_{i+1:03d}")
    hydro_path  = os.path.join(folder_path, "hydroData")
    if os.path.exists(hydro_path):
        print(f"   ✅ DOE_Exp_{i+1:03d}  (hydroData ✓)")
    elif os.path.exists(folder_path):
        print(f"   ⚠️  DOE_Exp_{i+1:03d}  (hydroData ✗)")
    else:
        print(f"   ❌ DOE_Exp_{i+1:03d}  (failed)")

print(f"\n🔧 Configuration used:")
print(f"   Frequencies: {len(frequencies)} pts  [{frequencies[0]:.2f}, {frequencies[-1]:.2f}] rad/s")
print(f"   Water depth: {abs(SEA_BOTTOM):.1f} m")
print(f"   Float mesh:  {NUM_SEGMENTS_float} × {Z_SUBDIVISIONS_float}")
print(f"   Spar mesh:   {NUM_SEGMENTS_spar} × {Z_SUBDIVISIONS_spar}")

print(f"\n🎯 Next steps:")
print(f"   1. Run St3_B_RAO.py    → RAO calculation per experiment")
print(f"   2. Run St3_B_Power.py  → power absorption (reads D6 from summary JSON)")
print(f"   3. Run St4_B_Results.py → build response vectors for metamodel")

print("\n" + "="*70)
