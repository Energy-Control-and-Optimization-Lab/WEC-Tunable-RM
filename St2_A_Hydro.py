"""
St2_Hydro.py - Hydrodynamic Analysis using DOE values
Reads DOE design from St1_DoeValues and executes hydrodynamic analysis

Author: Pablo Antonio Matamala Carvajal
Date: 2025-01-21
Updated: Improved error handling and directory verification
Updated: 2025-03-05 - ADDED: sea_bottom=-2.4 m (wave tank finite depth)
                              Fixed LiveError by using logging_level="CRITICAL" in batch mode
"""

import os
os.environ['MPLBACKEND'] = 'Agg'  # Non-interactive backend (fixes display issues)
import math
import numpy as np
import pickle
import shutil
import sys
import matplotlib.pyplot as plt

# Add EcoFunctions to path
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
START_FROM_EXPERIMENT = 1  # Change this to resume from specific experiment (e.g., 40)

# Frequency range
frequencies = np.arange(0.5, 8 + 0.1, 0.1)  # [rad/s]

# Water depth — wave tank
SEA_BOTTOM = -2.4  # [m] Finite depth. Use -np.inf for infinite depth.

# STL Generation Parameters - Float
NUM_SEGMENTS_float = 40
Z_SUBDIVISIONS_float = 6
HEIGHT_THRESHOLD_float = 0.05
MIN_SUBDIVISIONS_float = 6

# STL Generation Parameters - Spar
NUM_SEGMENTS_spar = 40
Z_SUBDIVISIONS_spar = 50
HEIGHT_THRESHOLD_spar = 0.06
MIN_SUBDIVISIONS_spar = 5
RAD_SUBDIVISION = 4

#%%============================================================================
# LOAD DOE VALUES
#==============================================================================

print("="*70)
print("HYDRODYNAMIC ANALYSIS - DOE-BASED")
print("="*70)

print(f"\n📂 Loading DOE values from: {DOE_FILE}")

try:
    with open(DOE_FILE, 'rb') as f:
        doe_data = pickle.load(f)
    
    design_matrix = doe_data['design_matrix']
    parameter_names = doe_data['parameter_names']
    n_experiments = doe_data['n_experiments']
    
    print(f"✅ DOE data loaded successfully!")
    print(f"   Experiments: {n_experiments}")
    print(f"   Parameters: {len(parameter_names)}")
    print(f"   Design matrix shape: {design_matrix.shape}")
    
except FileNotFoundError:
    print(f"❌ ERROR: DOE file not found: {DOE_FILE}")
    print(f"   Please run St1_DoeValues.py first to generate DOE values")
    sys.exit(1)
except Exception as e:
    print(f"❌ ERROR loading DOE file: {e}")
    sys.exit(1)

#%%============================================================================
# SETUP
#==============================================================================

# Create batch folder
os.makedirs(BATCH_FOLDER, exist_ok=True)
print(f"\n📁 Batch folder: {BATCH_FOLDER}/")

print(f"\n⚙️ Analysis Configuration:")
print(f"   Frequencies: {len(frequencies)} points from {frequencies[0]:.1f} to {frequencies[-1]:.1f} rad/s")
print(f"   Water depth: {abs(SEA_BOTTOM):.1f} m  (sea_bottom = {SEA_BOTTOM} m)")
print(f"   Float mesh: {NUM_SEGMENTS_float} segments × {Z_SUBDIVISIONS_float} Z-subdivisions")
print(f"   Spar mesh: {NUM_SEGMENTS_spar} segments × {Z_SUBDIVISIONS_spar} Z-subdivisions")
print(f"   Spar adaptive subdivision: height < {HEIGHT_THRESHOLD_spar}m → {MIN_SUBDIVISIONS_spar} subdivisions")

# Resume configuration
if START_FROM_EXPERIMENT > 1:
    print(f"\n⚡ RESUME MODE ACTIVE:")
    print(f"   Starting from experiment: {START_FROM_EXPERIMENT}")
    print(f"   Skipping experiments: 1 to {START_FROM_EXPERIMENT - 1}")
    experiments_to_skip = START_FROM_EXPERIMENT - 1
    experiments_to_process = n_experiments - experiments_to_skip
    print(f"   Experiments to process: {experiments_to_process} (of {n_experiments} total)")
else:
    print(f"\n▶️ Starting from beginning (experiment 1)")

print("="*70)

#%%============================================================================
# DOE HYDRODYNAMIC ANALYSIS LOOP
#==============================================================================

successful_experiments = 0
failed_experiments = 0

for experiment_id, experiment_vector in enumerate(design_matrix):
    case_counter = experiment_id + 1
    
    # Skip experiments before START_FROM_EXPERIMENT
    if case_counter < START_FROM_EXPERIMENT:
        continue
    
    D1, D2, D3, D4, D5, D6 = experiment_vector
    
    # Create folder name based on DOE experiment
    folder_name = f"DOE_Exp_{case_counter:03d}"
    folder_path = os.path.join(BATCH_FOLDER, folder_name)
    
    print(f"\n{'='*70}")
    print(f"Processing DOE Experiment {case_counter}/{n_experiments}")
    print(f"{'='*70}")
    print(f"Parameters: D1={D1:.2f}, D2={D2:.2f}, D3={D3:.2f}, D4={D4:.2f}, D5={D5:.2f}, D6={D6:.2f}")
    print(f"Folder: {folder_path}")
    
    # Remove existing folder if it exists
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        print(f"   Existing folder removed")
    
    # Create main folder and geometry subfolder
    os.makedirs(folder_path, exist_ok=True)
    geometry_path = os.path.join(folder_path, "geometry")
    os.makedirs(geometry_path, exist_ok=True)
    print(f"✅ Folders created:")
    print(f"   - {folder_path}")
    print(f"   - {geometry_path}")
    
    # Define geometry points for FLOAT using DOE parameters
    F1 = 0.2
    F2 = 0.04
    F3 = 0.15
    F4 = 0.4
    
    P_float = np.array([
        [F1/2, 0, -D2-((D1/2)-(F1/2))*math.tan(math.radians(D3))],
        [D1/2, 0, -D2],
        [D1/2, 0, F3],
        [F1/2, 0, F3],
    ])
    
    # Define geometry points for SPAR using DOE parameters
    RS = RAD_SUBDIVISION
    P_spar = np.array([
        [0, 0, -D4],
        *[[i*D5/(2*RS), 0, -D4] for i in range(1, RS+1)],
        [RS*D5/(2*RS), 0, -D4+F2],
        *[[i*D5/(2*RS), 0, -D4+F2] for i in range(RS-1, 0, -1) if i*D5/(2*RS) >= 1.8*F1/2],
        [F1/2, 0, -D4+F2],
        [F1/2, 0, F4],
        [0, 0, F4],
    ])
    
    print(f"\nGeometry points for FLOAT:")
    for k, point in enumerate(P_float):
        print(f"  P{k+1}: [{point[0]:6.3f}, {point[1]:6.3f}, {point[2]:6.3f}]")
    
    print(f"Geometry points for SPAR:")
    for k, point in enumerate(P_spar):
        print(f"  P{k+1}: [{point[0]:6.3f}, {point[1]:6.3f}, {point[2]:6.3f}]")
    
    # Change directory temporarily to save STL files
    current_dir = os.getcwd()
    os.chdir(geometry_path)
    
    try:
        # Generate FLOAT revolution solid
        print("\n🔄 Generating FLOAT geometry...")
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
        
        print(f"✅ FLOAT STL generated: {result_float['filename']}")
        print(f"   Vertices: {result_float['num_vertices']:,}")
        print(f"   Triangles: {result_float['num_triangles']:,}")
        
        # Generate SPAR revolution solid with adaptive subdivision
        print("🔄 Generating SPAR geometry...")
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
        
        print(f"✅ SPAR STL generated: {result_spar['filename']}")
        print(f"   Vertices: {result_spar['num_vertices']:,}")
        print(f"   Triangles: {result_spar['num_triangles']:,}")
        
    except Exception as e:
        print(f"⚠️ Error generating STL files: {e}")
        os.chdir(current_dir)
        failed_experiments += 1
        continue
    
    # Return to original directory
    os.chdir(current_dir)
    
    # Run hydrodynamic analysis
    try:
        print("\n🌊 Starting hydrodynamic analysis...")
        
        # Define mesh paths
        mesh1_path = os.path.join(folder_path, "geometry", "float.stl")
        mesh2_path = os.path.join(folder_path, "geometry", "spar.stl")
        
        # Verify both files exist
        if not os.path.exists(mesh1_path):
            print(f"⚠️ Error: {mesh1_path} not found")
            failed_experiments += 1
            continue
        if not os.path.exists(mesh2_path):
            print(f"⚠️ Error: {mesh2_path} not found")
            failed_experiments += 1
            continue
        
        # Output directory for hydrodynamic data
        hydro_output_dir = os.path.join(folder_path, "hydroData")
        print(f"📁 Hydrodynamic output will be saved to: {hydro_output_dir}")
        
        # Run hydrodynamic analysis
        # CRITICAL: logging_level="CRITICAL" prevents LiveError in batch loops
        #           sea_bottom=SEA_BOTTOM applies finite depth boundary condition
        results = analyze_two_body_hydrodynamics(
            mesh1_path=mesh1_path,
            mesh2_path=mesh2_path,
            frequency_range=frequencies,
            mesh1_position=[0.0, 0.0, 0.0],
            mesh2_position=[0.0, 0.0, 0.0],
            body_names=["Float", "Spar"],
            output_directory=hydro_output_dir,
            nc_filename="Eco2BPA.nc",
            plot_xlim=[-1.5, 1.5],
            plot_ylim=[-2, 1],
            save_plots=True,
            show_plots=False,
            logging_level="CRITICAL",   # ← prevents LiveError in batch loops
            sea_bottom=SEA_BOTTOM       # ← finite depth: -2.4 m
        )
        
        # Extract results
        A = results['added_mass']
        B = results['radiation_damping']
        Fe = results['excitation_force']
        
        print(f"✅ Analysis completed for {len(frequencies)} frequencies")
        print(f"   Added mass shape: {A.shape}")
        print(f"   Radiation damping shape: {B.shape}")
        print(f"   Excitation force shape: {Fe.shape}")
        print(f"   Water depth used: {abs(SEA_BOTTOM):.1f} m")
        
        # Save DOE results summary
        import json
        doe_summary = {
            'experiment_id': case_counter,
            'parameters': {
                'D1': float(D1), 'D2': float(D2), 'D3': float(D3),
                'D4': float(D4), 'D5': float(D5), 'D6': float(D6)
            },
            'frequencies': frequencies.tolist(),
            'sea_bottom_m': float(SEA_BOTTOM),
            'water_depth_m': float(abs(SEA_BOTTOM)),
            'float_vertices': int(result_float['num_vertices']),
            'spar_vertices': int(result_spar['num_vertices']),
            'float_panels': int(results['Npan1']),
            'spar_panels': int(results['Npan2']),
            'total_panels': int(results['Npan1'] + results['Npan2'])
        }
        
        summary_path = os.path.join(folder_path, "experiment_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(doe_summary, f, indent=2)
        print(f"✅ Experiment summary saved: experiment_summary.json")
        
        # Verify output files
        hydro_files = ['HydCoeff.npz', 'HydCoeff.pkl', 'HydCoeff.mat']
        print(f"🗂️ Verifying output files...")
        files_found = []
        for file in hydro_files:
            file_path = os.path.join(hydro_output_dir, file)
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                print(f"   ✅ {file}: {file_size:,} bytes")
                files_found.append(file)
            else:
                print(f"   ⚠️ {file}: NOT FOUND")
        
        if os.path.exists(hydro_output_dir):
            print(f"✅ hydroData folder verified: {hydro_output_dir}")
        else:
            print(f"❌ ERROR: hydroData folder was NOT created!")
            failed_experiments += 1
            continue
        
        if files_found:
            print(f"✅ {len(files_found)} output files created successfully")
            successful_experiments += 1
        else:
            print(f"⚠️ WARNING: No output files were created")
            failed_experiments += 1
        
    except Exception as e:
        print(f"⚠️ CRITICAL ERROR in hydrodynamic analysis for {folder_path}")
        print(f"   Error type: {type(e).__name__}")
        print(f"   Message: {str(e)}")
        print(f"   Continuing with next experiment...")
        failed_experiments += 1
        continue

#%%============================================================================
# FINAL SUMMARY
#==============================================================================

print("\n" + "="*70)
print("🎉 DOE BATCH PROCESSING COMPLETED")
print("="*70)
print(f"Total experiments: {n_experiments}")
print(f"Successful: {successful_experiments}")
print(f"Failed: {failed_experiments}")
print(f"Results saved in folder: {BATCH_FOLDER}/")
print(f"Water depth used: {abs(SEA_BOTTOM):.1f} m  (sea_bottom = {SEA_BOTTOM} m)")

print(f"\n📁 Generated folders:")
for i in range(n_experiments):
    folder_name = f"DOE_Exp_{i+1:03d}"
    folder_path = os.path.join(BATCH_FOLDER, folder_name)
    hydro_path = os.path.join(folder_path, "hydroData")
    
    if os.path.exists(folder_path):
        if os.path.exists(hydro_path):
            print(f"  ✅ {folder_name} (hydroData ✓)")
        else:
            print(f"  ⚠️ {folder_name} (hydroData ✗)")
    else:
        print(f"  ❌ {folder_name} (failed)")

print(f"\n🔧 Analysis Configuration Summary:")
print(f"   Frequencies: {len(frequencies)} points [{frequencies[0]:.1f}, {frequencies[-1]:.1f}] rad/s")
print(f"   Water depth: {abs(SEA_BOTTOM):.1f} m")
print(f"   Float mesh: {NUM_SEGMENTS_float} × {Z_SUBDIVISIONS_float}")
print(f"   Spar mesh: {NUM_SEGMENTS_spar} × {Z_SUBDIVISIONS_spar}")

print(f"\n✅ Analysis complete! Check results in '{BATCH_FOLDER}/' folder")
print("📋 Each experiment includes:")
print("  - Float geometry (parametric using DOE)")
print("  - Spar geometry (parametric using DOE)")
print("  - hydroData/ folder with:")
print("    * Hydrodynamic coefficients (A, B, Fe, M, C)")
print("    * NetCDF file (Eco2BPA.nc)")
print("    * Plots (geometry, damping, added mass)")
print("  - Experiment summary (JSON) — includes sea_bottom metadata")

print(f"\n🎯 Next steps:")
print(f"   1. Verify all experiments completed successfully")
print(f"   2. Run St3_A_RAO.py for RAO calculation")
print(f"   3. Run St3_B_Power.py for power calculation")
print(f"   4. Run St4_A_ResultsVector.py to build response vectors")
print(f"   5. Run St4_B_MetaModel.py to train surrogate")

print("\n" + "="*70)
