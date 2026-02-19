"""
St3_A_RAO.py - Batch RAO Calculation for DOE Experiments
Reads hydrodynamic coefficients and calculates heave RAOs for all DOE experiments

Author: Pablo Antonio Matamala Carvajal
Date: 2025-01-21
Updated: 2025-11-26 - Corrected batch folder path and enhanced for phase extraction
Description: Processes all DOE experiments and calculates decoupled heave RAOs (2x2 system)
"""

import os
import sys
import numpy as np
import pickle

# Add EcoFunctions to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'EcoFunctions'))

from EcoFunctions.Eco_RAO import calculate_rao_heave

#%%============================================================================
# CONFIGURATION
#==============================================================================

# Input configuration
BATCH_FOLDER = "EcoBatch"  # Folder containing DOE experiments (CORRECTED PATH)
DOE_FILE = "EcoData/DOE_values.pkl"  # DOE design file

# Output configuration
OUTPUT_FOLDER = "EcoData/RAO"  # Output folder for RAO results

# Physical constants
RHO = 1025  # Water density [kg/m³]
G = 9.81    # Gravity [m/s²]

# Resume configuration (for power outage recovery)
START_FROM_EXPERIMENT = 1  # Change this to resume from specific experiment

#%%============================================================================
# LOAD DOE INFORMATION
#==============================================================================

print("="*70)
print("RAO CALCULATION - DOE BATCH PROCESSING")
print("="*70)

print(f"\n📂 Loading DOE information from: {DOE_FILE}")

try:
    with open(DOE_FILE, 'rb') as f:
        doe_data = pickle.load(f)
    
    n_experiments = doe_data['n_experiments']
    design_matrix = doe_data['design_matrix']
    
    print(f"✅ DOE data loaded successfully!")
    print(f"   Total experiments: {n_experiments}")
    print(f"   Design matrix shape: {design_matrix.shape}")
    
except FileNotFoundError:
    print(f"❌ ERROR: DOE file not found: {DOE_FILE}")
    print(f"   Please run St1_DoeValues.py first")
    sys.exit(1)
except Exception as e:
    print(f"❌ ERROR loading DOE file: {e}")
    sys.exit(1)

#%%============================================================================
# SETUP
#==============================================================================

# Create output folder
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
print(f"\n📁 Output folder: {OUTPUT_FOLDER}/")

# Verify batch folder exists
if not os.path.exists(BATCH_FOLDER):
    print(f"❌ ERROR: Batch folder not found: {BATCH_FOLDER}")
    print(f"   Please run St2_A_Hydro.py first")
    sys.exit(1)

print(f"📁 Batch folder: {BATCH_FOLDER}/")

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
# PROCESS ALL DOE EXPERIMENTS
#==============================================================================

successful_experiments = 0
failed_experiments = 0
missing_experiments = 0

for exp_id in range(1, n_experiments + 1):
    
    # Skip experiments before START_FROM_EXPERIMENT
    if exp_id < START_FROM_EXPERIMENT:
        continue
    
    print(f"\n{'='*70}")
    print(f"Processing Experiment {exp_id}/{n_experiments}")
    print(f"{'='*70}")
    
    # Define paths
    exp_folder = os.path.join(BATCH_FOLDER, f"DOE_Exp_{exp_id:03d}")
    hydro_file = os.path.join(exp_folder, "hydroData", "HydCoeff.pkl")
    
    # Check if experiment folder exists
    if not os.path.exists(exp_folder):
        print(f"⚠️ Experiment folder not found: {exp_folder}")
        missing_experiments += 1
        continue
    
    # Check if hydrodynamic data exists
    if not os.path.exists(hydro_file):
        print(f"⚠️ Hydrodynamic data not found: {hydro_file}")
        failed_experiments += 1
        continue
    
    try:
        # Load hydrodynamic coefficients
        print(f"📂 Loading hydrodynamic data from: {hydro_file}")
        with open(hydro_file, 'rb') as f:
            hydro_data = pickle.load(f)
        
        # Extract full matrices (12×12 system)
        A_full = hydro_data['A']      # [ω, 12, 12]
        B_full = hydro_data['B']      # [ω, 12, 12]
        M_full = hydro_data['M']      # [12, 12]
        C_full = hydro_data['C']      # [12, 12]
        Fe_full = hydro_data['Fe']    # [12, ω] or [ω, 12] - need to check
        frequencies = hydro_data['w'] # [ω]
        
        print(f"✅ Hydrodynamic data loaded:")
        print(f"   A shape: {A_full.shape}")
        print(f"   B shape: {B_full.shape}")
        print(f"   Fe shape (original): {Fe_full.shape}")
        print(f"   Frequencies: {len(frequencies)} points")
        
        # Fix Fe transpose issue - ensure it's [ω, 12]
        if Fe_full.shape[0] != len(frequencies):
            print(f"   ⚠️ Fe transposed - fixing dimensions...")
            Fe_full = Fe_full.T  # Transpose to get [ω, 12]
            print(f"   Fe shape (corrected): {Fe_full.shape}")
        
        # Extract ONLY heave components (indices 2 and 8)
        # Index 2: Float heave (DOF 3 of body 1)
        # Index 8: Spar heave (DOF 3 of body 2)
        print(f"\n🔍 Extracting heave components (indices 2, 8)...")
        
        A_heave = A_full[:, [2, 8], :][:, :, [2, 8]]  # [ω, 2, 2]
        B_heave = B_full[:, [2, 8], :][:, :, [2, 8]]  # [ω, 2, 2]
        M_heave = M_full[[2, 8], :][:, [2, 8]]        # [2, 2]
        C_heave = C_full[[2, 8], :][:, [2, 8]]        # [2, 2]
        Fe_heave = Fe_full[:, [2, 8]]                 # [ω, 2]
        
        print(f"✅ Heave system extracted (2×2):")
        print(f"   A_heave: {A_heave.shape}")
        print(f"   B_heave: {B_heave.shape}")
        print(f"   M_heave: {M_heave.shape}")
        print(f"   C_heave: {C_heave.shape}")
        print(f"   Fe_heave: {Fe_heave.shape}")
        
        # Calculate RAOs with enhanced phase extraction
        print(f"\n🌊 Calculating heave RAOs with phase vectors...")
        rao_results = calculate_rao_heave(
            A_heave=A_heave,
            B_heave=B_heave,
            M_heave=M_heave,
            C_heave=C_heave,
            Fe_heave=Fe_heave,
            frequencies=frequencies,
            experiment_id=exp_id,
            output_folder=OUTPUT_FOLDER,
            save_plots=True,
            save_data=True,
            rho=RHO,
            g=G
        )
        
        print(f"✅ RAO calculation successful for Experiment {exp_id:03d}")
        print(f"   Enhanced features: Phase vectors included")
        successful_experiments += 1
        
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        failed_experiments += 1
        continue
        
    except KeyError as e:
        print(f"❌ Missing key in hydrodynamic data: {e}")
        print(f"   Available keys: {list(hydro_data.keys())}")
        failed_experiments += 1
        continue
        
    except Exception as e:
        print(f"❌ CRITICAL ERROR processing Experiment {exp_id:03d}")
        print(f"   Error type: {type(e).__name__}")
        print(f"   Message: {str(e)}")
        failed_experiments += 1
        continue

#%%============================================================================
# FINAL SUMMARY
#==============================================================================

print("\n" + "="*70)
print("🎉 ENHANCED RAO BATCH PROCESSING COMPLETED")
print("="*70)
print(f"Total experiments: {n_experiments}")
print(f"Successful: {successful_experiments}")
print(f"Failed: {failed_experiments}")
print(f"Missing: {missing_experiments}")
print(f"Results saved in folder: {OUTPUT_FOLDER}/")

print(f"\n📊 Generated files per experiment:")
print(f"   - DOE_Exp_XXX_RAO.npz (NumPy compressed)")
print(f"   - DOE_Exp_XXX_RAO.pkl (Pickle with full data)")
print(f"   - DOE_Exp_XXX_RAO.mat (MATLAB compatible)")
print(f"   - DOE_Exp_XXX_RAO.png (RAO plot)")

print(f"\n📋 Each file contains (ENHANCED):")
print(f"   • RAO_heave_float (complex + absolute)")
print(f"   • RAO_heave_spar (complex + absolute)")
print(f"   • RAO_heave_relative (complex + absolute)")
print(f"   • phase_relative_deg (NEW - full frequency range)")
print(f"   • phase_relative_rad (NEW - full frequency range)")
print(f"   • Peak frequencies and values")
print(f"   • 2×2 matrices: A, B, M, C")
print(f"   • Excitation force Fe (heave)")
print(f"   • Frequencies")

print(f"\n🎯 Next steps:")
print(f"   1. Review RAO plots in {OUTPUT_FOLDER}/")
print(f"   2. Run St4_A_ResultsVector.py to extract enhanced response vectors")
print(f"   3. Use enhanced RAO data (including phase) for metamodel generation")
print(f"   4. Run St4_B_MetaModel.py for Pareto analysis")

print(f"\n✨ ENHANCEMENTS:")
print(f"   • Phase vectors now saved for all experiments")
print(f"   • Corrected batch folder path: {BATCH_FOLDER}")
print(f"   • Ready for St4_A enhanced vector extraction")
print(f"   • Compatible with phase-based metamodeling")

if failed_experiments > 0:
    print(f"\n⚠️ WARNING: {failed_experiments} experiments failed")
    print(f"   Review error messages above for details")

if missing_experiments > 0:
    print(f"\n⚠️ WARNING: {missing_experiments} experiments missing")
    print(f"   Run St2_A_Hydro.py to generate missing experiments")

print("\n" + "="*70)
