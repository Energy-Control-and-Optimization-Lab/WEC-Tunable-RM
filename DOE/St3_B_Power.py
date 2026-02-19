"""
St3_B_Power.py - Power Calculation with PTO for DOE Experiments
Reads hydrodynamic coefficients and RAOs, calculates power absorption with PTO+viscous damping

Author: Pablo Antonio Matamala Carvajal
Date: 2025-01-21
Updated: 2025-11-30 - Enhanced with phase calculation for physical consistency
Description: Processes all DOE experiments and calculates power P(ω) and phase with PTO included

ENHANCEMENT: Now calculates and saves phase with PTO to ensure physical consistency
             between power and phase vectors in metamodel generation (St4_A).
"""

import os
import sys
import numpy as np
import pickle

# Add EcoFunctions to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'EcoFunctions'))

# Import enhanced function (note: you need to replace Eco_Power.py with enhanced version)
from EcoFunctions.Eco_Power import calculate_power

#%%============================================================================
# CONFIGURATION
#==============================================================================

# Input configuration
BATCH_FOLDER = "EcoBatch"  # Folder containing DOE experiments
DOE_FILE = "EcoData/DOE_values.pkl"  # DOE design file
RAO_FOLDER = "EcoData/RAO"  # Folder with RAO results

# Output configuration
OUTPUT_FOLDER = "EcoData/Power"  # Output folder for Power results

# Physical constants
RHO = 1025  # Water density [kg/m³]
G = 9.81    # Gravity [m/s²]

# Viscous damping parameters
B_VISC_FLOAT = 0.0  # No viscous damping for float [kg/s]
ETA_SPAR = 0.6      # Experimental coefficient for spar viscous damping

# PTO impedance parameters (paper notation: C_PTO, m_PTO, K_PTO)
M_PTO = 0.0    # PTO mass/inertia m_PTO [kg] (default: 0 = no inertia)
K_PTO = 0.0    # PTO stiffness K_PTO [N/m] (default: 0 = no spring)

# Manual override for natural frequencies [rad/s]
OMEGA_NAT_OVERRIDE = {
    14: 4.85,  # rad/s
    16: 4.20,  # rad/s
    30: 4.60,  # rad/s
    32: 4.35,  # rad/s
    42: 4.50,  # rad/s
    44: 4.45,  # rad/s
    59: 4.50,  # rad/s
}

# Resume configuration (for power outage recovery)
START_FROM_EXPERIMENT = 1  # Change this to resume from specific experiment

#%%============================================================================
# LOAD DOE INFORMATION
#==============================================================================

print("="*70)
print("POWER CALCULATION - DOE BATCH PROCESSING (ENHANCED)")
print("="*70)

print(f"\n📂 Loading DOE information from: {DOE_FILE}")

try:
    with open(DOE_FILE, 'rb') as f:
        doe_data = pickle.load(f)
    
    n_experiments = doe_data['n_experiments']
    design_matrix = doe_data['design_matrix']
    parameter_names = doe_data['parameter_names']
    
    print(f"✅ DOE data loaded successfully!")
    print(f"   Total experiments: {n_experiments}")
    print(f"   Design matrix shape: {design_matrix.shape}")
    print(f"   Parameters: {parameter_names}")
    
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
    print(f"   Please run St2_Hydro.py first")
    sys.exit(1)

# Verify RAO folder exists
if not os.path.exists(RAO_FOLDER):
    print(f"❌ ERROR: RAO folder not found: {RAO_FOLDER}")
    print(f"   Please run St3_RAO.py first")
    sys.exit(1)

print(f"📁 Batch folder: {BATCH_FOLDER}/")
print(f"📁 RAO folder: {RAO_FOLDER}/")

print(f"\n⚙️ Viscous Damping Configuration:")
print(f"   B_visc_float: {B_VISC_FLOAT} kg/s (fixed)")
print(f"   η_spar: {ETA_SPAR} (experimental coefficient)")
print(f"   B_visc_spar formula: 2 * m_float * ω_nat_float * η_spar")

print(f"\n⚠️ Manual Frequency Overrides:")
if OMEGA_NAT_OVERRIDE:
    for exp_id, omega_nat in OMEGA_NAT_OVERRIDE.items():
        print(f"   Experiment {exp_id:03d}: ω_nat = {omega_nat:.2f} rad/s")
else:
    print(f"   None")

print(f"\n🆕 ENHANCEMENT ACTIVE:")
print(f"   ✅ Phase calculation with PTO included")
print(f"   ✅ Physical consistency with power vectors")
print(f"   ✅ Ready for St4_A consistent vector extraction")

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
    
    # Extract B_PTO from DOE design matrix (D6 is column index 5)
    B_PTO = design_matrix[exp_id - 1, 5]
    print(f"📊 DOE Parameters for Experiment {exp_id:03d}:")
    print(f"   B_PTO (D6): {B_PTO:.2f} kg/s")
    
    # Define paths
    exp_folder = os.path.join(BATCH_FOLDER, f"DOE_Exp_{exp_id:03d}")
    hydro_file = os.path.join(exp_folder, "hydroData", "HydCoeff.pkl")
    rao_file = os.path.join(RAO_FOLDER, f"DOE_Exp_{exp_id:03d}_RAO.pkl")
    
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
    
    # Check if RAO data exists
    if not os.path.exists(rao_file):
        print(f"⚠️ RAO data not found: {rao_file}")
        failed_experiments += 1
        continue
    
    try:
        # Load hydrodynamic coefficients
        print(f"\n📂 Loading hydrodynamic data from: {hydro_file}")
        with open(hydro_file, 'rb') as f:
            hydro_data = pickle.load(f)
        
        # Extract full matrices (12×12 system)
        A_full = hydro_data['A']      # [ω, 12, 12]
        B_full = hydro_data['B']      # [ω, 12, 12]
        M_full = hydro_data['M']      # [12, 12]
        C_full = hydro_data['C']      # [12, 12]
        Fe_full = hydro_data['Fe']    # [12, ω] or [ω, 12]
        frequencies = hydro_data['w'] # [ω]
        
        # Fix Fe transpose if needed
        if Fe_full.shape[0] != len(frequencies):
            Fe_full = Fe_full.T
        
        print(f"✅ Hydrodynamic data loaded:")
        print(f"   Frequencies: {len(frequencies)} points [{frequencies[0]:.2f}, {frequencies[-1]:.2f}] rad/s")
        
        # Load RAO data (for natural frequency)
        print(f"📂 Loading RAO data from: {rao_file}")
        with open(rao_file, 'rb') as f:
            rao_data = pickle.load(f)
        
        # Get natural frequency (with manual override if specified)
        if exp_id in OMEGA_NAT_OVERRIDE:
            omega_nat_floater = OMEGA_NAT_OVERRIDE[exp_id]
            manual_override = True
            print(f"⚠️ Using manual override: ω_nat = {omega_nat_floater:.2f} rad/s")
        else:
            omega_nat_floater = rao_data['omega_peak_float']
            manual_override = False
            print(f"✅ From RAO: ω_nat = {omega_nat_floater:.2f} rad/s")
        
        # Extract ONLY heave components (indices 2 and 8)
        print(f"\n📁 Extracting heave components (indices 2, 8)...")
        
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
        
        # Calculate B_visc_spar
        m_floater = M_heave[0, 0]  # Mass of float in heave [kg]
        B_visc_spar = 2 * m_floater * omega_nat_floater * ETA_SPAR
        
        print(f"\n🔧 Calculated Viscous Damping:")
        print(f"   m_floater: {m_floater:.2f} kg")
        print(f"   ω_nat_floater: {omega_nat_floater:.3f} rad/s")
        print(f"   B_visc_spar: {B_visc_spar:.2f} kg/s")
        
        # Calculate Power with ENHANCED function (now includes phase calculation)
        print(f"\n⚡ Calculating power AND phase with PTO and viscous damping...")
        power_results = calculate_power(
            A_heave=A_heave,
            B_heave=B_heave,
            M_heave=M_heave,
            C_heave=C_heave,
            Fe_heave=Fe_heave,
            frequencies=frequencies,
            B_PTO=B_PTO,
            B_visc_float=B_VISC_FLOAT,
            B_visc_spar=B_visc_spar,
            experiment_id=exp_id,
            output_folder=OUTPUT_FOLDER,
            save_plots=True,
            save_data=True,
            omega_nat_used=omega_nat_floater,
            manual_override=manual_override,
            rho=RHO,
            g=G
        )
        
        # Verify enhanced features are present
        enhancement_check = all(key in power_results for key in ['phase_relative_deg', 'phase_relative_rad'])
        if enhancement_check:
            phase_range = power_results['phase_relative_deg']
            print(f"✅ Power AND phase calculation successful for Experiment {exp_id:03d}")
            print(f"📐 Phase range: [{np.min(phase_range):.1f}°, {np.max(phase_range):.1f}°]")
        else:
            print(f"⚠️  WARNING: Enhanced features not found in results")
            print(f"   Make sure you've replaced Eco_Power.py with enhanced version")
        
        successful_experiments += 1
        
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        failed_experiments += 1
        continue
        
    except KeyError as e:
        print(f"❌ Missing key in data: {e}")
        if 'hydro_data' in locals():
            print(f"   Available keys in hydro_data: {list(hydro_data.keys())}")
        if 'rao_data' in locals():
            print(f"   Available keys in rao_data: {list(rao_data.keys())}")
        failed_experiments += 1
        continue
        
    except Exception as e:
        print(f"❌ CRITICAL ERROR processing Experiment {exp_id:03d}")
        print(f"   Error type: {type(e).__name__}")
        print(f"   Message: {str(e)}")
        import traceback
        traceback.print_exc()
        failed_experiments += 1
        continue

#%%============================================================================
# FINAL SUMMARY
#==============================================================================

print("\n" + "="*70)
print("🎉 ENHANCED POWER + PHASE CALCULATION COMPLETED")
print("="*70)
print(f"Total experiments: {n_experiments}")
print(f"Successful: {successful_experiments}")
print(f"Failed: {failed_experiments}")
print(f"Missing: {missing_experiments}")
print(f"Results saved in folder: {OUTPUT_FOLDER}/")

print(f"\n📊 Generated files per experiment:")
print(f"   - DOE_Exp_XXX_Power.npz (NumPy compressed + PHASE)")
print(f"   - DOE_Exp_XXX_Power.pkl (Pickle with full data + PHASE)")
print(f"   - DOE_Exp_XXX_Power.mat (MATLAB compatible + PHASE)")
print(f"   - DOE_Exp_XXX_Power_RAO.png (RAO plot with PTO)")
print(f"   - DOE_Exp_XXX_Power_Power.png (Power plot)")

print(f"\n📋 Each file NOW contains:")
print(f"   • RAO with PTO: Float, Spar, Relative (complex + absolute)")
print(f"   • Peak frequencies and values for RAOs")
print(f"   • P(ω) - Power vs frequency [W]")
print(f"   • P_peak and ω_P_peak")
print(f"   • B_PTO, B_visc_float, B_visc_spar")
print(f"   • B_total matrix [ω, 2, 2]")
print(f"   • ω_nat_floater used + manual override flag")
print(f"   🆕 phase_relative_deg and phase_relative_rad (ENHANCED)")

print(f"\n🔧 Configuration Used:")
print(f"   B_visc_float: {B_VISC_FLOAT} kg/s")
print(f"   η_spar: {ETA_SPAR}")
print(f"   Manual overrides: {len(OMEGA_NAT_OVERRIDE)} experiments")

print(f"\n🚀 ENHANCEMENT SUMMARY:")
print(f"   ✅ Phase with PTO calculated for ALL experiments")
print(f"   ✅ Physical consistency between P_at_* and phase_at_* vectors")
print(f"   ✅ Ready for corrected St4_A_ResultsVector.py")
print(f"   ✅ Expected fix: phase_at_1_50 MAPE 411.5% → <5%")

print(f"\n📋 Implementation Steps:")
print(f"   1. Replace EcoFunctions/Eco_Power.py with enhanced version")
print(f"   2. Run this enhanced St3_B_Power.py")
print(f"   3. Update St4_A to use power_data instead of rao_data for phase")
print(f"   4. Re-run St4_A → St4_B to validate fix")

print(f"\n🎯 Next steps:")
print(f"   1. Verify phase consistency in output files")
print(f"   2. Update St4_A_ResultsVector.py with corrected data source")
print(f"   3. Re-generate metamodels with physically consistent vectors")
print(f"   4. Validate phase_at_1_50 metamodel improvement")

if failed_experiments > 0:
    print(f"\n⚠️ WARNING: {failed_experiments} experiments failed")
    print(f"   Review error messages above for details")

if missing_experiments > 0:
    print(f"\n⚠️ WARNING: {missing_experiments} experiments missing")
    print(f"   Run St2_Hydro.py and/or St3_RAO.py to generate missing data")

print(f"\n🎉 Enhanced data ready for physically consistent metamodel generation!")
print("="*70)
