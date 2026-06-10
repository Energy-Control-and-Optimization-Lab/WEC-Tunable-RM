"""
St0_PTODamping.py - Optimal PTO Damping Analysis
Calculates optimal B_PTO(ω) to define DOE range for D6 parameter

Author: Pablo Antonio Matamala Carvajal
Date: 2025-01-21
Description: 
- Uses convergence study geometries (St0_Conv*)
- Performs 2-body hydrodynamic analysis
- Calculates RAO without PTO
- Computes optimal B_PTO for each frequency (2 methods)
- Outputs range recommendation for DOE parameter D6
"""

import os
import sys
import numpy as np
import math
import pickle
import matplotlib.pyplot as plt

# Add EcoFunctions to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'EcoFunctions'))

from EcoFunctions.Eco_StlRev import generate_revolution_solid_stl
from EcoFunctions.Eco_Cap2B import analyze_two_body_hydrodynamics
from EcoFunctions.Eco_RAO import calculate_rao_heave

#%%============================================================================
# CONFIGURATION
#==============================================================================

# Output directory
OUTPUT_FOLDER = "EcoData/St0_Convergence/PTODamping"

# Geometry parameters (from St0 convergence studies)
F1 = 0.2
F2 = 0.04
F3 = 0.15
F4 = 0.4

D1 = 0.8
D2 = 0.1
D3 = 10
D4 = 1.4
D5 = 0.8

# Frequency range
frequencies = np.arange(0.5, 8 + 0.1, 0.1)  # [rad/s]
#frequencies = np.array([0.5, 1.0, 1.5, 2.0])  # [rad/s] - 4 frecuencias para prueba

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

# Viscous damping parameters
B_VISC_FLOAT = 0.0
ETA_SPAR = 0.6

# Physical constants
RHO = 1025  # kg/m³
G = 9.81    # m/s²

#%%============================================================================
# CREATE OUTPUT STRUCTURE
#==============================================================================

print("="*70)
print("OPTIMAL PTO DAMPING ANALYSIS - St0")
print("="*70)

# Create folder structure
geometry_path = os.path.join(OUTPUT_FOLDER, "geometry")
hydro_path = os.path.join(OUTPUT_FOLDER, "hydroData")
rao_path = os.path.join(OUTPUT_FOLDER, "RAO")

os.makedirs(geometry_path, exist_ok=True)
os.makedirs(hydro_path, exist_ok=True)
os.makedirs(rao_path, exist_ok=True)

print(f"\n📁 Output structure:")
print(f"   {OUTPUT_FOLDER}/")
print(f"   ├── geometry/")
print(f"   ├── hydroData/")
print(f"   └── RAO/")

#%%============================================================================
# GENERATE GEOMETRIES
#==============================================================================

print(f"\n{'='*70}")
print("STEP 1: GEOMETRY GENERATION")
print(f"{'='*70}")

# Define Float geometry
P_float = np.array([
    [F1/2, 0, -D2-((D1/2)-(F1/2))*math.tan(math.radians(D3))],
    [D1/2, 0, -D2],
    [D1/2, 0, F3],
    [F1/2, 0, F3],
])

print(f"\n🔷 Float Geometry:")
print(f"   D1 (diameter): {D1} m")
print(f"   D2 (draft): {D2} m")
print(f"   D3 (angle): {D3}°")

# Define Spar geometry
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

print(f"\n🔶 Spar Geometry:")
print(f"   D4 (draft): {D4} m")
print(f"   D5 (plate diameter): {D5} m")

# Change to geometry directory
current_dir = os.getcwd()
os.chdir(geometry_path)

# Generate Float STL
print("\n🔄 Generating Float STL...")
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

print(f"✅ Float STL: {result_float['num_vertices']:,} vertices, {result_float['num_triangles']:,} triangles")

# Generate Spar STL
print("\n🔄 Generating Spar STL...")
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

print(f"✅ Spar STL: {result_spar['num_vertices']:,} vertices, {result_spar['num_triangles']:,} triangles")

# Return to original directory
os.chdir(current_dir)

#%%============================================================================
# HYDRODYNAMIC ANALYSIS
#==============================================================================

print(f"\n{'='*70}")
print("STEP 2: HYDRODYNAMIC ANALYSIS (BEM)")
print(f"{'='*70}")

mesh1_path = os.path.join(geometry_path, "float.stl")
mesh2_path = os.path.join(geometry_path, "spar.stl")

print(f"\n🌊 Running Capytaine BEM solver...")
print(f"   Frequencies: {len(frequencies)} points [{frequencies[0]:.2f}, {frequencies[-1]:.2f}] rad/s")

results = analyze_two_body_hydrodynamics(
    mesh1_path=mesh1_path,
    mesh2_path=mesh2_path,
    frequency_range=frequencies,
    mesh1_position=[0.0, 0.0, 0.0],
    mesh2_position=[0.0, 0.0, 0.0],
    body_names=["Float", "Spar"],
    output_directory=hydro_path,
    nc_filename="rm3.nc",
    plot_xlim=[-1.5, 1.5],
    plot_ylim=[-2, 1],
    save_plots=True,
    show_plots=False,
    logging_level="WARNING"
)

# Extract hydrodynamic coefficients
A = results['added_mass']           # [ω, 12, 12]
B = results['radiation_damping']    # [ω, 12, 12]
Fe = results['excitation_force']    # [ω, 12]
M = results['inertia_matrix']       # [12, 12]
C = results['hydrostatic_stiffness'] # [12, 12]
w = results['frequencies']          # [ω]

print(f"\n✅ Hydrodynamic analysis completed")
print(f"   A shape: {A.shape}")
print(f"   B shape: {B.shape}")

#%%============================================================================
# EXTRACT HEAVE COMPONENTS
#==============================================================================

print(f"\n{'='*70}")
print("STEP 3: EXTRACT HEAVE SYSTEM (2×2)")
print(f"{'='*70}")

# Fix Fe transpose if needed
if Fe.shape[0] != len(frequencies):
    Fe = Fe.T

# Extract heave indices (2: Float heave, 8: Spar heave)
A_heave = A[:, [2, 8], :][:, :, [2, 8]]  # [ω, 2, 2]
B_heave = B[:, [2, 8], :][:, :, [2, 8]]  # [ω, 2, 2]
M_heave = M[[2, 8], :][:, [2, 8]]        # [2, 2]
C_heave = C[[2, 8], :][:, [2, 8]]        # [2, 2]
Fe_heave = Fe[:, [2, 8]]                 # [ω, 2]

print(f"✅ Heave system extracted:")
print(f"   A_heave: {A_heave.shape}")
print(f"   M_heave: {M_heave.shape}")

#%%============================================================================
# CALCULATE RAO WITHOUT PTO
#==============================================================================

print(f"\n{'='*70}")
print("STEP 4: CALCULATE RAO (WITHOUT PTO)")
print(f"{'='*70}")

print(f"\n🌊 Calculating RAO using Eco_RAO.py...")

rao_results = calculate_rao_heave(
    A_heave=A_heave,
    B_heave=B_heave,
    M_heave=M_heave,
    C_heave=C_heave,
    Fe_heave=Fe_heave,
    frequencies=w,
    experiment_id=None,  # St0 - no experiment ID
    output_folder=rao_path,
    save_plots=True,
    save_data=True
)

# Extract results
RAO_float = rao_results['RAO_heave_float']
RAO_spar = rao_results['RAO_heave_spar']
RAO_relative = rao_results['RAO_heave_relative']
RAO_float_abs = rao_results['RAO_heave_float_abs']
RAO_spar_abs = rao_results['RAO_heave_spar_abs']
RAO_relative_abs = rao_results['RAO_heave_relative_abs']
omega_peak_float = rao_results['omega_peak_float']
RAO_peak_float = rao_results['RAO_peak_float']

print(f"\n✅ RAO calculated successfully")
print(f"   Peak Float: ω={omega_peak_float:.3f} rad/s, |RAO|={RAO_peak_float:.3f} m/m")

#%%============================================================================
# CALCULATE VISCOUS DAMPING
#==============================================================================

print(f"\n{'='*70}")
print("STEP 5: CALCULATE VISCOUS DAMPING")
print(f"{'='*70}")

m_floater = M_heave[0, 0]
B_visc_spar = 2 * m_floater * omega_peak_float * ETA_SPAR
B_visc_total = B_VISC_FLOAT + B_visc_spar

print(f"   m_floater: {m_floater:.2f} kg")
print(f"   ω_nat_float: {omega_peak_float:.3f} rad/s")
print(f"   B_visc_float: {B_VISC_FLOAT:.2f} kg/s")
print(f"   B_visc_spar: {B_visc_spar:.2f} kg/s")
print(f"   B_visc_total: {B_visc_total:.2f} kg/s")

#%%============================================================================
# CALCULATE OPTIMAL B_PTO
#==============================================================================

print(f"\n{'='*70}")
print("STEP 6: CALCULATE OPTIMAL B_PTO(ω)")
print(f"{'='*70}")

B_PTO_opt_option1 = np.zeros(len(w))
B_PTO_opt_option2 = np.zeros(len(w))
B_rad_eff_array = np.zeros(len(w))

for i, omega in enumerate(w):
    # Effective radiation damping
    B_rad_eff = B_heave[i,0,0] + B_heave[i,1,1] - 2*B_heave[i,0,1]
    B_rad_eff_array[i] = B_rad_eff
    
    # System damping (radiation + viscous)
    B_system_eff = B_rad_eff + B_visc_total
    
    # Effective mass
    M_eff = (M_heave[0,0] + A_heave[i,0,0]) + (M_heave[1,1] + A_heave[i,1,1]) - 2*A_heave[i,0,1]
    
    # Effective stiffness
    C_eff = C_heave[0,0] + C_heave[1,1]
    
    # Reactance term
    reactance = omega * M_eff - C_eff / omega
    
    # Option 1: Complete (maximum power criterion)
    B_PTO_opt_option1[i] = np.sqrt(B_system_eff**2 + reactance**2)
    
    # Option 2: Simplified (radiation + viscous)
    B_PTO_opt_option2[i] = B_system_eff

print(f"\n✅ Optimal B_PTO calculated")
print(f"\n📊 Statistics:")
print(f"   Option 1 (Complete):")
print(f"      Min: {np.min(B_PTO_opt_option1):.2f} kg/s")
print(f"      Max: {np.max(B_PTO_opt_option1):.2f} kg/s")
print(f"      Mean: {np.mean(B_PTO_opt_option1):.2f} kg/s")
print(f"   Option 2 (Simplified):")
print(f"      Min: {np.min(B_PTO_opt_option2):.2f} kg/s")
print(f"      Max: {np.max(B_PTO_opt_option2):.2f} kg/s")
print(f"      Mean: {np.mean(B_PTO_opt_option2):.2f} kg/s")

# Suggested DOE range
B_min_suggested = 0.5 * B_PTO_opt_option2[0]
B_max_suggested = 1.5 * B_PTO_opt_option2[41]

print(f"\n🎯 SUGGESTED DOE RANGE for D6:")
print(f"   Conservative: [{B_min_suggested:.0f}, {B_max_suggested:.0f}] kg/s")
print(f"   Tight around optimal: [{np.min(B_PTO_opt_option1):.0f}, {np.max(B_PTO_opt_option1):.0f}] kg/s")

#%%============================================================================
# SAVE RESULTS
#==============================================================================

print(f"\n{'='*70}")
print("STEP 7: SAVE RESULTS")
print(f"{'='*70}")

# Prepare data
b_pto_data = {
    'frequencies': w,
    'B_PTO_opt_option1': B_PTO_opt_option1,
    'B_PTO_opt_option2': B_PTO_opt_option2,
    'B_rad_eff': B_rad_eff_array,
    'B_visc_total': B_visc_total,
    'B_visc_float': B_VISC_FLOAT,
    'B_visc_spar': B_visc_spar,
    'B_min_suggested': B_min_suggested,
    'B_max_suggested': B_max_suggested,
    'omega_peak_float': omega_peak_float,
    'metadata': {
        'D1': D1, 'D2': D2, 'D3': D3, 'D4': D4, 'D5': D5,
        'description': 'Optimal PTO damping analysis for DOE range definition'
    }
}

# Save NPZ
np.savez_compressed(os.path.join(OUTPUT_FOLDER, "B_PTO_opt_data.npz"), **b_pto_data)
print(f"✅ Saved: B_PTO_opt_data.npz")

# Save PKL
with open(os.path.join(OUTPUT_FOLDER, "B_PTO_opt_data.pkl"), 'wb') as f:
    pickle.dump(b_pto_data, f)
print(f"✅ Saved: B_PTO_opt_data.pkl")

# Save MAT
try:
    from scipy.io import savemat
    savemat(os.path.join(OUTPUT_FOLDER, "B_PTO_opt_data.mat"), b_pto_data)
    print(f"✅ Saved: B_PTO_opt_data.mat")
except Exception as e:
    print(f"⚠️ Could not save MAT: {e}")

#%%============================================================================
# PLOT OPTIMAL B_PTO
#==============================================================================

print(f"\n{'='*70}")
print("STEP 8: GENERATE PLOT")
print(f"{'='*70}")

fig, ax = plt.subplots(figsize=(12, 7))

# Plot Option 1 (Complete)

# Plot Option 2 (Simplified)
ax.plot(w, B_PTO_opt_option2, 'r--', linewidth=2, 
        label=f'Option 2: Simplified (B_rad + B_visc)')

# Mark min/max
idx_min = np.argmin(B_PTO_opt_option2)
idx_max = np.argmax(B_PTO_opt_option2)
ax.plot(w[0], B_PTO_opt_option2[0], 'go', markersize=10,
        label=f'Min: {B_PTO_opt_option1[idx_min]:.0f} kg/s')
ax.plot(w[41], B_PTO_opt_option2[41], 'ro', markersize=10,
        label=f'Max: {B_PTO_opt_option1[idx_max]:.0f} kg/s')

# Shaded region for suggested DOE range
ax.axhspan(B_min_suggested, B_max_suggested, alpha=0.15, color='green',
          label=f'Suggested DOE range: [{B_min_suggested:.0f}, {B_max_suggested:.0f}] kg/s')

ax.set_xlabel('Wave Frequency [rad/s]', fontsize=13)
ax.set_ylabel('B_PTO optimal [kg/s]', fontsize=13)
ax.set_title('Optimal PTO Damping vs Frequency\n' + 
             f'Geometry: D1={D1}m, D2={D2}m, D3={D3}°, D4={D4}m, D5={D5}m',
             fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(loc='best', fontsize=10)
ax.set_xlim([w[0], w[-1]])
ax.set_ylim([0, 1500])

plt.tight_layout()
plot_path = os.path.join(OUTPUT_FOLDER, "B_PTO_optimal.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"✅ Plot saved: B_PTO_optimal.png")
plt.close()

#%%============================================================================
# FINAL SUMMARY
#==============================================================================

print(f"\n{'='*70}")
print("🎉 OPTIMAL PTO DAMPING ANALYSIS COMPLETED")
print(f"{'='*70}")

print(f"\n📁 Results saved in: {OUTPUT_FOLDER}/")
print(f"   ├── geometry/ (STL files + profile plots)")
print(f"   ├── hydroData/ (A, B, M, C, Fe coefficients)")
print(f"   ├── RAO/ (RAO without PTO - calculated with Eco_RAO.py)")
print(f"   ├── B_PTO_optimal.png")
print(f"   ├── B_PTO_opt_data.npz")
print(f"   ├── B_PTO_opt_data.pkl")
print(f"   └── B_PTO_opt_data.mat")

print(f"\n🎯 RECOMMENDATION FOR DOE PARAMETER D6:")
print(f"   Current range: [500, 2000] kg/s")
print(f"   Optimal range: [{np.min(B_PTO_opt_option1):.0f}, {np.max(B_PTO_opt_option1):.0f}] kg/s")
print(f"   Suggested (±50%): [{B_min_suggested:.0f}, {B_max_suggested:.0f}] kg/s")

print(f"\n{'='*70}")
