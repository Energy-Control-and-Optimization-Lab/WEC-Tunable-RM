"""
Eco_RAO.py - RAO Calculation for Heave Motion (Decoupled Analysis)
Author: Pablo Antonio Matamala Carvajal
Date: 2025-01-21
Updated: 2025-11-26 - Added phase vectors for metamodel generation
Description: Calculates heave RAOs for two-body system with 2x2 decoupled dynamics
"""

import numpy as np
import matplotlib.pyplot as plt
import os


def calculate_rao_heave(A_heave, B_heave, M_heave, C_heave, Fe_heave, frequencies, 
                       experiment_id, output_folder="EcoData/RAO", 
                       save_plots=True, save_data=True, rho=1025, g=9.81):
    """
    Calculate heave RAOs for two-body system with decoupled dynamics (2x2 system).
    
    Parameters:
    -----------
    A_heave : array_like
        Added mass matrix [ω, 2, 2] - heave only (Float=index 0, Spar=index 1)
    B_heave : array_like
        Radiation damping matrix [ω, 2, 2] - heave only
    M_heave : array_like
        Inertia matrix [2, 2] - heave elements only
    C_heave : array_like
        Hydrostatic stiffness matrix [2, 2] - heave elements only
    Fe_heave : array_like
        Excitation force [ω, 2] - heave only (complex)
    frequencies : array_like
        Frequency array [rad/s]
    experiment_id : int
        DOE experiment number
    output_folder : str
        Output directory for RAO results
    save_plots : bool
        Save RAO plots
    save_data : bool
        Save RAO data files
    rho : float
        Water density [kg/m³]
    g : float
        Gravity acceleration [m/s²]
    
    Returns:
    --------
    dict : RAO results containing:
        - RAO_heave_float: Complex RAO for float heave
        - RAO_heave_float_abs: Magnitude of float heave RAO
        - RAO_heave_spar: Complex RAO for spar heave
        - RAO_heave_spar_abs: Magnitude of spar heave RAO
        - RAO_heave_relative: Complex RAO for relative heave
        - RAO_heave_relative_abs: Magnitude of relative heave RAO
        - phase_relative_deg: Phase of relative RAO in degrees [-180°, +180°]
        - phase_relative_rad: Phase of relative RAO in radians [-π, +π]
        - omega_peak_float: Peak frequency for float [rad/s]
        - RAO_peak_float: Peak RAO value for float [m/m]
        - omega_peak_spar: Peak frequency for spar [rad/s]
        - RAO_peak_spar: Peak RAO value for spar [m/m]
        - omega_peak_relative: Peak frequency for relative motion [rad/s]
        - RAO_peak_relative: Peak RAO value for relative motion [m/m]
        - frequencies: Frequency array
        - A_heave, B_heave, M_heave, C_heave: 2x2 matrices
    """
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    w = np.array(frequencies)
    n_freq = len(w)
    
    # Initialize RAO arrays
    RAO_float = np.zeros(n_freq, dtype=complex)
    RAO_spar = np.zeros(n_freq, dtype=complex)
    RAO_relative = np.zeros(n_freq, dtype=complex)
    
    print(f"\n{'='*70}")
    if experiment_id is not None:
        print(f"Calculating RAOs for Experiment {experiment_id:03d}")
    else:
        print(f"Calculating RAOs (St0 - Convergence Analysis)")
    print(f"{'='*70}")
    print(f"Frequency range: {w[0]:.2f} - {w[-1]:.2f} rad/s ({n_freq} points)")
    print(f"System: 2×2 (heave Float, heave Spar)")
    
    # Calculate RAOs for each frequency
    for i, omega in enumerate(w):
        # Frequency-dependent matrices
        A = A_heave[i, :, :]  # [2, 2]
        B = B_heave[i, :, :]  # [2, 2]
        Fe = Fe_heave[i, :]   # [2] complex
        
        # Impedance matrix Z = -ω²(M + A) + iωB + C
        Z = -omega**2 * (M_heave + A) + 1j * omega * B + C_heave
        
        # Wave amplitude (assuming unit amplitude wave: ζ_a = 1)
        zeta_a = 1.0
        
        try:
            # Solve for response amplitudes: Z * X = Fe * ζ_a
            X = np.linalg.solve(Z, Fe * zeta_a)  # [2] complex
            
            # RAOs = X / ζ_a
            RAO_float[i] = X[0] / zeta_a      # Float heave RAO
            RAO_spar[i] = X[1] / zeta_a       # Spar heave RAO
            RAO_relative[i] = (X[0] - X[1]) / zeta_a  # Relative heave RAO
            
        except np.linalg.LinAlgError:
            print(f"   ⚠️ Singular matrix at ω={omega:.3f} rad/s - setting RAO=0")
            RAO_float[i] = 0
            RAO_spar[i] = 0
            RAO_relative[i] = 0
    
    # Calculate absolute values
    RAO_float_abs = np.abs(RAO_float)
    RAO_spar_abs = np.abs(RAO_spar)
    RAO_relative_abs = np.abs(RAO_relative)
    
    # Calculate phase vectors (NEW FEATURE)
    phase_relative_deg = np.degrees(np.angle(RAO_relative))  # [-180°, +180°]
    phase_relative_rad = np.angle(RAO_relative)              # [-π, +π]
    
    # Find peak values
    idx_peak_float = np.argmax(RAO_float_abs)
    omega_peak_float = w[idx_peak_float]
    RAO_peak_float = RAO_float_abs[idx_peak_float]
    
    idx_peak_spar = np.argmax(RAO_spar_abs)
    omega_peak_spar = w[idx_peak_spar]
    RAO_peak_spar = RAO_spar_abs[idx_peak_spar]
    
    idx_peak_relative = np.argmax(RAO_relative_abs)
    omega_peak_relative = w[idx_peak_relative]
    RAO_peak_relative = RAO_relative_abs[idx_peak_relative]
    
    print(f"\n📊 Peak RAO Values:")
    print(f"   Float:    ω_peak={omega_peak_float:.3f} rad/s, |RAO|_peak={RAO_peak_float:.3f} m/m")
    print(f"   Spar:     ω_peak={omega_peak_spar:.3f} rad/s, |RAO|_peak={RAO_peak_spar:.3f} m/m")
    print(f"   Relative: ω_peak={omega_peak_relative:.3f} rad/s, |RAO|_peak={RAO_peak_relative:.3f} m/m")
    
    print(f"\n🔄 Phase Information:")
    print(f"   Phase range: [{np.min(phase_relative_deg):.1f}°, {np.max(phase_relative_deg):.1f}°]")
    print(f"   Phase at peak frequency: {phase_relative_deg[idx_peak_relative]:.1f}°")
    
    # Prepare results dictionary
    results = {
        # Complex RAOs
        'RAO_heave_float': RAO_float,
        'RAO_heave_spar': RAO_spar,
        'RAO_heave_relative': RAO_relative,
        # Absolute RAOs
        'RAO_heave_float_abs': RAO_float_abs,
        'RAO_heave_spar_abs': RAO_spar_abs,
        'RAO_heave_relative_abs': RAO_relative_abs,
        # Phase vectors (NEW)
        'phase_relative_deg': phase_relative_deg,
        'phase_relative_rad': phase_relative_rad,
        # Peak information
        'omega_peak_float': omega_peak_float,
        'RAO_peak_float': RAO_peak_float,
        'omega_peak_spar': omega_peak_spar,
        'RAO_peak_spar': RAO_peak_spar,
        'omega_peak_relative': omega_peak_relative,
        'RAO_peak_relative': RAO_peak_relative,
        # System matrices
        'A_heave': A_heave,
        'B_heave': B_heave,
        'M_heave': M_heave,
        'C_heave': C_heave,
        'Fe_heave': Fe_heave,
        'frequencies': w,
        # Metadata
        'experiment_id': experiment_id if experiment_id is not None else 0,
        'n_frequencies': n_freq,
        'rho': rho,
        'g': g
    }
    
    # Save data files
    if save_data:
        print(f"\n💾 Saving RAO data files...")
        
        # Save as NPZ
        try:
            if experiment_id is not None:
                npz_path = os.path.join(output_folder, f"DOE_Exp_{experiment_id:03d}_RAO.npz")
            else:
                npz_path = os.path.join(output_folder, "RAO_results.npz")
            
            np.savez_compressed(
                npz_path,
                RAO_heave_float=RAO_float,
                RAO_heave_spar=RAO_spar,
                RAO_heave_relative=RAO_relative,
                RAO_heave_float_abs=RAO_float_abs,
                RAO_heave_spar_abs=RAO_spar_abs,
                RAO_heave_relative_abs=RAO_relative_abs,
                phase_relative_deg=phase_relative_deg,
                phase_relative_rad=phase_relative_rad,
                omega_peak_float=omega_peak_float,
                RAO_peak_float=RAO_peak_float,
                omega_peak_spar=omega_peak_spar,
                RAO_peak_spar=RAO_peak_spar,
                omega_peak_relative=omega_peak_relative,
                RAO_peak_relative=RAO_peak_relative,
                A_heave=A_heave,
                B_heave=B_heave,
                M_heave=M_heave,
                C_heave=C_heave,
                Fe_heave_real=Fe_heave.real,
                Fe_heave_imag=Fe_heave.imag,
                frequencies=w,
                experiment_id=experiment_id if experiment_id is not None else 0
            )
            if experiment_id is not None:
                print(f"   ✅ NPZ saved: DOE_Exp_{experiment_id:03d}_RAO.npz")
            else:
                print(f"   ✅ NPZ saved: RAO_results.npz")
        except Exception as e:
            print(f"   ⚠️ Error saving NPZ: {e}")
        
        # Save as PKL
        try:
            import pickle
            if experiment_id is not None:
                pkl_path = os.path.join(output_folder, f"DOE_Exp_{experiment_id:03d}_RAO.pkl")
            else:
                pkl_path = os.path.join(output_folder, "RAO_results.pkl")
            
            with open(pkl_path, 'wb') as f:
                pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            if experiment_id is not None:
                print(f"   ✅ PKL saved: DOE_Exp_{experiment_id:03d}_RAO.pkl")
            else:
                print(f"   ✅ PKL saved: RAO_results.pkl")
        except Exception as e:
            print(f"   ⚠️ Error saving PKL: {e}")
        
        # Save as MAT
        try:
            from scipy.io import savemat
            if experiment_id is not None:
                mat_path = os.path.join(output_folder, f"DOE_Exp_{experiment_id:03d}_RAO.mat")
            else:
                mat_path = os.path.join(output_folder, "RAO_results.mat")
            mat_data = {
                'RAO_heave_float_real': RAO_float.real,
                'RAO_heave_float_imag': RAO_float.imag,
                'RAO_heave_spar_real': RAO_spar.real,
                'RAO_heave_spar_imag': RAO_spar.imag,
                'RAO_heave_relative_real': RAO_relative.real,
                'RAO_heave_relative_imag': RAO_relative.imag,
                'RAO_heave_float_abs': RAO_float_abs,
                'RAO_heave_spar_abs': RAO_spar_abs,
                'RAO_heave_relative_abs': RAO_relative_abs,
                'phase_relative_deg': phase_relative_deg,
                'phase_relative_rad': phase_relative_rad,
                'omega_peak_float': omega_peak_float,
                'RAO_peak_float': RAO_peak_float,
                'omega_peak_spar': omega_peak_spar,
                'RAO_peak_spar': RAO_peak_spar,
                'omega_peak_relative': omega_peak_relative,
                'RAO_peak_relative': RAO_peak_relative,
                'A_heave': A_heave,
                'B_heave': B_heave,
                'M_heave': M_heave,
                'C_heave': C_heave,
                'Fe_heave_real': Fe_heave.real,
                'Fe_heave_imag': Fe_heave.imag,
                'frequencies': w,
                'experiment_id': experiment_id if experiment_id is not None else 0
            }
            savemat(mat_path, mat_data, do_compression=True)
            print(f"   ✅ MAT saved: DOE_Exp_{experiment_id:03d}_RAO.mat")
        except Exception as e:
            print(f"   ⚠️ Error saving MAT: {e}")
    
    # Generate plot
    if save_plots:
        print(f"\n📊 Generating RAO plot...")
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Plot RAO magnitudes
        ax.plot(w, RAO_float_abs, 'b-', linewidth=2, label='RAO Float (heave)')
        ax.plot(w, RAO_spar_abs, 'r-', linewidth=2, label='RAO Spar (heave)')
        ax.plot(w, RAO_relative_abs, 'g-', linewidth=2, label='RAO Relative (Float-Spar)')
        
        # Mark peak values
        ax.plot(omega_peak_float, RAO_peak_float, 'bo', markersize=8, 
                label=f'Peak Float: ω={omega_peak_float:.2f} rad/s')
        ax.plot(omega_peak_spar, RAO_peak_spar, 'ro', markersize=8,
                label=f'Peak Spar: ω={omega_peak_spar:.2f} rad/s')
        ax.plot(omega_peak_relative, RAO_peak_relative, 'go', markersize=8,
                label=f'Peak Relative: ω={omega_peak_relative:.2f} rad/s')
        
        ax.set_xlabel('Wave Frequency [rad/s]', fontsize=12)
        ax.set_ylabel('|RAO| [m/m]', fontsize=12)
        if experiment_id is not None:
            title = f'Heave RAOs - DOE Experiment {experiment_id:03d}'
        else:
            title = 'Heave RAOs - Convergence Geometry'
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=10)
        ax.set_xlim([w[0], w[-1]])
        
        plt.tight_layout()
        
        if experiment_id is not None:
            plot_path = os.path.join(output_folder, f"DOE_Exp_{experiment_id:03d}_RAO.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"   ✅ Plot saved: DOE_Exp_{experiment_id:03d}_RAO.png")
        else:
            plot_path = os.path.join(output_folder, "RAO_plot.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"   ✅ Plot saved: RAO_plot.png")
        
        plt.close()
    
    print(f"✅ RAO calculation completed for Experiment {experiment_id if experiment_id is not None else 'St0'}")
    
    return results


if __name__ == "__main__":
    print("This module should be imported, not run directly.")
    print("Use St3_RAO.py to process DOE experiments.")
