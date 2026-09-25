"""
Eco_Power.py - Power Calculation with PTO and Viscous Damping
Author: Pablo Antonio Matamala Carvajal
Date: 2025-01-21
Updated: 2025-11-30 - Added phase calculation for physical consistency
Description: Calculates RAOs with PTO+viscous damping and power absorption P(ω)
ENHANCEMENT: Now includes phase calculation with PTO for consistent vector extraction
"""

import numpy as np
import matplotlib.pyplot as plt
import os


def calculate_power(A_heave, B_heave, M_heave, C_heave, Fe_heave, frequencies,
                   B_PTO, B_visc_float, B_visc_spar,
                   M_PTO=0.0, K_PTO=0.0,
                   experiment_id=None, output_folder="EcoData/Power",
                   save_plots=True, save_data=True,
                   omega_nat_used=None, manual_override=False,
                   rho=1025, g=9.81):
    """
    Calculate power absorption with PTO and viscous damping included.
    
    ENHANCEMENT: Now includes phase calculation with PTO for consistency.
    
    Parameters:
    -----------
    A_heave : array_like
        Added mass matrix [ω, 2, 2] - heave only
    B_heave : array_like
        Hydrodynamic radiation damping matrix [ω, 2, 2] - heave only
    M_heave : array_like
        Inertia matrix [2, 2] - heave elements only
    C_heave : array_like
        Hydrostatic stiffness matrix [2, 2] - heave elements only
    Fe_heave : array_like
        Excitation force [ω, 2] - heave only (complex)
    frequencies : array_like
        Frequency array [rad/s]
    B_PTO : float
        PTO damping coefficient [kg/s]
    B_visc_float : float
        Viscous damping for float [kg/s]
    B_visc_spar : float
        Viscous damping for spar [kg/s]
    experiment_id : int
        DOE experiment number
    output_folder : str
        Output directory
    save_plots : bool
        Save plots
    save_data : bool
        Save data files
    omega_nat_used : float
        Natural frequency used for B_visc calculation (for metadata)
    manual_override : bool
        Flag indicating if manual frequency override was used
    rho : float
        Water density [kg/m³]
    g : float
        Gravity [m/s²]
    
    Returns:
    --------
    dict : Power calculation results (ENHANCED with phase)
    """
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    w = np.array(frequencies)
    n_freq = len(w)
    
    # Initialize arrays
    RAO_float_PTO = np.zeros(n_freq, dtype=complex)
    RAO_spar_PTO = np.zeros(n_freq, dtype=complex)
    RAO_relative_PTO = np.zeros(n_freq, dtype=complex)
    P_omega = np.zeros(n_freq)
    
    print(f"\n{'='*70}")
    print(f"Calculating Power for Experiment {experiment_id:03d}")
    print(f"{'='*70}")
    print(f"Frequency range: {w[0]:.2f} - {w[-1]:.2f} rad/s ({n_freq} points)")
    print(f"System: 2×2 (heave Float, heave Spar) with PTO + Viscous damping")
    
    # Construct PTO damping matrix
    B_PTO_matrix = np.array([[ B_PTO, -B_PTO],
                             [-B_PTO,  B_PTO]])
    
    # Construct viscous damping matrix
    B_viscous = np.array([[B_visc_float,      0       ],
                          [     0,        B_visc_spar]])
    
    print(f"\n🔧 Damping Components:")
    print(f"   B_PTO: {B_PTO:.2f} kg/s")
    print(f"   B_visc_float: {B_visc_float:.2f} kg/s")
    print(f"   B_visc_spar: {B_visc_spar:.2f} kg/s")
    if omega_nat_used is not None:
        print(f"   ω_nat used: {omega_nat_used:.3f} rad/s {'(manual override)' if manual_override else '(from RAO)'}")
    
    # Calculate RAOs and Power for each frequency
    print(f"\n⚡ Calculating RAOs with PTO and Power...")
    
    # Store B_total for all frequencies
    B_total_all = np.zeros((n_freq, 2, 2))
    
    for i, omega in enumerate(w):
        # Frequency-dependent matrices
        A = A_heave[i, :, :]  # [2, 2]
        B_hydro = B_heave[i, :, :]  # [2, 2]
        Fe = Fe_heave[i, :]   # [2] complex
        
        # Total damping matrix
        B_total = B_hydro + B_PTO_matrix + B_viscous
        B_total_all[i, :, :] = B_total
        
        # Impedance matrix Z = -ω²(M + A) + iωB_total + C
        Z = -omega**2 * (M_heave + A) + 1j * omega * B_total + C_heave
        
        # Wave amplitude (unit amplitude: ζ_a = 1)
        zeta_a = 1.0
        
        try:
            # Solve for response amplitudes: Z * X = Fe * ζ_a
            X = np.linalg.solve(Z, Fe * zeta_a)  # [2] complex
            
            # RAOs with PTO
            RAO_float_PTO[i] = X[0] / zeta_a
            RAO_spar_PTO[i] = X[1] / zeta_a
            RAO_relative_PTO[i] = (X[0] - X[1]) / zeta_a
            
            # Power calculation: P(ω) = ½ · B_PTO · ω² · |ζ_rel|²
            # where ζ_rel = X[0] - X[1] for unit wave amplitude
            v_rel = omega * (X[0] - X[1])  # Relative velocity
            P_omega[i] = 0.5 * B_PTO * np.abs(v_rel)**2
            
        except np.linalg.LinAlgError:
            print(f"   ⚠️ Singular matrix at ω={omega:.3f} rad/s - setting outputs=0")
            RAO_float_PTO[i] = 0
            RAO_spar_PTO[i] = 0
            RAO_relative_PTO[i] = 0
            P_omega[i] = 0
    
    # Calculate absolute values
    RAO_float_PTO_abs = np.abs(RAO_float_PTO)
    RAO_spar_PTO_abs = np.abs(RAO_spar_PTO)
    RAO_relative_PTO_abs = np.abs(RAO_relative_PTO)
    
    # Find peak values for RAOs
    idx_peak_float = np.argmax(RAO_float_PTO_abs)
    omega_peak_float_PTO = w[idx_peak_float]
    RAO_peak_float_PTO = RAO_float_PTO_abs[idx_peak_float]
    
    idx_peak_spar = np.argmax(RAO_spar_PTO_abs)
    omega_peak_spar_PTO = w[idx_peak_spar]
    RAO_peak_spar_PTO = RAO_spar_PTO_abs[idx_peak_spar]
    
    idx_peak_relative = np.argmax(RAO_relative_PTO_abs)
    omega_peak_relative_PTO = w[idx_peak_relative]
    RAO_peak_relative_PTO = RAO_relative_PTO_abs[idx_peak_relative]
    
    # Find peak power
    idx_P_peak = np.argmax(P_omega)
    P_peak = P_omega[idx_P_peak]
    omega_P_peak = w[idx_P_peak]
    
    print(f"\n📊 Peak RAO Values (with PTO):")
    print(f"   Float:    ω_peak={omega_peak_float_PTO:.3f} rad/s, |RAO|_peak={RAO_peak_float_PTO:.3f} m/m")
    print(f"   Spar:     ω_peak={omega_peak_spar_PTO:.3f} rad/s, |RAO|_peak={RAO_peak_spar_PTO:.3f} m/m")
    print(f"   Relative: ω_peak={omega_peak_relative_PTO:.3f} rad/s, |RAO|_peak={RAO_peak_relative_PTO:.3f} m/m")
    
    print(f"\n⚡ Peak Power:")
    print(f"   P_peak = {P_peak:.2f} W at ω = {omega_P_peak:.3f} rad/s")
    
    #%%========================================================================
    # NEW ENHANCEMENT: Calculate phase with PTO
    #%%========================================================================
    
    print(f"\n📐 NEW: Calculating phase with PTO...")
    
    # Calculate phase from complex RAO with PTO
    phase_relative_rad = np.angle(RAO_relative_PTO)  # [-π, π]
    phase_relative_deg = np.degrees(phase_relative_rad)  # [-180°, 180°]
    
    print(f"   Phase range: [{np.min(phase_relative_deg):.1f}°, {np.max(phase_relative_deg):.1f}°]")
    print(f"   ✅ Phase calculation with PTO completed")
    
    # Prepare results dictionary
    results = {
        # RAOs with PTO (complex)
        'RAO_with_PTO_float': RAO_float_PTO,
        'RAO_with_PTO_spar': RAO_spar_PTO,
        'RAO_with_PTO_relative': RAO_relative_PTO,
        # RAOs with PTO (absolute)
        'RAO_with_PTO_float_abs': RAO_float_PTO_abs,
        'RAO_with_PTO_spar_abs': RAO_spar_PTO_abs,
        'RAO_with_PTO_relative_abs': RAO_relative_PTO_abs,
        # Peak RAO information
        'omega_peak_float_PTO': omega_peak_float_PTO,
        'RAO_peak_float_PTO': RAO_peak_float_PTO,
        'omega_peak_spar_PTO': omega_peak_spar_PTO,
        'RAO_peak_spar_PTO': RAO_peak_spar_PTO,
        'omega_peak_relative_PTO': omega_peak_relative_PTO,
        'RAO_peak_relative_PTO': RAO_peak_relative_PTO,
        # Power
        'P_omega': P_omega,
        'P_peak': P_peak,
        'omega_P_peak': omega_P_peak,
        # NEW: Phase with PTO
        'phase_relative_rad': phase_relative_rad,
        'phase_relative_deg': phase_relative_deg,
        # Damping values used
        'B_PTO': B_PTO,
        'B_visc_float': B_visc_float,
        'B_visc_spar': B_visc_spar,
        'omega_nat_floater_used': omega_nat_used,
        'manual_override': manual_override,
        # System matrices
        'B_total': B_total_all,
        'frequencies': w,
        # Metadata
        'experiment_id': experiment_id,
        'n_frequencies': n_freq,
        'rho': rho,
        'g': g
    }
    
    # Save data files
    if save_data:
        print(f"\n💾 Saving Power data files...")
        
        # Save as NPZ
        try:
            npz_path = os.path.join(output_folder, f"DOE_Exp_{experiment_id:03d}_Power.npz")
            np.savez_compressed(
                npz_path,
                RAO_with_PTO_float=RAO_float_PTO,
                RAO_with_PTO_spar=RAO_spar_PTO,
                RAO_with_PTO_relative=RAO_relative_PTO,
                RAO_with_PTO_float_abs=RAO_float_PTO_abs,
                RAO_with_PTO_spar_abs=RAO_spar_PTO_abs,
                RAO_with_PTO_relative_abs=RAO_relative_PTO_abs,
                omega_peak_float_PTO=omega_peak_float_PTO,
                RAO_peak_float_PTO=RAO_peak_float_PTO,
                omega_peak_spar_PTO=omega_peak_spar_PTO,
                RAO_peak_spar_PTO=RAO_peak_spar_PTO,
                omega_peak_relative_PTO=omega_peak_relative_PTO,
                RAO_peak_relative_PTO=RAO_peak_relative_PTO,
                P_omega=P_omega,
                P_peak=P_peak,
                omega_P_peak=omega_P_peak,
                # NEW: Phase fields
                phase_relative_rad=phase_relative_rad,
                phase_relative_deg=phase_relative_deg,
                B_PTO=B_PTO,
                B_visc_float=B_visc_float,
                B_visc_spar=B_visc_spar,
                omega_nat_floater_used=omega_nat_used,
                manual_override=manual_override,
                B_total=B_total_all,
                frequencies=w,
                experiment_id=experiment_id
            )
            print(f"   ✅ NPZ saved: DOE_Exp_{experiment_id:03d}_Power.npz")
        except Exception as e:
            print(f"   ⚠️ Error saving NPZ: {e}")
        
        # Save as PKL
        try:
            import pickle
            pkl_path = os.path.join(output_folder, f"DOE_Exp_{experiment_id:03d}_Power.pkl")
            with open(pkl_path, 'wb') as f:
                pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"   ✅ PKL saved: DOE_Exp_{experiment_id:03d}_Power.pkl")
        except Exception as e:
            print(f"   ⚠️ Error saving PKL: {e}")
        
        # Save as MAT
        try:
            from scipy.io import savemat
            mat_path = os.path.join(output_folder, f"DOE_Exp_{experiment_id:03d}_Power.mat")
            mat_data = {
                'RAO_with_PTO_float_real': RAO_float_PTO.real,
                'RAO_with_PTO_float_imag': RAO_float_PTO.imag,
                'RAO_with_PTO_spar_real': RAO_spar_PTO.real,
                'RAO_with_PTO_spar_imag': RAO_spar_PTO.imag,
                'RAO_with_PTO_relative_real': RAO_relative_PTO.real,
                'RAO_with_PTO_relative_imag': RAO_relative_PTO.imag,
                'RAO_with_PTO_float_abs': RAO_float_PTO_abs,
                'RAO_with_PTO_spar_abs': RAO_spar_PTO_abs,
                'RAO_with_PTO_relative_abs': RAO_relative_PTO_abs,
                'omega_peak_float_PTO': omega_peak_float_PTO,
                'RAO_peak_float_PTO': RAO_peak_float_PTO,
                'omega_peak_spar_PTO': omega_peak_spar_PTO,
                'RAO_peak_spar_PTO': RAO_peak_spar_PTO,
                'omega_peak_relative_PTO': omega_peak_relative_PTO,
                'RAO_peak_relative_PTO': RAO_peak_relative_PTO,
                'P_omega': P_omega,
                'P_peak': P_peak,
                'omega_P_peak': omega_P_peak,
                # NEW: Phase fields
                'phase_relative_rad': phase_relative_rad,
                'phase_relative_deg': phase_relative_deg,
                'B_PTO': B_PTO,
                'B_visc_float': B_visc_float,
                'B_visc_spar': B_visc_spar,
                'omega_nat_floater_used': omega_nat_used if omega_nat_used else 0,
                'manual_override': int(manual_override),
                'B_total': B_total_all,
                'frequencies': w,
                'experiment_id': experiment_id
            }
            savemat(mat_path, mat_data, do_compression=True)
            print(f"   ✅ MAT saved: DOE_Exp_{experiment_id:03d}_Power.mat")
        except Exception as e:
            print(f"   ⚠️ Error saving MAT: {e}")
    
    # Generate plots
    if save_plots:
        print(f"\n📊 Generating plots...")
        
        # Plot 1: RAOs with PTO
        fig1, ax1 = plt.subplots(figsize=(12, 7))
        
        ax1.plot(w, RAO_float_PTO_abs, 'b-', linewidth=2, label='RAO Float (with PTO)')
        ax1.plot(w, RAO_spar_PTO_abs, 'r-', linewidth=2, label='RAO Spar (with PTO)')
        ax1.plot(w, RAO_relative_PTO_abs, 'g-', linewidth=2, label='RAO Relative (with PTO)')
        
        # Mark peaks
        ax1.plot(omega_peak_float_PTO, RAO_peak_float_PTO, 'bo', markersize=8,
                label=f'Peak Float: ω={omega_peak_float_PTO:.2f} rad/s')
        ax1.plot(omega_peak_spar_PTO, RAO_peak_spar_PTO, 'ro', markersize=8,
                label=f'Peak Spar: ω={omega_peak_spar_PTO:.2f} rad/s')
        ax1.plot(omega_peak_relative_PTO, RAO_peak_relative_PTO, 'go', markersize=8,
                label=f'Peak Relative: ω={omega_peak_relative_PTO:.2f} rad/s')
        
        ax1.set_xlabel('Wave Frequency [rad/s]', fontsize=12)
        ax1.set_ylabel('|RAO| [m/m]', fontsize=12)
        ax1.set_title(f'Heave RAOs with PTO - DOE Experiment {experiment_id:03d}', 
                     fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best', fontsize=10)
        ax1.set_xlim([w[0], w[-1]])
        
        plt.tight_layout()
        
        plot1_path = os.path.join(output_folder, f"DOE_Exp_{experiment_id:03d}_Power_RAO.png")
        plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
        print(f"   ✅ RAO plot saved: DOE_Exp_{experiment_id:03d}_Power_RAO.png")
        plt.close()
        
        # Plot 2: Power
        fig2, ax2 = plt.subplots(figsize=(12, 7))
        
        ax2.plot(w, P_omega, 'b-', linewidth=2, label='Power P(ω)')
        ax2.plot(omega_P_peak, P_peak, 'ro', markersize=10,
                label=f'Peak Power: {P_peak:.2f} W at ω={omega_P_peak:.2f} rad/s')
        
        ax2.set_xlabel('Wave Frequency [rad/s]', fontsize=12)
        ax2.set_ylabel('Power [W]', fontsize=12)
        ax2.set_title(f'Power Absorption - DOE Experiment {experiment_id:03d}\n' +
                     f'B_PTO={B_PTO:.0f} kg/s, B_visc_spar={B_visc_spar:.2f} kg/s',
                     fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='best', fontsize=11)
        ax2.set_xlim([w[0], w[-1]])
        
        plt.tight_layout()
        
        plot2_path = os.path.join(output_folder, f"DOE_Exp_{experiment_id:03d}_Power_Power.png")
        plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
        print(f"   ✅ Power plot saved: DOE_Exp_{experiment_id:03d}_Power_Power.png")
        plt.close()
    
    print(f"✅ Enhanced Power+Phase calculation completed for Experiment {experiment_id:03d}")
    
    return results


if __name__ == "__main__":
    print("This module should be imported, not run directly.")
    print("Use St3_Power.py to process DOE experiments.")
