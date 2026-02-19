"""
Two-body hydrodynamic analysis with Capytaine and NetCDF export
Author: Pablo Antonio Matamala Carvajal
Date: 2025-01-21
Updated: 2025-02-10 - CORRECTED: M calculated using immersed_part() only
         Ensures M represents displaced mass (Archimedes principle)

CRITICAL FIX: Inertia matrix M now uses only submerged volume (z < 0)
              M = ρ_water × V_submerged
              Previous: M = ρ_assumed × V_total (incorrect for floating bodies)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import capytaine as cpt

def analyze_two_body_hydrodynamics(mesh1_path, mesh2_path, frequency_range, 
                                 mesh1_position=[0.0, 0.0, 0.0], 
                                 mesh2_position=[0.0, 0.0, 0.0],
                                 wave_direction=0.0,
                                 output_directory="hydroData",
                                 nc_filename="rm3.nc",
                                 body_names=["Float", "Plate"],
                                 plot_xlim=[-20, 20],
                                 plot_ylim=[-35, 15],
                                 save_plots=True,
                                 show_plots=True,
                                 logging_level="INFO"):
    """
    Performs hydrodynamic analysis for two floating bodies with coupled interaction.
    Returns only hydrodynamic coefficients (A, B, Fe, M, C) without RAO calculations.
    
    Parameters:
    -----------
    mesh1_path : str
        STL file path for first body
    mesh2_path : str
        STL file path for second body
    frequency_range : array_like
        Frequency range to analyze [rad/s]
    mesh1_position : list
        First body position [x, y, z]
    mesh2_position : list
        Second body position [x, y, z]
    wave_direction : float
        Incident wave direction [rad]
    output_directory : str
        Output directory
    nc_filename : str
        NetCDF filename
    body_names : list
        Body names
    save_plots : bool
        Save plots
    show_plots : bool
        Show plots
    logging_level : str
        Capytaine logging level
    
    Returns:
    --------
    dict : Analysis results containing:
        - dataset: Capytaine xarray dataset
        - frequencies: Frequency array
        - added_mass: Added mass matrices
        - radiation_damping: Radiation damping matrices
        - excitation_force: Total excitation force
        - froude_krylov_force: Froude-Krylov force
        - inertia_matrix: Inertia matrix
        - hydrostatic_stiffness: Hydrostatic stiffness matrix
        - Npan1, Npan2: Panel counts
        - bodies: Combined body object
    """
    
    # Disable Capytaine progress display to avoid LiveError in loops
    import logging
    logging.getLogger('capytaine').setLevel(logging.CRITICAL)
    
    cpt.set_logging(level=logging_level)
    
    # Validate STL files
    if not os.path.exists(mesh1_path):
        raise FileNotFoundError(f"STL file not found: {mesh1_path}")
    if not os.path.exists(mesh2_path):
        raise FileNotFoundError(f"STL file not found: {mesh2_path}")
    
    # Create output directory FIRST (before any calculations)
    os.makedirs(output_directory, exist_ok=True)
    print(f"ðŸ“ Output directory created/verified: {output_directory}")
    
    # Load and configure first body
    mesh1 = cpt.load_mesh(mesh1_path)
    mesh1.translate(mesh1_position)
    body1 = cpt.FloatingBody(
        mesh=mesh1,
        dofs=cpt.rigid_body_dofs(rotation_center=mesh1.center_of_buoyancy),
        center_of_mass=mesh1.center_of_buoyancy,
        name=body_names[0].lower()
    )
    
    # CRITICAL: Use immersed_part() for both M and C calculations
    # This ensures M represents displaced mass (Archimedes principle)
    # M = ρ_water × V_submerged (only z < 0)
    body1.inertia_matrix = body1.immersed_part().compute_rigid_body_inertia()
    body1.hydrostatic_stiffness = body1.immersed_part().compute_hydrostatic_stiffness()
    
    # Load and configure second body
    mesh2 = cpt.load_mesh(mesh2_path)
    mesh2.translate(mesh2_position)
    body2 = cpt.FloatingBody(
        mesh=mesh2,
        dofs=cpt.rigid_body_dofs(rotation_center=mesh2.center_of_buoyancy),
        center_of_mass=mesh2.center_of_buoyancy,
        name=body_names[1].lower()
    )
    
    # CRITICAL: Use immersed_part() for both M and C calculations
    # This ensures M represents displaced mass (Archimedes principle)
    # M = ρ_water × V_submerged (only z < 0)
    body2.inertia_matrix = body2.immersed_part().compute_rigid_body_inertia()
    body2.hydrostatic_stiffness = body2.immersed_part().compute_hydrostatic_stiffness()
    
    # Combine bodies
    all_bodies = body1 + body2
    
    # Create BEM problems
    w = np.array(frequency_range)
    problems = []
    
    for omega in w:
        # Radiation problems for each DOF
        for dof in all_bodies.dofs:
            problems.append(cpt.RadiationProblem(body=all_bodies, radiating_dof=dof, omega=omega))
        # Diffraction problem
        problems.append(cpt.DiffractionProblem(body=all_bodies, omega=omega, wave_direction=wave_direction))
    
    # Solve BEM problems
    solver = cpt.BEMSolver()
    results = solver.solve_all(problems)
    dataset = cpt.assemble_dataset(results)
    
    # Extract hydrodynamic coefficients
    A = dataset.added_mass.values
    B = dataset.radiation_damping.values
    Ffk = dataset.Froude_Krylov_force.values[:, 0, :].T
    Fe = dataset.excitation_force.values[:, 0, :].T
    
    # Get inertia and hydrostatic stiffness matrices
    M = all_bodies.inertia_matrix
    C = all_bodies.hydrostatic_stiffness
    
    # Get number of panels for each body
    Npan1 = len(mesh1.faces)
    Npan2 = len(mesh2.faces)
    
    # Save ONLY hydrodynamic coefficients (no RAO)
    hydro_coeffs = {
        'A': A,                          # Added mass
        'B': B,                          # Radiation damping
        'Fe': Fe,                        # Excitation force
        'Ffk': Ffk,                      # Froude-Krylov force
        'M': M,                          # Inertia matrix
        'C': C,                          # Hydrostatic stiffness matrix
        'Npan1': Npan1,                  # Number of panels in first body
        'Npan2': Npan2,                  # Number of panels in second body
        'w': w,                          # Frequencies
        'metadata': {
            'body_names': body_names,
            'wave_direction': wave_direction,
            'mesh1_position': mesh1_position,
            'mesh2_position': mesh2_position,
            'frequency_range': [float(w[0]), float(w[-1])],
            'num_frequencies': int(len(w)),
            'Npan1': int(Npan1),
            'Npan2': int(Npan2),
            'total_panels': int(Npan1 + Npan2),
            'description': 'Hydrodynamic coefficients from Capytaine BEM analysis (A, B, Fe, M, C)'
        }
    }
    
    # Save hydrodynamic coefficients in multiple formats
    files_created = []
    
    # Save as NPZ (NumPy compressed)
    try:
        npz_path = os.path.join(output_directory, "HydCoeff.npz")
        np_data = {k: v for k, v in hydro_coeffs.items() if k != 'metadata' and isinstance(v, np.ndarray)}
        np_data['w'] = w
        np.savez_compressed(npz_path, **np_data)
        files_created.append(npz_path)
        print(f"âœ… HydCoeff.npz saved successfully")
    except Exception as e:
        print(f"Error saving NPZ: {e}")
    
    # Save as Pickle
    try:
        import pickle
        pkl_path = os.path.join(output_directory, "HydCoeff.pkl")
        with open(pkl_path, 'wb') as f:
            pickle.dump(hydro_coeffs, f, protocol=pickle.HIGHEST_PROTOCOL)
        files_created.append(pkl_path)
        print(f"âœ… HydCoeff.pkl saved successfully")
    except Exception as e:
        print(f"Error saving PKL: {e}")
    
    # Save as MATLAB .mat file
    try:
        from scipy.io import savemat
        mat_path = os.path.join(output_directory, "HydCoeff.mat")
        matlab_data = {}
        for key, value in hydro_coeffs.items():
            if key == 'metadata':
                for meta_key, meta_value in value.items():
                    matlab_data[f"meta_{meta_key}"] = meta_value
            elif isinstance(value, np.ndarray) and np.iscomplexobj(value):
                matlab_data[f"{key}_real"] = value.real
                matlab_data[f"{key}_imag"] = value.imag
            elif isinstance(value, np.ndarray):
                matlab_data[key] = value
            else:
                matlab_data[key] = value
        
        savemat(mat_path, matlab_data, do_compression=True)
        files_created.append(mat_path)
        print(f"âœ… HydCoeff.mat saved successfully")
    except Exception as e:
        print(f"Error saving MAT: {e}")
    
    # Export to NetCDF
    nc_filename_final = nc_filename.replace('.h5', '.nc') if nc_filename.endswith('.h5') else nc_filename
    nc_path = os.path.join(output_directory, nc_filename_final)
    
    try:
        try:
            dataset.to_netcdf(nc_path, engine='netcdf4')
        except (ImportError, ValueError):
            try:
                dataset.to_netcdf(nc_path, engine='h5netcdf')
            except (ImportError, ValueError):
                # Separate complex variables for compatibility
                dataset_real = dataset.copy(deep=True)
                for var_name in list(dataset.data_vars.keys()):
                    var_data = dataset[var_name]
                    if np.iscomplexobj(var_data.values):
                        dataset_real[f"{var_name}_real"] = var_data.real.astype(np.float64)
                        dataset_real[f"{var_name}_imag"] = var_data.imag.astype(np.float64)
                        dataset_real = dataset_real.drop_vars(var_name)
                
                dataset_real.to_netcdf(nc_path)
        
        files_created.append(nc_path)
        print(f"âœ… NetCDF file saved: {nc_path}")
    except Exception as e:
        print(f"Error exporting NetCDF: {e}")
    
    # Generate plots (without RAO)
    if save_plots or show_plots:
        # Radiation Damping plot
        plt.figure(figsize=(10, 6))
        plt.plot(w, B[:, 2, 2], label=f'Heave damping {body_names[0]}')
        plt.plot(w, B[:, 8, 8], label=f'Heave damping {body_names[1]}')
        plt.title("Radiation Damping Coefficients (diagonal terms)")
        plt.xlabel("Wave frequency [rad/s]")
        plt.ylabel("Damping [kg/s or kgÂ·mÂ²/s]")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        
        if save_plots:
            damping_plot_path = os.path.join(output_directory, "radiation_damping_coefficients.png")
            plt.savefig(damping_plot_path, dpi=300, bbox_inches='tight')
            print(f"âœ… Radiation damping plot saved: {damping_plot_path}")
        
        if show_plots:
            plt.show()
        else:
            plt.close()
        
        # Added Mass plot
        plt.figure(figsize=(10, 6))
        plt.plot(w, A[:, 2, 2], label=f'Heave added mass {body_names[0]}')
        plt.plot(w, A[:, 8, 8], label=f'Heave added mass {body_names[1]}')
        plt.title("Added Mass Coefficients (kg)")
        plt.xlabel("Wave frequency [rad/s]")
        plt.ylabel("Added Mass [kg or kgÂ·mÂ²]")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        
        if save_plots:
            added_mass_plot_path = os.path.join(output_directory, "added_mass_coefficients.png")
            plt.savefig(added_mass_plot_path, dpi=300, bbox_inches='tight')
            print(f"âœ… Added mass plot saved: {added_mass_plot_path}")
        
        if show_plots:
            plt.show()
        else:
            plt.close()
        
        # Geometry plot
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111)
        
        # Plot mesh1 (Float) - XZ projection
        verts1 = mesh1.vertices
        faces1 = mesh1.faces
        for face in faces1:
            face_verts = [verts1[idx] for idx in face]
            x_coords = [v[0] for v in face_verts] + [face_verts[0][0]]
            z_coords = [v[2] for v in face_verts] + [face_verts[0][2]]
            ax.fill(x_coords, z_coords, facecolor='skyblue', edgecolor='k', 
                   linewidth=0.1, alpha=0.9)
        
        # Plot mesh2 (Plate) - XZ projection
        verts2 = mesh2.vertices
        faces2 = mesh2.faces
        for face in faces2:
            face_verts = [verts2[idx] for idx in face]
            x_coords = [v[0] for v in face_verts] + [face_verts[0][0]]
            z_coords = [v[2] for v in face_verts] + [face_verts[0][2]]
            ax.fill(x_coords, z_coords, facecolor='lightcoral', edgecolor='k', 
                   linewidth=0.1, alpha=0.9)
        
        ax.set_xlim(plot_xlim)
        ax.set_ylim(plot_ylim)
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Z [m]")
        ax.set_title(f"Lateral view (XZ plane): {body_names[0]} and {body_names[1]}")
        ax.set_aspect('equal')
        ax.grid(True)
        plt.tight_layout()
        
        if save_plots:
            geo_plot_path = os.path.join(output_directory, "geometry_lateral_view.png")
            plt.savefig(geo_plot_path, dpi=300, bbox_inches='tight')
            print(f"âœ… Geometry plot saved: {geo_plot_path}")
        
        if show_plots:
            plt.show()
        else:
            plt.close()
    
    print(f"\nðŸ—‚ Files created: {len(files_created)}")
    for file_path in files_created:
        print(f"   - {file_path}")
    
    # Print matrix information for verification
    print(f"\nðŸ“Š Matrix and Mesh Information:")
    print(f"   - Inertia matrix (M): {M.shape} - diagonal elements: {np.diag(M)[:6]}")
    print(f"   - Hydrostatic stiffness (C): {C.shape} - diagonal elements: {np.diag(C)[:6]}")
    print(f"   - Number of panels {body_names[0]} (Npan1): {Npan1}")
    print(f"   - Number of panels {body_names[1]} (Npan2): {Npan2}")
    print(f"   - Total panels: {Npan1 + Npan2}")
    
    return {
        'dataset': dataset,
        'frequencies': w,
        'bodies': all_bodies,
        'added_mass': A,
        'radiation_damping': B,
        'excitation_force': Fe,
        'froude_krylov_force': Ffk,
        'inertia_matrix': M,
        'hydrostatic_stiffness': C,
        'Npan1': Npan1,
        'Npan2': Npan2,
        'hydro_coeffs': hydro_coeffs
    }


if __name__ == "__main__":
    frequencies = np.linspace(0.1, 2.0, 20)
    
    results = analyze_two_body_hydrodynamics(
        mesh1_path="geometry/float.stl",
        mesh2_path="geometry/plate.stl",
        frequency_range=frequencies,
        mesh1_position=[0.0, 0.0, 0.0],
        mesh2_position=[0.0, 0.0, -20.0],
        body_names=["Float", "Plate"],
        output_directory="hydroData",
        nc_filename="rm3.nc"
    )
    
    A = results['added_mass']
    B = results['radiation_damping']
    Fe = results['excitation_force']
    M = results['inertia_matrix']
    C = results['hydrostatic_stiffness']
    frequencies = results['frequencies']
    Npan1 = results['Npan1']
    Npan2 = results['Npan2']
    
    print(f"\nðŸ“Š Analysis Summary:")
    print(f"   - Frequencies analyzed: {len(frequencies)}")
    print(f"   - Added mass shape: {A.shape}")
    print(f"   - Radiation damping shape: {B.shape}")
    print(f"   - Excitation force shape: {Fe.shape}")
    print(f"   - Inertia matrix shape: {M.shape}")
    print(f"   - Hydrostatic stiffness shape: {C.shape}")
    print(f"   - Panel count Float: {Npan1}")
    print(f"   - Panel count Plate: {Npan2}")
    print(f"   - Total panels: {Npan1 + Npan2}")
