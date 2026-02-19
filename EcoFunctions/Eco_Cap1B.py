"""
Single-body hydrodynamic analysis with Capytaine and NetCDF export
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

def analyze_single_body_hydrodynamics(mesh_path, frequency_range, 
                                    mesh_position=[0.0, 0.0, 0.0], 
                                    wave_direction=0.0,
                                    output_directory="hydroData",
                                    nc_filename="single_body.nc",
                                    body_name="Body",
                                    plot_xlim=[-20, 20],
                                    plot_ylim=[-35, 15],
                                    save_plots=True,
                                    show_plots=True,
                                    logging_level="INFO"):
    """
    Performs hydrodynamic analysis for a single floating body.
    Returns only hydrodynamic coefficients (A, B, Fe, M, C) without RAO calculations.
    
    Parameters:
    -----------
    mesh_path : str
        STL file path for the body
    frequency_range : array_like
        Frequency range to analyze [rad/s]
    mesh_position : list
        Body position [x, y, z]
    wave_direction : float
        Incident wave direction [rad]
    output_directory : str
        Output directory
    nc_filename : str
        NetCDF filename
    body_name : str
        Body name
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
        - Npan: Panel count
        - body: Body object
    """
    
    # Disable Capytaine progress display to avoid LiveError in loops
    import logging
    logging.getLogger('capytaine').setLevel(logging.CRITICAL)
    
    cpt.set_logging(level=logging_level)
    
    # Validate STL file
    if not os.path.exists(mesh_path):
        raise FileNotFoundError(f"STL file not found: {mesh_path}")
    
    # Load and configure body
    mesh = cpt.load_mesh(mesh_path)
    mesh.translate(mesh_position)
    body = cpt.FloatingBody(
        mesh=mesh,
        dofs=cpt.rigid_body_dofs(rotation_center=mesh.center_of_buoyancy),
        center_of_mass=mesh.center_of_buoyancy,
        name=body_name.lower()
    )
    
    # CRITICAL: Use immersed_part() for both M and C calculations
    # This ensures M represents displaced mass (Archimedes principle)
    # M = ρ_water × V_submerged (only z < 0)
    body.inertia_matrix = body.immersed_part().compute_rigid_body_inertia()
    body.hydrostatic_stiffness = body.immersed_part().compute_hydrostatic_stiffness()
    
    # Create BEM problems
    w = np.array(frequency_range)
    problems = []
    
    for omega in w:
        # Radiation problems for each DOF
        for dof in body.dofs:
            problems.append(cpt.RadiationProblem(body=body, radiating_dof=dof, omega=omega))
        # Diffraction problem
        problems.append(cpt.DiffractionProblem(body=body, omega=omega, wave_direction=wave_direction))
    
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
    M = body.inertia_matrix
    C = body.hydrostatic_stiffness
    
    # Get number of panels for the body
    Npan = len(mesh.faces)
    
    # Create output directory
    os.makedirs(output_directory, exist_ok=True)
    
    # Save ONLY hydrodynamic coefficients (no RAO)
    hydro_coeffs = {
        'A': A,                          # Added mass
        'B': B,                          # Radiation damping
        'Fe': Fe,                        # Excitation force
        'Ffk': Ffk,                      # Froude-Krylov force
        'M': M,                          # Inertia matrix
        'C': C,                          # Hydrostatic stiffness matrix
        'Npan': Npan,                    # Number of panels in the body
        'w': w,                          # Frequencies
        'metadata': {
            'body_name': body_name,
            'wave_direction': wave_direction,
            'mesh_position': mesh_position,
            'frequency_range': [float(w[0]), float(w[-1])],
            'num_frequencies': int(len(w)),
            'Npan': int(Npan),
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
        plt.plot(w, B[:, 0, 0], label='Surge damping')
        plt.plot(w, B[:, 1, 1], label='Sway damping')
        plt.plot(w, B[:, 2, 2], label='Heave damping')
        plt.plot(w, B[:, 3, 3], label='Roll damping')
        plt.plot(w, B[:, 4, 4], label='Pitch damping')
        plt.plot(w, B[:, 5, 5], label='Yaw damping')
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
        plt.plot(w, A[:, 0, 0], label='Surge added mass')
        plt.plot(w, A[:, 1, 1], label='Sway added mass')
        plt.plot(w, A[:, 2, 2], label='Heave added mass')
        plt.plot(w, A[:, 3, 3], label='Roll added mass')
        plt.plot(w, A[:, 4, 4], label='Pitch added mass')
        plt.plot(w, A[:, 5, 5], label='Yaw added mass')
        plt.title("Added Mass Coefficients")
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
        
        # Plot mesh - XZ projection
        verts = mesh.vertices
        faces = mesh.faces
        for face in faces:
            face_verts = [verts[idx] for idx in face]
            x_coords = [v[0] for v in face_verts] + [face_verts[0][0]]
            z_coords = [v[2] for v in face_verts] + [face_verts[0][2]]
            ax.fill(x_coords, z_coords, facecolor='skyblue', edgecolor='k', 
                   linewidth=0.1, alpha=0.9)
        
        ax.set_xlim(plot_xlim)
        ax.set_ylim(plot_ylim)
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Z [m]")
        ax.set_title(f"Lateral view (XZ plane): {body_name}")
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
    print(f"   - Inertia matrix (M): {M.shape} - diagonal elements: {np.diag(M)}")
    print(f"   - Hydrostatic stiffness (C): {C.shape} - diagonal elements: {np.diag(C)}")
    print(f"   - Number of panels {body_name} (Npan): {Npan}")
    
    return {
        'dataset': dataset,
        'frequencies': w,
        'body': body,
        'added_mass': A,
        'radiation_damping': B,
        'excitation_force': Fe,
        'froude_krylov_force': Ffk,
        'inertia_matrix': M,
        'hydrostatic_stiffness': C,
        'Npan': Npan,
        'hydro_coeffs': hydro_coeffs
    }


if __name__ == "__main__":
    frequencies = np.linspace(0.1, 2.0, 20)
    
    results = analyze_single_body_hydrodynamics(
        mesh_path="geometry/float.stl",
        frequency_range=frequencies,
        mesh_position=[0.0, 0.0, 0.0],
        body_name="Float",
        output_directory="hydroData",
        nc_filename="single_body.nc"
    )
    
    A = results['added_mass']
    B = results['radiation_damping']
    Fe = results['excitation_force']
    M = results['inertia_matrix']
    C = results['hydrostatic_stiffness']
    frequencies = results['frequencies']
    Npan = results['Npan']
    
    print(f"\nðŸ“Š Analysis Summary:")
    print(f"   - Frequencies analyzed: {len(frequencies)}")
    print(f"   - Added mass shape: {A.shape}")
    print(f"   - Radiation damping shape: {B.shape}")
    print(f"   - Excitation force shape: {Fe.shape}")
    print(f"   - Inertia matrix shape: {M.shape}")
    print(f"   - Hydrostatic stiffness shape: {C.shape}")
    print(f"   - Panel count: {Npan}")