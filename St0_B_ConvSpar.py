#%%     MAIN SCRIPT FOR EXECUTION OF ECO FUNCTIONS
import numpy as np
import math
import shutil
import os
import sys
import matplotlib.pyplot as plt

# Add EcoFunctions to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'EcoFunctions'))

from EcoFunctions.Eco_StlRev import generate_revolution_solid_stl
from EcoFunctions.Eco_Cap2B import analyze_two_body_hydrodynamics
from EcoFunctions.Eco_Cap1B import analyze_single_body_hydrodynamics

#%%     MAIN CONFIGURATION
# ==================
FOLDER_NAME = "EcoData/Convergence/MeshAnalysis/batch_ConvSpar"  # Main results folder name
os.makedirs(FOLDER_NAME, exist_ok=True)
print(f"\nMain batch folder: {FOLDER_NAME}")

#%%     PARAMETERS
#frequencies = np.arange(0.2, 8.2 + 0.05, 0.05)  # [rad/s]
frequencies = np.arange(0.2, 10 + 0.05, 0.05)  # [rad/s]
#frequencies = np.arange(0.2, 0.3 + 0.5, 0.5)  # [rad/s]


# STL Generation Parameters
NUM_SEGMENTS = np.array([30, 40, 50, 60])         # Revolution segments (circumferential)
Z_SUBDIVISIONS = np.array([30, 35, 40, 45])        # Z-axis subdivisions per segment (configurable mesh density in Z direction)
HEIGHT_THRESHOLD = 0.06
MIN_SUBDIVISIONS = 5

RAD_SUBDIVISION = np.array([4, 4, 4, 4, 4])

#%%  HYDRODYNAMICAL ANALYSIS
for i, (NS, ZS, RS) in enumerate(zip(NUM_SEGMENTS, Z_SUBDIVISIONS, RAD_SUBDIVISION)):
    # Create folder name with R value
    folder_name = f"MESH_Seg{NS}_Zsub{ZS}"
    folder_path = os.path.join(FOLDER_NAME, folder_name)
     
    # Remove existing folder
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        print(f"Existing folder {folder_path} removed")
    
    # Define geometry points with current R and D values
    F1 = 0.2
    F2 = 0.05
    F3 = 0.15
    F4 = 0.4
    
    D1 = 0.8
    D2 = 0.1
    D3 = 10
    D4 = 1.4
    D5 = 0.8
    D6 = 1
    
    P = np.array([
        [0, 0, -D4],                                      # P1 - Uses D4
        *[[i*D5/(2*RS), 0, -D4] for i in range(1, RS+1)],  # P2 ida - Uses D5 and D4
        [RS*D5/(2*RS), 0, -D4+F2],                         # P3 - Uses D5 and D4
        *[[i*D5/(2*RS), 0, -D4+F2] for i in range(RS-1, 0, -1) if i*D5/(2*RS) >= 1.8*F1/2],  # P2 reversa con condición
        [F1/2, 0, -D4+F2],                               # P4 - Uses D4
        [F1/2, 0, F4],                                   # P5 - Uses D6
        [0, 0, F4],                                      # P6 - Uses D6
    ])
    
    # Create geometry folder
    geometry_path = os.path.join(folder_path, "geometry")
    os.makedirs(geometry_path, exist_ok=True)
    
    # Remove existing float.stl
    body_stl_path = os.path.join(geometry_path, "body.stl")
    if os.path.exists(body_stl_path):
        try:
            os.remove(body_stl_path)
            print(f"File {body_stl_path} removed")
        except PermissionError:
            print(f"Could not remove {body_stl_path} - file in use")
    
    # Change directory temporarily to save STL
    current_dir = os.getcwd()
    os.chdir(geometry_path)
    
    try:
        # Generate FLOAT revolution solid
        print("\n🔄 Generating geometry...")
        result_body = generate_revolution_solid_stl(
            points=P,
            filename="body.stl",
            num_segments=NS,
            z_subdivisions=ZS,
            visualize=False,
            save_plot_path=os.getcwd(),
            plot_filename="body_profile_plot.png",
            height_threshold=HEIGHT_THRESHOLD,
            min_subdivisions=MIN_SUBDIVISIONS
        )
        
        print(f"✅ STL generated: {result_body['filename']}")
        print(f"   Vertices: {result_body['num_vertices']:,}")
        print(f"   Triangles: {result_body['num_triangles']:,}")
        
    except Exception as e:
        print(f"⚠ Error generating STL: {e}")
        os.chdir(current_dir)
        continue
    
    # Return to original directory
    os.chdir(current_dir)
    
    # Run hydrodynamic analysis
    try:
        print("🌊 Starting hydrodynamic analysis...")
        
        # Define mesh paths
        mesh1_path = os.path.join(folder_path, "geometry", "body.stl")

        # Verify both files exist
        if not os.path.exists(mesh1_path):
            print(f"⚠ Error: {mesh1_path} not found")
            continue

        
        # Output directory for hydrodynamic data
        hydro_output_dir = os.path.join(folder_path, "hydroData")
        
        # Run hydrodynamic analysis
        results = analyze_single_body_hydrodynamics(
            mesh_path=mesh1_path,  # Solo necesitas un mesh path
            frequency_range=frequencies,
            mesh_position=[0.0, 0.0, 0.0],  # Solo una posición
            body_name="Spar",  # Solo un nombre de cuerpo
            output_directory=hydro_output_dir,
            nc_filename="body.nc",  # Cambiado el nombre por defecto
            plot_xlim=[-1, 1],
            plot_ylim=[-1.8, 0.6],
            save_plots=True,
            show_plots=False,  # No plots during batch processing
            logging_level="WARNING",  # Reduce output for batch processing
        )
        
        # Show key results

        print(f"✅ Analysis completed for {len(frequencies)} frequencies")
        
        # Verify output files after analysis
        hydro_files = ['HydCoeff.npz', 'HydCoeff.pkl', 'HydCoeff.mat']
        print(f"🗂 Verifying output files...")
        files_found = []
        for file in hydro_files:
            file_path = os.path.join(hydro_output_dir, file)
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                print(f"   ✅ {file}: {file_size:,} bytes")
                files_found.append(file)
            else:
                print(f"   ⚠ {file}: NOT FOUND")
        
        if files_found:
            print(f"✅ {len(files_found)} output files created successfully")
        else:
            print(f"⚠ NO output files were created")
            print(f"   Verify that Eco_Cap2B.py is updated with corrected version")
        
    except Exception as e:
        print(f"⚠ CRITICAL ERROR in hydrodynamic analysis for {folder_path}")
        print(f"   Error type: {type(e).__name__}")
        print(f"   Message: {str(e)}")
        print(f"   Continuing with next case...")
        continue

# Final summary

print(f"\n🔧 Mesh Generation Parameters:")
print(f"  - Circumferential segments: {NUM_SEGMENTS}")
print(f"  - Z-axis subdivisions per segment: {Z_SUBDIVISIONS}")

print(f"\n✅ Analysis complete! Check results in '{FOLDER_NAME}/' folder")
