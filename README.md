# EcoFunctions

**Parametric Hydrodynamic Analysis for Wave Energy Converters**

[![Version](https://img.shields.io/badge/version-2.0.0-blue.svg)](https://github.com/Energy-Control-and-Optimization-Lab/Parametrization-Design)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

---

## Overview

EcoFunctions is a Python package for parametric hydrodynamic analysis of Wave Energy Converters (WECs). It automates the process of geometry generation and Boundary Element Method (BEM) analysis using Capytaine to study floating body dynamics across multiple parameter combinations.

### Key Features

- **Parametric Geometry Generation**: Automatically creates STL files from 2D profiles with configurable mesh density
- **Batch Processing**: Analyzes multiple geometric configurations systematically
- **BEM Hydrodynamic Analysis**: Computes radiation, diffraction, and excitation forces using Capytaine
- **Two-Body Interaction**: Calculates coupled dynamics between floating bodies
- **Multi-Format Export**: Results saved in NPZ, PKL, MAT, and NetCDF formats
- **Automated Visualization**: Generates plots for RAO, added mass, damping, and geometry

---

## Installation

### Requirements

```bash
numpy >= 1.20.0
matplotlib >= 3.3.0
scipy >= 1.7.0
capytaine >= 2.0
xarray >= 0.19.0
netCDF4 >= 1.5.7
h5netcdf >= 0.14.0
```

### Setup

```bash
# Clone repository
git clone https://github.com/Energy-Control-and-Optimization-Lab/Parametrization-Design.git
cd Parametrization-Design

# Install dependencies
pip install numpy matplotlib scipy capytaine xarray netCDF4 h5netcdf vtk

# Verify Capytaine installation
python -c "import capytaine; print('Capytaine installed successfully!')"
```

---

## Project Structure

```
Parametrization-Design/
├── EcoFunctions/
│   ├── __init__.py              # Package initialization
│   ├── Eco_StlRev.py           # STL geometry generation
│   ├── Eco_Cap2B.py            # Two-body hydrodynamics
│   └── Eco_Cap1B.py            # Single-body hydrodynamics
├── Main_ACC2026.py             # Main parametric analysis script
├── Main_Convergence_floater.py # Mesh convergence study
└── Default/
    └── geometry/
        └── plate_refined.GDF    # Reference plate geometry
```

---

## Usage

### Main Parametric Analysis

The `Main_ACC2026.py` script performs systematic parametric studies by varying geometric parameters and analyzing the hydrodynamic response.

#### Parameter Configuration

The script defines a relationship between radius (R) and depth (D):

```python
R_values = np.array([6, 7, 8, 9, 10, 11, 12, 13])
D_values = 273 / ((R_values**2) - 9)
Dint_values = np.array([0, 0.5, 1])
```

Where:
- **R**: Outer radius of the float [m]
- **D**: Submersion depth calculated from R [m]
- **Dint**: Interior depth offset [m]

This generates **8 × 3 = 24 geometric configurations**.

#### Float Geometry Definition

For each combination of (R, D, Dint), a float geometry is defined by 4 profile points:

```python
P = np.array([
    [3, 0, -D-Dint],   # P1: Inner radius at bottom
    [R, 0, -D],        # P2: Outer radius at bottom
    [R, 0, 2],         # P3: Outer radius at top
    [3, 0, 2],         # P4: Inner radius at top
])
```

#### Mesh Generation Parameters

```python
NUM_SEGMENTS = 50        # Circumferential segments
Z_SUBDIVISIONS = 8       # Z-axis subdivisions per segment
```

Higher values increase mesh density and computational cost but improve accuracy.

#### Frequency Range

```python
frequencies = np.linspace(0.2, 1.8, 500)  # [rad/s]
```

Analyzes the system response across 500 frequency points from 0.2 to 1.8 rad/s.

---

## Running the Analysis

### Basic Execution

```bash
python Main_ACC2026.py
```

### What Happens During Execution

For each geometric configuration:

1. **Creates folder structure**: `batch_ACC2026/Geometry_R{R}_Dint{Dint}/`
2. **Copies reference files** from `Default/` folder
3. **Generates float geometry**: Creates `float.stl` with specified parameters
4. **Runs hydrodynamic analysis**: Computes interaction between float and plate
5. **Saves results**: Multiple formats in `hydroData/` subfolder
6. **Generates plots**: RAO, damping, added mass, and geometry visualizations

### Output Structure

```
batch_ACC2026/
├── Geometry_R6_Dint0/
│   ├── geometry/
│   │   ├── float.stl
│   │   ├── plate_refined.GDF
│   │   └── profile_plot_subdivided.png
│   └── hydroData/
│       ├── HydCoeff.npz           # NumPy compressed
│       ├── HydCoeff.pkl           # Python pickle
│       ├── HydCoeff.mat           # MATLAB format
│       ├── rm3.nc                 # NetCDF dataset
│       ├── RAO_heave_comparison.png
│       ├── added_mass_coefficients.png
│       ├── radiation_damping_coefficients.png
│       └── geometry_lateral_view.png
├── Geometry_R6_Dint5/
│   └── ...
└── Geometry_R13_Dint10/
    └── ...
```

---

## EcoFunctions Modules

### 1. Geometry Generation (`Eco_StlRev.py`)

Generates solids of revolution from 2D profiles.

```python
from EcoFunctions import generate_revolution_solid_stl

result = generate_revolution_solid_stl(
    points=P,                      # Profile points [(x,y,z), ...]
    filename="float.stl",          # Output filename
    num_segments=50,               # Circumferential resolution
    z_subdivisions=8,              # Z-axis density
    visualize=False,               # Show plots
    save_plot_path="./geometry"    # Save location
)
```

**Key Features:**
- Revolution around Z-axis
- Configurable mesh density in both circumferential and Z directions
- Automatic profile subdivision for smooth surfaces
- Visualization of original vs subdivided profiles

**Output:**
```python
{
    'filename': 'float.stl',
    'num_vertices': 3200,
    'num_triangles': 6400,
    'num_original_points': 4,
    'num_profile_points': 32,
    'subdivision_factor': 8
}
```

### 2. Two-Body Hydrodynamics (`Eco_Cap2B.py`)

Performs coupled hydrodynamic analysis between two floating bodies.

```python
from EcoFunctions import analyze_two_body_hydrodynamics

results = analyze_two_body_hydrodynamics(
    mesh1_path="geometry/float.stl",
    mesh2_path="geometry/plate_refined.GDF",
    frequency_range=frequencies,
    mesh1_position=[0.0, 0.0, 0.0],
    mesh2_position=[0.0, 0.0, 0.0],
    body_names=["Float", "Plate"],
    output_directory="hydroData",
    nc_filename="rm3.nc",
    save_plots=True,
    show_plots=False
)
```

**Computed Quantities:**

- **A(ω)**: Added mass matrices [kg, kg·m²]
- **B(ω)**: Radiation damping [kg/s, kg·m²/s]
- **Fe(ω)**: Excitation forces [N, N·m]
- **M**: Inertia matrix [kg, kg·m²]
- **C**: Hydrostatic stiffness [N/m, N·m/rad]
- **RAO**: Response Amplitude Operators for all 12 DOFs
- **Relative Heave**: Relative motion between bodies

**Equation of Motion:**

```
[M + A(ω)] ẍ + B(ω) ẋ + C x = Fe(ω)
```

**RAO Calculation:**

```
RAO(ω) = |Z(ω)⁻¹ Fe(ω)|

where Z(ω) = -ω²[M + A(ω)] + iω B(ω) + C
```

**Return Values:**
```python
{
    'dataset': xarray.Dataset,       # Complete Capytaine dataset
    'RAO': np.ndarray,               # RAO magnitudes (12 × Nω)
    'relative_heave_RAO': np.ndarray,# Relative heave RAO
    'frequencies': np.ndarray,       # Frequency array
    'added_mass': np.ndarray,        # A(ω) matrices
    'radiation_damping': np.ndarray, # B(ω) matrices
    'excitation_force': np.ndarray,  # Fe(ω) vectors
    'inertia_matrix': np.ndarray,    # M matrix
    'hydrostatic_stiffness': np.ndarray, # C matrix
    'Npan1': int,                    # Panel count body 1
    'Npan2': int                     # Panel count body 2
}
```

### 3. Single-Body Hydrodynamics (`Eco_Cap1B.py`)

For analyzing individual floating bodies (6 DOF).

```python
from EcoFunctions import analyze_single_body_hydrodynamics

results = analyze_single_body_hydrodynamics(
    mesh_path="geometry/float.stl",
    frequency_range=frequencies,
    mesh_position=[0.0, 0.0, 0.0],
    body_name="Float",
    output_directory="hydroData"
)
```

---

## Mesh Convergence Study

Use `Main_Convergence_floater.py` to study mesh convergence:

```python
NUM_SEGMENTS = np.array([40, 50, 60, 70, 80])
Z_SUBDIVISIONS = np.array([6, 8, 10, 12, 14])
```

This script analyzes the same geometry with increasing mesh refinement to verify solution convergence.

---

## Configuration Guidelines

### Mesh Density Recommendations

**For quick testing:**
```python
NUM_SEGMENTS = 40
Z_SUBDIVISIONS = 6
frequencies = np.linspace(0.2, 1.8, 50)
```

**For production analysis:**
```python
NUM_SEGMENTS = 50-60
Z_SUBDIVISIONS = 8-10
frequencies = np.linspace(0.2, 1.8, 500)
```

**For high-accuracy studies:**
```python
NUM_SEGMENTS = 70-80
Z_SUBDIVISIONS = 12-14
frequencies = np.linspace(0.2, 1.8, 500)
```

### Frequency Range Selection

- **Standard WEC analysis**: 0.2 - 1.8 rad/s (typical ocean wave conditions)
- **Resolution**: 500 points recommended for smooth RAO curves
- **Quick testing**: 50 points sufficient for initial validation

---

## Accessing Results

### Loading NPZ Files

```python
import numpy as np

data = np.load('hydroData/HydCoeff.npz')
A = data['A']      # Added mass
B = data['B']      # Radiation damping
Fe = data['Fe']    # Excitation force
M = data['M']      # Inertia matrix
C = data['C']      # Hydrostatic stiffness
w = data['w']      # Frequencies
RAO = data['RAO']  # Response amplitudes
```

### Loading Pickle Files

```python
import pickle

with open('hydroData/HydCoeff.pkl', 'rb') as f:
    hydro_coeffs = pickle.load(f)

A = hydro_coeffs['A']
metadata = hydro_coeffs['metadata']
```

### Loading MATLAB Files

```matlab
load('hydroData/HydCoeff.mat')

% Complex variables split into real/imaginary
Fe = Fe_real + 1i*Fe_imag;
RAO = RAO_real + 1i*RAO_imag;
```

---

## Troubleshooting

### Capytaine Installation Issues

**Error**: Missing Fortran compiler

**Solution (Linux)**:
```bash
sudo apt-get install build-essential gfortran
pip install capytaine
```

**Solution (Windows)**:
Install Visual Studio Build Tools with C++ support, then:
```bash
pip install capytaine
```

### NetCDF Export Errors

The code automatically falls back to h5netcdf if netCDF4 fails. Ensure at least one is installed:
```bash
pip install netCDF4 h5netcdf
```

### Memory Issues

Reduce mesh density or frequency points:
```python
NUM_SEGMENTS = 40          # Instead of 70
Z_SUBDIVISIONS = 6         # Instead of 12
frequencies = np.linspace(0.2, 1.8, 100)  # Instead of 500
```

### Missing Default Folder

Ensure the `Default/` folder exists with `plate_refined.GDF`:
```
Default/
└── geometry/
    └── plate_refined.GDF
```

---

## Theory Background

### Hydrodynamic Coefficients

The BEM solver computes frequency-dependent coefficients describing wave-structure interaction:

- **Added Mass A(ω)**: Virtual mass effect from surrounding fluid
- **Radiation Damping B(ω)**: Energy dissipation from radiated waves
- **Excitation Force Fe(ω)**: Wave forces on the structure (Froude-Krylov + diffraction)

### Degrees of Freedom

**Single body**: 6 DOF (surge, sway, heave, roll, pitch, yaw)

**Two bodies**: 12 DOF (6 for each body)

### Response Amplitude Operator (RAO)

The RAO quantifies the motion response per unit wave amplitude:

```
RAO(ω) = Motion amplitude / Wave amplitude
```

Units: [m/m] for translations, [rad/m] for rotations

---

## Author & Contact

**Author**: Pablo Antonio Matamala Carvajal  
**Laboratory**: Energy Control and Optimization Lab  
**Institution**: University of New Hampshire  
**Email**: Pablo.MatamalaCarvajal@unh.edu

---

## References

- **Capytaine Documentation**: [https://capytaine.github.io/](https://capytaine.github.io/)
- **BEM Theory**: Boundary Element Methods for marine hydrodynamics
- **Wave Energy**: Ocean wave energy conversion systems

---

## Citation

If you use this repository, please cite:

> Matamala *et al.*, "Tunable WEC PTO Modelling," *Proceedings of MECC 2026*.

The full paper is available at `IDETC_2026_Tunable_RM_Pablo.pdf`.

If this package helps your research, please cite:

```bibtex
@software{ecofunctions2025,
  author = {Matamala Carvajal, Pablo Antonio},
  title = {EcoFunctions: Parametric Hydrodynamic Analysis for Wave Energy Converters},
  year = {2025},
  version = {2.0.0},
  url = {https://github.com/Energy-Control-and-Optimization-Lab/Parametrization-Design}
}
```

---

## License

[Add license information]

---

**Happy analyzing!**
