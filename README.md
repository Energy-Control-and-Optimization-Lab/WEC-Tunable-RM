# Wave Energy Converter – Design Optimization Pipeline

Codebase for the design and optimization of a two-body wave energy converter (WEC). The workflow covers the full pipeline from geometry generation and hydrodynamic analysis to metamodel construction and multi-objective optimization.

> 📄 **A paper based on this work has been submitted to *Renewable Energy* and is currently under review.**

---

## Project Structure

```
/
├── EcoFunctions/          # Reusable core functions (internal library)
│   ├── Eco_StlRev.py      # Revolution solid STL mesh generation
│   ├── Eco_Cap1B.py       # Single-body hydrodynamics (BEM)
│   ├── Eco_Cap2B.py       # Two-body hydrodynamics (BEM)
│   ├── Eco_RAO.py         # Response Amplitude Operator (RAO) calculation
│   ├── Eco_Power.py       # Captured power estimation
│   └── Eco_DOE.py         # Design of Experiments (DOE)
│
└── Scripts/               # Stage-by-stage execution scripts
    ├── St0_*.py           # Mesh convergence studies and initial setup
    ├── St1_*.py           # DOE value generation
    ├── St2_*.py           # Batch hydrodynamic analysis
    ├── St3_*.py           # RAO and power computation
    ├── St4_*.py           # Results vector assembly and metamodel training
    └── St5_*.py           # Optimization, Pareto analysis, ANOVA, post-processing
```

---

## Workflow

```
St0  →  St1  →  St2  →  St3  →  St4  →  St5
```

| Stage | Scripts | Description |
|-------|---------|-------------|
| **St0** | `ConvFloater`, `ConvSpar`, `PTOdamping` | Mesh convergence studies and PTO damping range estimation |
| **St1** | `DoeValues` | Design of Experiments matrix generation |
| **St2** | `Hydro` | Hydrodynamic analysis for each DOE experiment |
| **St3** | `RAO`, `Power` | Heave RAO calculation and captured power estimation |
| **St4** | `ResultsVector`, `MetaModel` | Response vector assembly and quadratic RSM metamodel training |
| **St5** | `OptimizationSetup`, `ScientificPareto`, `ANOVA`, `Analysis` | Differential Evolution optimization, Pareto analysis, ANOVA, and post-processing |

---

## Core Functions (`EcoFunctions/`)

| Function | Description |
|----------|-------------|
| `Eco_StlRev` | Generates STL meshes of revolution solids from a 2D profile |
| `Eco_Cap1B` / `Eco_Cap2B` | Single- and two-body hydrodynamic analysis using BEM |
| `Eco_RAO` | Computes the heave RAO with or without PTO damping |
| `Eco_Power` | Estimates mean captured power for a given wave spectrum |
| `Eco_DOE` | Generates design matrices (Box-Behnken, full factorial, etc.) |

---

## Requirements

- Python 3.8+
- `numpy`, `scipy`, `matplotlib`
- `scikit-learn` (metamodel)
- Capytaine (BEM solver)

---

## Basic Usage

Run the scripts sequentially:

```bash
python St0_A_ConvFloater.py        # Mesh convergence
python St1_A_DoeValues.py          # Generate DOE
python St2_A_Hydro.py              # Hydrodynamics
python St3_A_RAO.py                # RAO
python St3_B_Power.py              # Power
python St4_A_ResultsVector.py      # Results vector
python St4_B_MetaModel.py          # Metamodel
python St5_A_OptimizationSetup.py  # Optimization
```

Intermediate results are saved to `EcoData/` and consumed by the next stage.

---

## Author

Pablo Antonio Matamala Carvajal
