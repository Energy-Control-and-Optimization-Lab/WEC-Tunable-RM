# Wave Energy Converter – Research Repository

**Energy Control and Optimization Lab · University of New Hampshire**

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![MATLAB](https://img.shields.io/badge/MATLAB-R2021b+-orange.svg)](https://www.mathworks.com/)
[![WEC-Sim](https://img.shields.io/badge/WEC--Sim-v6.0-green.svg)](https://wec-sim.github.io/WEC-Sim/)
[![Capytaine](https://img.shields.io/badge/Capytaine-2.0+-blue.svg)](https://capytaine.github.io/)

---

## Overview

This repository contains the full research pipeline for the **hydrodynamic analysis, design optimization, and power take-off (PTO) modelling** of a two-body point-absorber Wave Energy Converter (WEC). The work spans three main development branches, each corresponding to a distinct research stage and publication.

---

## Repository Branches

### `main` — Reference Model 3 (RM3) Parametric Analysis

> 📄 *Full paper: `IDETC_2026_Tunable_RM_Pablo.pdf`*

The `main` branch implements a **parametric hydrodynamic analysis** of the Reference Model 3 (RM3) two-body WEC using the `EcoFunctions` Python library. It automates geometry generation and Boundary Element Method (BEM) analysis via Capytaine across a systematic sweep of geometric configurations (8 outer radii × 3 interior depth offsets = **24 configurations**).

**Key capabilities:**
- Parametric STL geometry generation from 2D revolution profiles
- Single- and two-body hydrodynamic analysis (added mass, radiation damping, excitation forces)
- Response Amplitude Operator (RAO) computation across 500 frequency points (0.2–1.8 rad/s)
- Multi-format export: `.npz`, `.pkl`, `.mat`, and `.nc` (NetCDF)
- Automated visualization of RAO, hydrodynamic coefficients, and geometry

**Stack:** Python 3.8+, Capytaine ≥ 2.0, NumPy, SciPy, Matplotlib, xarray

---

### `DOE` — Design Optimization Pipeline

> 📄 *Paper submitted to **Renewable Energy** — currently under review.*

The `DOE` branch implements a **full multi-objective design optimization** workflow for the two-body WEC, covering the complete pipeline from hydrodynamic simulation to Pareto-optimal design selection.

The workflow is organized into sequential stages:

| Stage | Description |
|-------|-------------|
| **St0** | Mesh convergence studies and PTO damping range estimation |
| **St1** | Design of Experiments (DOE) matrix generation |
| **St2** | Batch hydrodynamic analysis for each DOE experiment |
| **St3** | Heave RAO and captured power computation |
| **St4** | Response vector assembly and quadratic RSM metamodel training |
| **St5** | Differential Evolution optimization, Pareto analysis, ANOVA, and post-processing |

**Key capabilities:**
- Box-Behnken and full factorial design matrices via `Eco_DOE`
- Quadratic Response Surface Metamodel (RSM) via `scikit-learn`
- Differential Evolution optimization with Pareto front extraction
- ANOVA sensitivity analysis for design variable ranking

**Stack:** Python 3.8+, Capytaine, NumPy, SciPy, scikit-learn, Matplotlib

---

### `PTO-Modelling` — Power Take-Off Configurations

> 📄 *Conference paper: `MECC2026_Tunable_WEC_PTO.pdf` (MECC 2026)*

The `PTO-Modelling` branch implements and compares **three PTO configurations** for the two-body WEC in time-domain simulation using WEC-Sim v6.0. All configurations are tuned to an equivalent PTO damping of **540 N·s/m** for fair comparison.

| Model | Type | Description |
|-------|------|-------------|
| **SD** | Spring-Damper | Passive linear baseline; no electrical subsystem |
| **DDLG** | Direct-Drive Linear Generator | Tubular PM generator (K&J Magnetics N52 magnet + Arbor Scientific coil); no gearbox |
| **EGEC** | Electric Generator Equivalent Circuit | Rotary PM generator (Akribis ADR175-B143) coupled via rack-and-pinion (r = 43 mm) |

Wave conditions used for all models: T = 1.57 s, H = 0.32 m (regular waves, 120 s simulation).

**Stack:** MATLAB R2021b+, Simulink, Simscape Multibody, WEC-Sim v6.0, Capytaine (BEM data generation)

---

## Release History

| Release | Branch | Description | Publication |
|---------|--------|-------------|-------------|
| `v1.0` | `main` | RM3 parametric hydrodynamic sweep (24 configurations) | IDETC 2026 |
| `v2.0` | `DOE` | Multi-objective design optimization pipeline | *Renewable Energy* (under review) |
| `v3.0` | `PTO-Modelling` | Time-domain PTO comparison (SD, DDLG, EGEC) | MECC 2026 |

---

## Getting Started

Each branch contains its own `README.md` with detailed installation instructions, usage examples, and configuration guidelines. Clone the desired branch directly:

```bash
# RM3 parametric analysis (main)
git clone https://github.com/Energy-Control-and-Optimization-Lab/Parametrization-Design.git

# Design optimization pipeline
git clone -b DOE https://github.com/Energy-Control-and-Optimization-Lab/Parametrization-Design.git

# PTO modelling
git clone -b PTO-Modelling https://github.com/Energy-Control-and-Optimization-Lab/WEC-Tunable-RM.git
```

### Common Requirements

```bash
# Python dependencies (main and DOE branches)
pip install numpy matplotlib scipy capytaine xarray netCDF4 h5netcdf scikit-learn

# MATLAB dependencies (PTO-Modelling branch)
# Install WEC-Sim v6.0: https://wec-sim.github.io/WEC-Sim/main/user/installation.html
```

---

## Author & Contact

**Pablo Antonio Matamala Carvajal**  
Energy Control and Optimization Lab  
University of New Hampshire  
📧 Pablo.MatamalaCarvajal@unh.edu  
🔗 [github.com/Energy-Control-and-Optimization-Lab](https://github.com/Energy-Control-and-Optimization-Lab)

---

## License

See individual branch `LICENSE` files for terms.
