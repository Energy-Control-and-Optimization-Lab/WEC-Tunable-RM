# Two-Body WEC – Climate-Informed Design Optimization with Battery Storage

**Energy Control and Optimization Lab · University of New Hampshire**

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Capytaine](https://img.shields.io/badge/Capytaine-2.0+-blue.svg)](https://capytaine.github.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-RSM-orange.svg)](https://scikit-learn.org/)
[![SciPy](https://img.shields.io/badge/SciPy-SLSQP-green.svg)](https://scipy.org/)

---

## Overview

This branch implements the complete design pipeline for a **two-body heaving point-absorber Wave Energy Converter (WEC)**: a float sliding along a spar with a heave plate, connected through a linear Power Take-Off (PTO). The goal is to find the **lightest device** that can continuously power an ocean sensor payload through a **15-year real wave climate**, with a battery (Energy Storage System, ESS) covering low-energy months.

Compared with the earlier `DOE` branch, this workflow adds:

- A **7-parameter design space**: float geometry, spar geometry, PTO damping **and PTO stiffness** (resonance tuning).
- A **15-year monthly wave climatology** (2010–2024) built with Measure-Correlate-Predict (MCP): local AWAC measurements × WIS offshore hindcast (station ST63042). That gives **181 JONSWAP sea states**.
- **182 quadratic response-surface metamodels**: annual power, power-to-mass ratio, and one surrogate per month-year.
- **Mass minimization via multi-start SLSQP** subject to **monthly battery feasibility constraints** (energy balance + state-of-charge bounds) across all 180 months.

The geometry generation and BEM analysis rely on the same `EcoFunctions` library used in the other branches of this repository.

---

## Design Variables

| Variable | Description | Range | Unit | Enters |
|----------|-------------|-------|------|--------|
| **D1** | Float diameter | 2.1 – 2.8 | m | BEM, mass |
| **D2** | Float external draft | 0.3 – 0.5 | m | BEM, mass |
| **D3** | Float internal draft offset (conical bottom) | 0.0 – 0.20 | m | BEM, mass |
| **D4** | Spar draft | 3.9 – 5.2 | m | BEM, mass |
| **D5** | Spar heave-plate diameter | 3.0 – 4.0 | m | BEM, mass |
| **D6** | PTO damping $B_{PTO}$ | 50 000 – 100 000 | kg/s | Power only |
| **D7** | PTO stiffness $K_{PTO}$ | 0 – 50 000 | N/m | Power only |

D6 and D7 act on the **float–spar relative motion**. They are not used in the BEM step and enter the dynamics in the impedance matrix (St3_B).

Fixed geometric constants: `F1` (inner/connection diameter), `F2` (heave-plate thickness), `F3` (float freeboard), `F4` (spar freeboard). Site: **10 m water depth**, BEM frequencies **0.5–2.5 rad/s** (Δω = 0.025 rad/s).

---

## Pipeline

```
St1 ──► St2 ──► St3_B ──► St3_C ──► St4_A ──► St4_B ──► St5_A
 DOE     BEM     P(ω)      P_sea      vectors   RSM       SLSQP + ESS
                   ▲
          St3_A ───┘ (JONSWAP spectra, independent of St1/St2)
```

| Stage | Script | Description |
|-------|--------|-------------|
| **St1** | `St1_A_DoeValues.py` | Box-Behnken design for 7 factors (+3 center points) via `Eco_DOE` |
| **St2** | `St2_A_Hydro.py` | Parametric STL generation + two-body Capytaine BEM for each experiment |
| **St3_A** | `St3_A_Spectra.py` | 181 JONSWAP spectra (γ = 3.3): 1 AWAC reference + 180 monthly climatological states |
| **St3_B** | `St3_B_Power.py` | Coupled 2-DOF heave RAOs with PTO damping + stiffness, power curve $P(\omega)$ |
| **St3_C** | `St3_C_SpectralPower.py` | Spectral convolution: mean power per sea state $P_{sea}$ (181 values per experiment) |
| **St4_A** | `St4_A_ResultsVector.py` | Joins DOE matrix + responses, applies PTO efficiency, computes mass and PMR |
| **St4_B** | `St4_B_MetaModel.py` | 182 quadratic RSM metamodels with pure-error / lack-of-fit validation |
| **St5_A** | `St5_A_OptimizationSetup_V3.py` | Multi-start SLSQP mass minimization with monthly battery (ESS) constraints |

---

## Methodology

### St1 – Design of Experiments
A **Box-Behnken** design over D1–D7 with 3 center points. BEM is deterministic, so the replicated center points mainly serve as a consistency check for the pure-error estimate in St4_B.

### St2 – Hydrodynamics (BEM)
For each experiment, the float and spar are defined as 2D profiles and revolved into STL meshes (`Eco_StlRev.generate_revolution_solid_stl`). The two-body problem is then solved with Capytaine (`Eco_Cap2B.analyze_two_body_hydrodynamics`), which exports added mass $A(\omega)$, radiation damping $B(\omega)$, excitation force $F_e(\omega)$, and the mass and hydrostatic matrices ($12 \times 12$, 6 DOF per body).

### St3_A – Wave Climate
The following sea states are converted to Hs-normalized JONSWAP spectra:
- `Mean_2024`: annual mean measured by AWAC (Hs = 0.27 m, Tp = 7.56 s), kept as a validation reference.
- `y01_Jan … y15_Dec`: 180 monthly mean states (Hs, Tp) from the 15-year synthetic MCP series (y01 = 2010 … y15 = 2024).

### St3_B – Power Absorption
The heave DOFs (float = DOF 3, spar = DOF 9) are extracted into a coupled $2 \times 2$ system:

$$
\mathbf{Z}(\omega) = -\omega^2(\mathbf{M}+\mathbf{A}) + i\omega(\mathbf{B}_{hyd}+\mathbf{B}_{PTO}) + (\mathbf{C}+\mathbf{K}_{PTO}),
\qquad \mathbf{Z}\,\mathbf{X} = \mathbf{F}_e
$$

with relative-motion PTO matrices
$\mathbf{B}_{PTO} = B_{PTO}\begin{bmatrix}1&-1\\-1&1\end{bmatrix}$, $\mathbf{K}_{PTO} = K_{PTO}\begin{bmatrix}1&-1\\-1&1\end{bmatrix}$.

The absorbed power for a unit wave amplitude is:

$$
P(\omega) = \tfrac{1}{2}\, B_{PTO}\, \omega^2 \,|X_{float} - X_{spar}|^2
$$

### St3_C – Spectral Power
Each spectrum is interpolated onto the BEM frequency grid and convolved with the power curve:

$$
S_P(\omega) = 2\,P(\omega)\,S(\omega), \qquad P_{sea} = \int S_P(\omega)\,d\omega
$$

`P_avg_annual` is the value for `Mean_2024`.

### St4_A – Response Vectors
- PTO efficiency $\eta_{PTO} = 0.50$ is applied **once**, here, to `P_avg_annual` and `P_sea`. It must not be re-applied downstream.
- $M_{total} = M_{float} + M_{spar}$, taken from the BEM mass matrix.
- $\mathrm{PMR} = P_{avg,annual} / (M_{total}/1000)$ [W/t].

### St4_B – Metamodels
A full quadratic polynomial is fitted for each response (1 + 7 linear + 21 interaction + 7 quadratic = **36 terms**). There are 182 responses: `P_avg_annual`, `PMR`, and `P_sea_yNN_Mon` × 180. Each model reports:
- R², adjusted R², RMSE, MAE, MAPE, with 🟢/🟡/🔴 quality flags
- Pure error from the center points and lack-of-fit F-test
- Standardized effects (parameters coded A–G), kept as table and MATLAB data

The plots are grouped into 16 figures: 15 per-year 3×4 Actual-vs-Predicted grids plus 1 figure for `P_avg_annual` and PMR.

### St5_A – Constrained Mass Optimization
**Objective:** minimize the analytical displaced mass $M_{total}(D1..D5)$ (float frustum + cylinder − inner hole + spar tube + heave plate, ρ = 1025 kg/m³).

**Battery model**, evaluated each month $m$ over `N_YEARS × 12` months using the monthly surrogates $AP(m)$:

| Quantity | Definition |
|----------|------------|
| Energy balance | $E_{diff}(m) = (AP(m) - P_T)\,h(m)$ |
| Stored energy | $\eta_c E_{diff}$ if surplus, $E_{diff}/\eta_d$ if deficit |
| Battery sizing | $C_B = \max_{y}\; \lvert\min_m E_{net,y}(m)\rvert / (SOC_{max}-SOC_{min})$, i.e. the **worst of the 15 years** |
| State of charge | Two-sided clipped recursion over the full horizon with fixed $C_B$ |

**Constraints per month**, 3 × 180 = 540 in total:
1. Balance (single discharge efficiency): $AP(m)\,h(m) + \eta_d\,E_B(m-1) \ge P_T\,h(m)$
2. $SOC(m) \ge SOC_{min}$
3. $SOC(m) \le SOC_{max}$

**Default parameters:** $P_T$ = 15 W (sensor suite × safety factor), $\eta_c = \eta_d$ = 0.92, SOC ∈ [0.10, 0.90], 100 random starts (seed 1000), `N_YEARS` = 15.

Results are written **live** after every start via atomic write-then-rename, so they can be inspected while the optimization is still running.

---

## Data Structure

```
EcoData/
├── St1_DOEvalues/            DOE_values.{pkl,mat}
├── St2_EcoBatch/
│   └── DOE_Exp_XXX/
│       ├── geometry/         float.stl, spar.stl, profile plots
│       ├── hydroData/        HydCoeff.{npz,pkl,mat}, Eco2BPA.nc
│       └── experiment_summary.json
├── St3_Spectra/              Spectra.{npz,pkl,mat} + Plots/ (16 figures)
├── St3_Power/
│   └── DOE_Exp_XXX/          St3_B_Exp_XXX.{npz,pkl,mat} + Plots/RAO_Power.png
├── St3_SpectralPower/
│   └── DOE_Exp_XXX/          St3_C_Exp_XXX.{npz,pkl,mat} + Plots/ (16 figures)
├── St4_ResultVector/         VectorValues.{npz,pkl,mat}
├── St4_MetaModel/
│   ├── Analysis/             enhanced_metamodel_results.{pkl,mat}, summary CSV,
│   │                         individual_enhanced_metamodels/ (182 × .pkl + .mat)
│   └── Plots/                MetaModel_y01..y15.png, MetaModel_Pavg_PMR.png
└── St5_Optimization_Iterative/
    ├── final_results/        Mass_optimal_designs.{pkl,csv,mat}
    ├── raw_results/          all_starts_raw.pkl
    ├── Mass_optimization_results.png
    └── Best_Candidate_SOC_trajectory.png
```

All stages export `.pkl` (Python), `.npz` and `.mat` (MATLAB).

---

## Getting Started

### Requirements

```bash
pip install numpy scipy matplotlib pandas scikit-learn capytaine xarray netCDF4 h5netcdf
```

The `EcoFunctions/` package must be located next to the scripts. The following modules are used:

| Module | Function | Used in |
|--------|----------|---------|
| `Eco_DOE` | `generate_doe_vectors` | St1 |
| `Eco_StlRev` | `generate_revolution_solid_stl` | St2 |
| `Eco_Cap2B` | `analyze_two_body_hydrodynamics` | St2 |

### Running the pipeline

Run the scripts from the repository root, in order:

```bash
python St1_A_DoeValues.py
python St2_A_Hydro.py              # most expensive step (BEM for every experiment)
python St3_A_Spectra.py            # independent of St1/St2
python St3_B_Power.py
python St3_C_SpectralPower.py
python St4_A_ResultsVector.py
python St4_B_MetaModel.py
python St5_A_OptimizationSetup_V3.py
```

**Resume after interruption:** St2, St3_B and St3_C expose `START_FROM_EXPERIMENT` in their configuration block.

**Key settings:**

| Setting | Script | Default |
|---------|--------|---------|
| `frequencies`, `SEA_BOTTOM` | St2, St3_A | 0.5–2.5 rad/s, −10 m |
| `GAMMA` | St3_A | 3.3 |
| `ETA_PTO` | St4_A | 0.50 |
| `QUALITY_THRESHOLDS` | St4_B | R² 0.90 / 0.75, MAPE 5 % / 15 % |
| `PT`, `ETA_CHARGE`, `ETA_DISCHARGE`, `SOC_MIN/MAX` | St5_A | 15 W, 0.92, 0.92, 0.10/0.90 |
| `N_YEARS`, `N_STARTS` | St5_A | 15, 100 |

---

## Related Branches

| Branch | Description | Publication |
|--------|-------------|-------------|
| `main` | RM3 parametric hydrodynamic sweep | IDETC 2026 |
| `DOE` | Multi-objective design optimization (Differential Evolution, Pareto) | *Renewable Energy* (under review) |
| `PTO-Modelling` | Time-domain PTO comparison (SD, DDLG, EGEC) in WEC-Sim | MECC 2026 |

---

## Author & Contact

**Pablo Antonio Matamala Carvajal**
Energy Control and Optimization Lab
University of New Hampshire
📧 Pablo.MatamalaCarvajal@unh.edu
🔗 [github.com/Energy-Control-and-Optimization-Lab](https://github.com/Energy-Control-and-Optimization-Lab)

---

## License

See the `LICENSE` file for terms.
