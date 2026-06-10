# WEC-Tunable-RM — PTO Modelling

Simulation models for a two-body point-absorber **Wave Energy Converter (WEC)** built on the [WEC-Sim](https://wec-sim.github.io/WEC-Sim/) framework (v6.0). This branch (`PTO-Modelling`) implements and compares three Power Take-Off (PTO) configurations under regular wave conditions, as part of research presented at **MECC 2026**.

---

## Repository Structure

```
PTO-Modelling/
├── DDLG/                            # Direct-Drive Linear Generator
│   ├── geometry/                    # Float and spar STL meshes
│   ├── hydroData/                   # BEM data (Capytaine, .h5 format)
│   ├── ECO_RM_WEC_DDLG.slx          # Simulink model
│   ├── userDefinedFunctions.m       # Post-processing plots
│   └── wecSimInputFile.m            # Simulation & PTO parameters
│
├── EGEC/                            # Electric Generator Equivalent Circuit
│   ├── geometry/
│   ├── hydroData/
│   ├── ECO_RM_WEC_EGEC.slx
│   ├── userDefinedFunctions.m
│   └── wecSimInputFile.m
│
└── SD/                              # Spring-Damper (passive baseline)
│   ├── ECO_RM_WEC_SD.slx
│   ├── userDefinedFunctions.m
│   ├── wecSimInputFile.m
└── MECC2026_Tunable_WEC_PTO.pdf  # Conference paper
```

> **Note:** `geometry/` and `hydroData/` are shared across all three PTO models. The hydrodynamic data was computed with [Capytaine](https://capytaine.github.io/) (open-source BEM solver) and stored in HDF5 (`.h5`) format.

---

## Device Description

The modelled device is a **two-body heaving point absorber** consisting of:

| Body | Name  | Mass (kg) | Description |
|------|-------|-----------|-------------|
| 1    | Float | 35.20     | Outer buoy, oscillates with incident waves |
| 2    | Spar  | 70.71     | Inner reference body, heave-constrained |

Relative heave motion between the float and spar drives the PTO system. Both bodies are loaded from a shared BEM dataset (`hydroData/ECO_RM_WEC.h5`) and share the same STL geometry files. The spar includes a linear viscous damping term of **257.64 N·s/m** in the heave direction for all three models.

---

## PTO Configurations

All three configurations are tuned to achieve the same equivalent PTO damping of **B_pto = 540 N·s/m**, validated from Capytaine hydrodynamic results. This allows a fair comparison of mechanical and electrical performance across PTO types.

---

### 1. SD — Spring-Damper *(Passive Baseline)*

A purely passive linear PTO modelled as a spring-damper element directly within WEC-Sim. No electrical subsystem is present.

| Parameter | Value |
|-----------|-------|
| PTO stiffness | 0 N/m |
| PTO damping | 540 N·s/m |
| WEC-Sim PTO type | `ptoClass` (native) |

---

### 2. DDLG — Direct-Drive Linear Generator

A tubular permanent-magnet linear generator coupled directly to the float–spar relative motion. No gearbox. Uses WEC-Sim's `ptoSim` module (`directLinearGenerator` type).

**Hardware:**
- Magnet: K&J Magnetics DY08-N52 (2 in diameter, ½ in thick, Grade N52)
- Coil: Arbor Scientific P8-6500 (N = 560 turns, R = 1.1 Ω, L = 13 mH, D_int = 5.74 cm)

**Electromagnetic parameters:**

| Parameter | Symbol | Value |
|-----------|--------|-------|
| Pole pitch | τ_p | 15.2 mm |
| Air gap factor | k_g | 0.885 |
| Fringing factor | k_fringe | 0.75 |
| Flux linkage | λ_fd | 1.115 Wb |
| Force constant | K | 230.46 N/A |
| Winding resistance | R_s | 1.10 Ω |
| Load resistance | R_load | 97.25 Ω |
| Electrical efficiency | η | 98.9% |
| Electrical time constant | τ_e | 1.32 × 10⁻⁴ s |
| Min. coil length | — | 0.530 m |

> ⚠️ The simulation uses `dt = 0.00025 s`. For accurate electromagnetic transient resolution, the recommended timestep is `τ_e / 5 ≈ 2.64 × 10⁻⁵ s`. A coarser step is used here as a trade-off for simulation speed.

---

### 3. EGEC — Electric Generator Equivalent Circuit

A rotary permanent-magnet generator coupled to the WEC via a **rack-and-pinion transmission** (pinion radius r = 43 mm). Uses WEC-Sim's `ptoSim` module (type 5 — Electric Generator Equivalent Circuit).

**Hardware:** Akribis ADR175-B143 (3-phase, Y-connected)

**Electrical parameters (per-phase, Y-equivalent):**

| Parameter | Symbol | Value |
|-----------|--------|-------|
| Armature resistance | R_a | 0.89 Ω |
| Armature inductance | L_a | 5.0 mH |
| Back-EMF constant | K_e | 4.1062 V/(rad/s) |
| Rotor inertia | J_em | 5.422 × 10⁻³ kg·m² |
| Gear ratio | N_g | 23.3 rad/m |
| Load resistance | R_load | 15.997 Ω |
| Electrical efficiency | η | 94.7% |
| Operating speed | — | 222.1 rpm |

**Simulation results (steady-state, t ≥ 30 s):**

| Quantity | Value |
|----------|-------|
| Relative velocity RMS | 0.2602 m/s |
| PTO force RMS | 140.68 N |
| Effective B_pto | 540.7 N·s/m |
| Mechanical power | 36.60 W |
| Electrical power | 34.72 W |
| Power loss | 1.88 W |
| Voltage RMS | 23.57 V |
| Current RMS | 1.47 A |
| Efficiency | 94.88% |

> ⚠️ The required torque (23.22 N·m) exceeds the motor's continuous rating (16.0 N·m). To resolve, increase the pinion radius `r` or reduce `B_pto_target` in `wecSimInputFile.m`.

---

## Simulation Settings

All three models share the same wave environment and time-domain settings:

| Setting | Value |
|---------|-------|
| Wave type | Regular (constant hydrodynamic coefficients) |
| Wave period | 1.57 s  *(ω = 4 rad/s)* |
| Wave height | 0.32 m  *(amplitude η = 0.16 m)* |
| Start time | 0 s |
| End time | 120 s |
| Time step | 0.00025 s |
| Ramp time | 30 s |
| Domain size | 10 m |

Post-processing uses data from **t ≥ 30 s** (after the ramp) for all power and efficiency calculations.

> **Wave conditions are fully configurable.** To simulate a different sea state, edit the following lines in `wecSimInputFile.m` of the desired PTO folder:
>
> ```matlab
> waves.period = 1.57;   % [s]  — change to desired wave period
> waves.height = 0.32;   % [m]  — change to desired wave height
> ```
>
> The BEM hydrodynamic data (`hydroData/ECO_RM_WEC.h5`) covers a range of frequencies computed by Capytaine, so the model is valid across the frequency band included in that dataset. For the EGEC and DDLG models, the PTO electrical parameters are tuned for a target damping (`B_pto_target`) independent of wave conditions and do not need to be recalculated when changing the wave input.

---

## Requirements

| Software | Version | Notes |
|----------|---------|-------|
| MATLAB | R2021b or later | Required |
| Simulink + Simscape Multibody | (with MATLAB) | Required |
| [WEC-Sim](https://wec-sim.github.io/WEC-Sim/) | v6.0 | Free, open-source |
| [Capytaine](https://capytaine.github.io/) | Any | Only needed to regenerate BEM data |

---

## Getting Started

### 1. Install WEC-Sim

Follow the [official installation guide](https://wec-sim.github.io/WEC-Sim/main/user/installation.html). WEC-Sim v6.0 is required.

### 2. Clone this repository

```bash
git clone -b PTO-Modelling https://github.com/Energy-Control-and-Optimization-Lab/WEC-Tunable-RM.git
cd WEC-Tunable-RM
```

### 3. Run a simulation

Navigate to the desired PTO folder and run WEC-Sim from MATLAB:

```matlab
% Example: Spring-Damper model
cd SD
wecSim
```

WEC-Sim reads `wecSimInputFile.m` automatically and launches the corresponding Simulink model. The `userDefinedFunctions.m` script runs post-processing at the end of the simulation.

For the **EGEC** and **DDLG** models, `wecSimInputFile.m` also computes and pushes all electromagnetic parameters into the Simulink block masks before running.

### 4. Post-processing outputs

The following plots are generated automatically after each run:

- Wave elevation (with ramp period marked)
- Heave response — float (Body 1) and spar (Body 2)
- Heave forces — float and spar

The **EGEC** model additionally produces a 4-panel figure with:
- Mechanical power absorbed
- Terminal voltage
- Armature current
- Electrical power output

---

## Reference

If you use this repository, please cite:

> Matamala *et al.*, "Tunable WEC PTO Modelling," *Proceedings of MECC 2026*.

The full paper is available at `SD/MECC2026_Tunable_WEC_PTO.pdf`.

---

## License

See the repository root for license information.

---

## Contact

**Energy Control and Optimization Lab**  
University of New Hampshire (UNH)  
[github.com/Energy-Control-and-Optimization-Lab](https://github.com/Energy-Control-and-Optimization-Lab)
