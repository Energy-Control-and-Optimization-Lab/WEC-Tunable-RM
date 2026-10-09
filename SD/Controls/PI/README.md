# PI Controller for the Passive SD WEC

This folder contains the WEC-Sim/Simulink implementation of a PI controller
for relative-velocity tracking of the two-body point-absorber WEC.

## Operating Point

- Wave frequency: 4 rad/s
- Wave period: 1.5708 s
- Wave height: 0.32 m
- WEC-Sim ramp time: 30 s
- Simulation duration: 120 s

## Passive PTO

The original passive spring-damper PTO is retained with

- PTO damping: 540 N s/m
- PTO stiffness: 0 N/m

The active PI controller supplies an additional correction force:

`F_PTO,total = -540*v_rel + F_PI`

## PI Controller

The implemented gains are

- Kp = 1625.09 N s/m
- Ki = 4991.19 N/m

The controller tracks the energy-optimal relative-velocity reference with

- Reference amplitude: 1.039494 m/s
- Reference phase: 21.259 deg
- Reference frequency: 4 rad/s

## Main Files

- `ECO_RM_WEC_SD_PI.slx`
  - WEC-Sim/Simulink model with active PI force input.

- `wecSimInputFile.m`
  - WEC-Sim input file for the PI-controlled case.
  - Uses the shared SD hydrodynamic and geometry files located in
    `SD/hydroData` and `SD/geometry`.

- `userDefinedFunctions.m`
  - WEC-Sim post-processing and plotting functions.

- `Reference_velocity_calculation.m`
  - Calculation of the energy-optimal relative-velocity reference.

- `Results/`
  - Passive-versus-PI comparison processing and report figures.

## Running the PI WEC-Sim Case

Set the MATLAB current folder to

`SD/Controls/PI`

and run:

```matlab
wecSim
```

The input file automatically locates the shared files:

- `SD/hydroData/ECO_RM_WEC.h5`
- `SD/geometry/float.stl`
- `SD/geometry/spar.stl`

## Steady-State Results

At the design operating point:

- Passive SD average absorbed mechanical power: 37.06 W
- PI-controlled net absorbed mechanical power: 63.53 W
- Net mechanical power increase: approximately 71.4%
- PI-controlled relative-velocity amplitude: 0.7981 m/s
- PI-controlled relative-velocity phase: 26.304 deg
- Active PI force amplitude: approximately 521.2 N

The PI controller reduces the passive-to-reference phasor error by
approximately 67.4%, compared with the analytical design target of 70%.

## Power Convention

For the controlled PTO,

`p_net = 540*v_rel^2 - F_PI*v_rel`

The reported 63.53 W value is net mechanical PTO power. It is not yet
net electrical output because generator, inverter, storage, and conversion
losses are not included.

## Results Reproduction

The passive-versus-PI comparison plots are generated using

`Results/plot_PI_vs_passive_report.m`

The processed summary values are stored in

`Results/passive_vs_PI_summary.csv`

and the final comparison figures are stored in

`Results/report_figures/`.

### Raw Simulation Data

The plotting script uses the following simulation result files:

- `PI_run_ramped.mat`
- `passive_SD_run.mat`

These files are generated simulation outputs and are not version-controlled
because of their size. They are excluded through the repository `.gitignore`.

To reproduce the passive-versus-PI comparison plots, generate or obtain these
two result files and place them in the `Results/` directory before running

`plot_PI_vs_passive_report.m`.

The processed summary values and final report figures are retained in the
repository through:

- `passive_vs_PI_summary.csv`
- `report_figures/`

## PI Controller Design Reproduction

The frequency-domain controller design and analysis used to develop the PI
controller are retained in this directory so that the design process can be
reproduced independently of the final WEC-Sim implementation.

### Plant Analysis

Run

`Plant_analysis.m`

to analyze the uncompensated WEC plant used for controller design. This script
contains the plant-level frequency-domain analysis used before PI compensation.

### Compensated-Loop Analysis

Run

`Compensated_loop_analysis.m`

to reproduce the compensated-loop analysis for the selected PI controller,
including the frequency-domain and closed-loop characteristics used to evaluate
the design.

The corresponding controller-design figures are stored in

`PI_Controller_Analysis_Figures/`

and include:

- `Gu_bode_omega4.png`
- `Gu_pole_zero_map.png`
- `Gu_root_locus_uncompensated.png`
- `PI_compensated_bode_nominal.png`
- `PI_compensated_nyquist_nominal.png`
- `PI_compensated_root_locus.png`
- `PI_sensitivity_complementary_sensitivity.png`

Together, `Plant_analysis.m`, `Compensated_loop_analysis.m`, and the retained
analysis figures document the design path from the uncompensated plant to the
final PI controller used in the WEC-Sim/Simulink model.
