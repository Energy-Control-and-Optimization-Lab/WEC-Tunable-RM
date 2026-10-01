CASE 2 - SINGLE VOLTAGE PI          (regular 6 V / 0.6667 Hz wave)
================================================================================

WHAT THIS IS
  One feedback loop. The battery voltage is compared with the 12 V reference
  and a PI controller sets the duty cycle directly. The simplest controller
  here that actually regulates.

FILES
  Case2_Regular.slx   the Simulink model
  RUN_ME.m   open it and press Run (F5). It prepares the wave, simulates 10 s,
             prints the results and draws the figure. Nothing else is needed.

See the paper for the controller gains and the results.
