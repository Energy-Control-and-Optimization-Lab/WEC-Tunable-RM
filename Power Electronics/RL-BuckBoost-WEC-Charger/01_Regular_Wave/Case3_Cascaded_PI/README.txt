CASE 3 - CASCADED (DOUBLE) PI          (regular 6 V / 0.6667 Hz wave)
================================================================================

WHAT THIS IS
  Two nested loops, the standard structure for a switching converter. A slow
  outer loop regulates the battery voltage and sets an inductor-current
  reference; a fast inner loop follows that reference and sets the duty cycle.

FILES
  Case3_Regular.slx   the Simulink model
  RUN_ME.m   open it and press Run (F5). It prepares the wave, simulates 10 s,
             prints the results and draws the figure. Nothing else is needed.

See the paper for the controller gains and the results.
