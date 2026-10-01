CASE 1 - FIXED DUTY CYCLE          (regular 6 V / 0.6667 Hz wave)
================================================================================

WHAT THIS IS
  The baseline. The converter switch runs at a constant duty cycle with no
  feedback of any kind: no voltage sensor, no current sensor, no controller.
  It shows what the circuit does on its own.

FILES
  Case1_Regular.slx   the Simulink model
  RUN_ME.m   open it and press Run (F5). It prepares the wave, simulates 10 s,
             prints the results and draws the figure. Nothing else is needed.

See the paper for the controller gains and the results.
