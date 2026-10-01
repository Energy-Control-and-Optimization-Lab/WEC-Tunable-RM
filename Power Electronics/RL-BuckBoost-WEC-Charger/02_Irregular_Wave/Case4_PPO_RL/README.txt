CASE 4 - PPO REINFORCEMENT LEARNING          (irregular JONSWAP wave)
================================================================================

WHAT THIS IS
  A controller whose policy is a neural network trained with proximal policy
  optimization (PPO) on the switched circuit. The policy is already trained:
  this folder runs it, it does not train it. RUN_ME switches exploration off,
  so the run is deterministic and repeats exactly.

  Needs the Reinforcement Learning Toolbox and the Deep Learning Toolbox.

FILES
  Case4_Irregular.slx         the Simulink model
  ppo_agent_irregular.mat     the trained policy
  RUN_ME.m   open it and press Run (F5). It prepares the wave, simulates 10 s,
             prints the results and draws the figure. Nothing else is needed.

See the paper for the controller gains and the results.
