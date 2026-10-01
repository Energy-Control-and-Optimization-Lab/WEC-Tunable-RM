================================================================================
 REINFORCEMENT LEARNING CONTROL OF A BUCK-BOOST BATTERY CHARGER
 FOR A TWO-BODY WAVE ENERGY CONVERTER
================================================================================

 Simulation code accompanying the paper of the same title.

 M. Farrukh, P. A. Matamala-Carvajal, A. Hasankhani
 University of New Hampshire, Durham, NH 03824, USA

 Supported by the U.S. Department of Energy under award DE-EE-0011379 and by
 the Atlantic Marine Energy Center (AMEC), in partnership with the ECOLab
 research group at the University of New Hampshire.

 License: MIT (see LICENSE.txt)


--------------------------------------------------------------------------------
 WHAT IS IN HERE
--------------------------------------------------------------------------------

 Eight Simulink simulations: four controllers, each run on two sea states.

 All eight use the same power circuit, so only the controller and the wave
 change between them. See the paper for the circuit, the gains and the results.

   Case 1   Fixed duty cycle   open loop, no feedback
   Case 2   Voltage PI         one loop
   Case 3   Cascaded PI        two nested loops
   Case 4   PPO                a trained reinforcement-learning policy

   01_Regular_Wave      sinusoidal generator EMF, 6 V peak, 0.6667 Hz
   02_Irregular_Wave    JONSWAP sea state, Hs = 0.25 m, Tp = 1.5 s


--------------------------------------------------------------------------------
 WHAT YOU NEED
--------------------------------------------------------------------------------

   MATLAB R2024b
   Simulink
   Simscape
   Simscape Electrical
   Reinforcement Learning Toolbox   ) Case 4 only
   Deep Learning Toolbox            )

 Use R2024b. Release R2026a removed the Specialized Power Systems library, so
 the models will not compile there.


--------------------------------------------------------------------------------
 HOW TO RUN IT
--------------------------------------------------------------------------------

   1. Open MATLAB.
   2. Browse to any case folder, for example  01_Regular_Wave/Case4_PPO_RL
   3. Open RUN_ME.m and press Run (F5).

 That is all. The script prepares the wave, loads the controller, simulates
 10 s, prints the results in the Command Window and draws a figure of the
 generator voltage, battery voltage, battery current, duty cycle and state of
 charge. You do not have to set a path, define a variable, or open the model
 first.

 From the Command Window instead:

   addpath('00_Setup')
   run_case('regular',   3)
   run_case('irregular', 4)

 To run all eight and write a summary table:

   addpath('00_Setup'); run_all

 Each run simulates 10 s of a switching circuit at a 1 microsecond step, so it
 takes roughly 10-25 minutes. A single RUN_ME is the quick way to see one
 result; run_all takes a few hours.


--------------------------------------------------------------------------------
 FOLDERS
--------------------------------------------------------------------------------

 README.txt         this file
 LICENSE.txt        MIT license

 00_Setup/          run_case.m             runs one case
                    run_all.m              runs all eight
                    make_irregular_wave.m  builds the JONSWAP wave record

 01_Regular_Wave/   Case1_Fixed_Duty, Case2_Voltage_PI,
 02_Irregular_Wave/ Case3_Cascaded_PI, Case4_PPO_RL

 Every case folder holds its model, a RUN_ME.m, and a short README.txt. The
 Case 4 folders also hold the trained policy as a .mat file.


--------------------------------------------------------------------------------
 IF SOMETHING GOES WRONG
--------------------------------------------------------------------------------

 A Simscape or 'powerlib' compile error
     You are most likely on R2026a or later. Use R2024b.

 "Undefined function or variable 'emf_ts'"
     An irregular-wave model was run straight from Simulink instead of through
     RUN_ME.m. Use RUN_ME.m.

 "Invalid setting ... for parameter 'Agent'"
     Same cause in Case 4: the trained policy was never loaded. Use RUN_ME.m.

 Out of memory
     run_case('regular', 4, 'Decimation', 100)
     or shorten the run with 'StopTime', 4.

================================================================================
