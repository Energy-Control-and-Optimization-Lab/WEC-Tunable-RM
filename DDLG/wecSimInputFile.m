%% =========================================================
%  WEC-Sim Input File — Direct Drive Linear Generator
%  Magnet: K&J Magnetics DY08-N52
%          2 in diameter (50.8 mm), 1/2 in thick (12.7 mm), Grade N52
%  Coil:   Arbor Scientific P8-6500
%          N=560, R=1.1Ω, L=13mH, D_int=5.74cm
%
%  Parameter names confirmed from official WEC-Sim example (RM3_DD_PTO):
%    directLinearGenerator.Rs          — winding resistance
%    directLinearGenerator.Ls          — winding inductance
%    directLinearGenerator.Bfric       — friction coefficient (negative convention)
%    directLinearGenerator.tau_p       — pole pitch
%    directLinearGenerator.lambda_fd   — d-axis flux linkage
%    directLinearGenerator.theta_d_0   — initial rotor angle
%    directLinearGenerator.lambda_sq_0 — initial q-axis flux = 0
%    directLinearGenerator.lambda_sd_0 — initial d-axis flux = lambda_fd (NOT zero)
%    directLinearGenerator.Rload       — load resistance (negative convention)
% =========================================================

%% Design parameters (EDIT HERE)
tau_p  = 0.0152;    % [m]   pole pitch = magnet thickness (12.7mm) + separation (2.5mm)
N      = 560;       % [-]   turns — Arbor P8-6500
Br     = 1.48;      % [T]   N52 grade
Dcoil  = 5.74/100;  % [m]   Arbor inner diameter (fixed — do not change)
Dmag   = 50.8/1000; % [m]   magnet outer diameter = 2 in = 50.8 mm
R_int  = Dcoil/2;   % [m]   coil inner radius = 0.0287 m
R_iman = Dmag/2;    % [m]   magnet outer radius = 0.0254 m
g_gap  = R_int - R_iman; % [m] radial air gap = 3.3 mm
R_wind = 1.1;       % [Ω]   Arbor datasheet
L_wind = 0.013;     % [H]   Arbor datasheet (13 mH)
B_fric = 0;         % [N·s/m] mechanical friction — 0=ideal, ~10=realistic prototype

B_pto_target = 540; % [kg/s] — validated from Capytaine results

%% Electromagnetic parameter calculation
k_g       = R_iman / (R_iman + g_gap);    % radial air gap correction
A_polo    = pi * R_iman^2;                % magnet cross-sectional area [m²]
k_fringe  = 0.75;                         % fringing loss factor (~25% additional loss)
Phi_max   = k_fringe * k_g * Br * A_polo; % peak flux per pole [Wb]
lambda_fd = N * Phi_max;                  % total flux linkage [Wb]
K         = (pi / tau_p) * lambda_fd;    % force constant [N/A]
Rload     = K^2 / B_pto_target - R_wind; % load resistance [Ω]

% Electrical time constant
tau_e     = L_wind / (R_wind + Rload);   % [s]

% Minimum coil length
stroke     = 0.5;
L_coil_min = stroke + 2 * tau_p;         % [m]

% Verification
B_pto_calc = K^2 / (R_wind + Rload);
eta_elec   = Rload / (R_wind + Rload);

fprintf('\n--- System: Direct Drive Linear Generator ---\n');
fprintf('k_g (air gap factor)       = %.3f  (loss: %.1f%%)\n', k_g, (1-k_g)*100);
fprintf('k_fringe                   = %.2f  (fringing loss: %.0f%%)\n', k_fringe, (1-k_fringe)*100);
fprintf('lambda_fd                  = %.4f Wb\n', lambda_fd);
fprintf('K                          = %.2f N/A\n', K);
fprintf('R_load                     = %.4f ohm\n', Rload);
fprintf('Computed B_pto             = %.1f kg/s  (target: %d kg/s)\n', B_pto_calc, B_pto_target);
fprintf('Electrical efficiency      = %.1f%%\n', eta_elec * 100);
fprintf('tau_e                      = %.2e s\n', tau_e);
fprintf('Recommended dt             = %.2e s\n', tau_e/5);
fprintf('Min. coil length           = %.3f m\n', L_coil_min);
fprintf('B_fric                     = %.2f N·s/m\n\n', B_fric);

if Rload < 0
    warning('Negative R_load — K too low. Increase N, Br, or decrease tau_p.');
elseif eta_elec < 0.5
    warning('Efficiency below 50%%. Consider increasing K to allow higher R_load.');
end

%% Simulation Data
simu = simulationClass();
simu.simMechanicsFile = 'ECO_RM_WEC_DDLG';
simu.startTime        = 0;
simu.rampTime         = 30;
simu.endTime          = 120;
simu.dt               = 0.00025;
simu.domainSize       = 10;

%% Wave Information
waves = waveClass('regular');
waves.period = 1.57;    % [s]  T = 2pi/omega = 2pi/4 ~ 1.57 s
waves.height = 0.32;    % [m]  wave amplitude eta = 0.16 m

%% Body Data
% Float
body(1)                 = bodyClass('hydroData/ECO_RM_WEC.h5');
body(1).geometryFile    = 'geometry/float.stl';
body(1).mass            = 'equilibrium';
body(1).inertia         = [1 1 1];
B_float                 = zeros(6,6);               % [N·s/m] heave-heave viscous damping on spar
B_float(3,3)            = 0;               % [N·s/m] heave-heave viscous damping on spar
body(1).linearDamping   = B_float;
% body(1).quadDrag.cd     = [0 0 0.15 0 0 0];
% body(1).quadDrag.area   = [0 0 pi*(0.35)^2 0 0 0]; % ajusta D al tuyo

% Spar/Plate
body(2)                 = bodyClass('hydroData/ECO_RM_WEC.h5');
body(2).geometryFile    = 'geometry/spar.stl';
body(2).mass            = 'equilibrium';
body(2).inertia         = [1 1 1];
B_spar                  = zeros(6,6);               % [N·s/m] heave-heave viscous damping on spar
B_spar(3,3)             = 257.64;               % [N·s/m] heave-heave viscous damping on spar
body(2).linearDamping   = B_spar;
% body(2).quadDrag.cd     = [0 0 2.8 0 0 0];
% body(2).quadDrag.area   = [0 0 pi*(0.5)^2 0 0 0]; % ajusta D al tuyo

%% Constraints
constraint(1) = constraintClass('Constraint1');
constraint(1).location = [0, 0, 0];

%% PTO — mechanical connector
pto(1) = ptoClass('PTO1');
pto(1).stiffness = 0;
pto(1).damping   = 0;               % force handled by ptoSim
pto(1).location  = [0, 0, 0];

%% ptoSim — Direct Drive Linear Generator
ptoSim(1) = ptoSimClass('ddLinearGen');
ptoSim(1).directLinearGenerator.Rs          = R_wind;
ptoSim(1).directLinearGenerator.Ls          = L_wind;
ptoSim(1).directLinearGenerator.Rload       = -Rload;      % negative convention
ptoSim(1).directLinearGenerator.Bfric       = -B_fric;     % negative convention
ptoSim(1).directLinearGenerator.tau_p       = tau_p;
ptoSim(1).directLinearGenerator.lambda_fd   = lambda_fd;
ptoSim(1).directLinearGenerator.theta_d_0   = 0;
ptoSim(1).directLinearGenerator.lambda_sq_0 = 0;
ptoSim(1).directLinearGenerator.lambda_sd_0 = lambda_fd;   % must equal lambda_fd

% Confirm values sent to block
fprintf('--- Parameters sent to block ---\n');
fprintf('Rs     = %.4f Ω\n',     ptoSim(1).directLinearGenerator.Rs);
fprintf('Rload  = %.4f Ω\n',     ptoSim(1).directLinearGenerator.Rload);
fprintf('Bfric  = %.4f N·s/m\n', ptoSim(1).directLinearGenerator.Bfric);
fprintf('tau_p  = %.4f m\n',     ptoSim(1).directLinearGenerator.tau_p);
fprintf('lam_fd = %.4f Wb\n\n',  ptoSim(1).directLinearGenerator.lambda_fd);

%% Post-simulation power analysis
% Run in Command Window after wecSim completes:
%
% t      = output.ptoSim.time;
% idx    = t > simu.rampTime;
% v      = output.ptoSim.vel;
% fricF  = output.ptoSim.fricForce;
% force  = output.ptoSim.force;
% P_mec  = mean(output.ptoSim.absPower(idx,end));
% P_elec = mean(abs(output.ptoSim.elecPower(idx,end)));
%
% fprintf('\n--- Power Results ---\n');
% fprintf('v_max     = %.4f m/s\n',  max(abs(v(idx,end))));
% fprintf('fricF_max = %.4f N\n',    max(abs(fricF(idx,end))));
% fprintf('force_max = %.4f N\n',    max(abs(force(idx,end))));
% fprintf('P_mec     = %.4f W\n',    P_mec);
% fprintf('P_elec    = %.4f W\n',    P_elec);
% fprintf('eta       = %.2f%%\n',    P_elec/P_mec*100);
% fprintf('eta_exp   = %.2f%%\n',    eta_elec*100);
