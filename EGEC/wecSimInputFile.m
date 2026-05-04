%% =========================================================
%  WEC-Sim Input File — Electric Generator Equivalent Circuit
%  Motor: Akribis ADR175-B143 (Series winding)
%  Transmission: Rack & Pinion
%
%  PTO damping model (quasi-static, La·di/dt ≈ 0):
%
%    F_PTO = [Ke² / (r² · (Ra + R_load))] · v_rel  =  B_pto · v_rel
%
%  Target: B_pto = 540 kg/s  (validated from Capytaine)
% =========================================================

clear ll
close all
clc

%% ── Motor parameters — ADR175-B143 (per-phase, Y-equivalent) ─
Kt_motor = 5.0;           % [N·m/Arms]  torque constant
Ke_LL    = 0.43;          % [Vpeak/rpm] back-EMF constant (line-to-line)
R_LL     = 1.78;          % [Ω]         line-to-line resistance
L_LL     = 10.0e-3;       % [H]         line-to-line inductance
Jr       = 5.422e-3;      % [kg·m²]     rotor inertia
Tcn      = 16.0;          % [N·m]       continuous torque @100°C
rpm_max  = 880;           % [rpm]       max speed @continuous torque

% Per-phase conversion (Y/star equivalent)
Ra = R_LL / 2;             % [Ω]
La = L_LL / 2;             % [H]
Ke = Ke_LL * 60 / (2*pi); % [V/(rad/s)]

%% ── Rack & pinion ────────────────────────────────────────
r  = 0.043;                 % [m]     pinion radius
%r  = 0.05;                 % [m]     pinion radius
Ng = 1 / r;                % [rad/m] gear ratio

%% ── B_pto target → R_load ────────────────────────────────
B_pto_target = 540;        % [kg/s]

R_load  = Ke^2 / (B_pto_target * r^2) - Ra;   % [Ω]
R_total = Ra + R_load;                          % [Ω]

% Verification
B_pto_calc = Ke^2 / (R_total * r^2);
eta_elec   = R_load / R_total;
rpm_op     = (1/r) * 60 / (2*pi);
T_req      = B_pto_target * r^2 * (1/r);

fprintf('\n--- ADR175-B143 Electric Generator Equivalent Circuit ---\n');
fprintf('Ra (per-phase)       = %.4f ohm\n', Ra);
fprintf('La (per-phase)       = %.6f H\n',   La);
fprintf('Ke                   = %.4f V/(rad/s)\n', Ke);
fprintf('Gear ratio Ng        = %.1f rad/m\n', Ng);
fprintf('R_load               = %.6f ohm\n', R_load);
fprintf('Computed B_pto       = %.1f kg/s  (target: %d kg/s)\n', B_pto_calc, B_pto_target);
fprintf('Electrical efficiency= %.1f%%\n',   eta_elec * 100);
fprintf('Operating RPM        = %.1f rpm  (limit: %d rpm)\n', rpm_op, rpm_max);
fprintf('Required torque      = %.2f N·m  (limit: %.1f N·m)\n', T_req, Tcn);

if R_load < 0
    warning('Negative R_load — increase r or use a motor with higher Ke.');
elseif rpm_op > rpm_max
    warning('RPM exceeds motor limit — increase r.');
elseif T_req > Tcn
    warning('Required torque exceeds continuous rating — increase r or reduce B_pto.');
elseif T_req > 0.8*Tcn
    warning('Required torque is >80%% of continuous rating. Consider increasing r slightly.');
end

%% ── Push all parameters into Simulink block masks ────────
modelName = 'ECO_RM_WEC_EGEC';
egec      = [modelName '/Electric Generator Equivalent Circuit'];
const     = [modelName '/Constant'];

load_system(modelName);

% EGEC mask parameters (Ra, La, Ke, bShaft, Jem)
set_param(egec, 'Ke',     num2str(Ke, '%.6f'));
set_param(egec, 'Ra',     num2str(Ra, '%.6f'));
set_param(egec, 'La',     num2str(La, '%.8f'));
set_param(egec, 'bShaft', '0');
set_param(egec, 'Jem',    num2str(Jr, '%.6e'));

% Constant block (R_load)
set_param(const, 'Value', num2str(R_load, '%.6f'));

save_system(modelName);

fprintf('\n>>> Simulink block parameters updated and saved:\n');
fprintf('    Ke     = %.4f V/(rad/s)\n', Ke);
fprintf('    Ra     = %.4f ohm\n',       Ra);
fprintf('    La     = %.6f H\n',         La);
fprintf('    Jem    = %.4e kg·m²\n',     Jr);
fprintf('    R_load = %.6f ohm\n\n',     R_load);

%% ── Simulation Data ──────────────────────────────────────
simu = simulationClass();
simu.simMechanicsFile = modelName;
simu.startTime        = 0;
simu.rampTime         = 30;
simu.endTime          = 120;
simu.dt               = 0.00025;
simu.domainSize       = 10;

%% ── Wave Information ─────────────────────────────────────
waves = waveClass('regular');
waves.period = 1.57;    % [s]
waves.height = 0.32;    % [m]

%% ── Body Data ────────────────────────────────────────────
% Float
body(1) = bodyClass('hydroData/ECO_RM_WEC.h5');
body(1).geometryFile  = 'geometry/float.stl';
body(1).mass          = 'equilibrium';
body(1).inertia       = [1 1 1];
body(1).linearDamping = zeros(6,6);

% Spar/Plate
body(2) = bodyClass('hydroData/ECO_RM_WEC.h5');
body(2).geometryFile  = 'geometry/spar.stl';
body(2).mass          = 'equilibrium';
body(2).inertia       = [1 1 1];
B_spar        = zeros(6,6);
B_spar(3,3)   = 257.64;               % [N·s/m] heave viscous damping
body(2).linearDamping = B_spar;

%% ── Constraints ──────────────────────────────────────────
constraint(1) = constraintClass('Constraint1');
constraint(1).location = [0, 0, 0];

%% ── PTO ──────────────────────────────────────────────────
pto(1) = ptoClass('PTO1');
pto(1).stiffness = 0;
pto(1).damping   = 0;
pto(1).location  = [0, 0, 0];

%% ── ptoSim — Electric Generator Equivalent Circuit ───────
ptoSim(1) = ptoSimClass('ptoSim');
ptoSim(1).type = 5;

ptoSim(1).electricGeneratorEC.Ra     = Ra;
ptoSim(1).electricGeneratorEC.La     = La;
ptoSim(1).electricGeneratorEC.Ke     = Ke;
ptoSim(1).electricGeneratorEC.bShaft = 0;
ptoSim(1).electricGeneratorEC.Jem    = Jr;
