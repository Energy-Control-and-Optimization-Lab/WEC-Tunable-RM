%% Simulation Data
simu = simulationClass();
simu.simMechanicsFile = 'ECO_RM_WEC_SD.slx';
simu.startTime        = 0;
simu.rampTime         = 30;
simu.endTime          = 120;
simu.dt               = 0.00025;
simu.domainSize       = 10;

%% Wave Information
waves = waveClass('regular');
waves.period = 3.93;    % [s]  T = 2pi/omega = 2pi/4 ~ 1.57 s
waves.height = 0.1;    % [m]  wave amplitude eta = 0.16 m

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

%% PTO
pto(1) = ptoClass('PTO1');
pto(1).stiffness = 0;       % [N/m]
pto(1).damping   = 540;     % [N·s/m]
pto(1).location  = [0, 0, 0];