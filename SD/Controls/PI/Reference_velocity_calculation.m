function [Results, ref] = Reference_velocity_calculation( ...
    omega0, H, T, sdWorkspace)

% Purpose:
%   1) Import the ACTUAL relative PTO velocity from the passive
%      Spring-Damper WEC-Sim simulation.
%
%   2) Extract its steady-state amplitude and phase.
%
%   3) Build the two-body relative-port impedance Z_rel from the
%      WEC-Sim/BEMIO H5 hydrodynamic data at the requested frequency.
%
%   4) Infer the effective excitation acting at the relative-motion
%      port from the validated SD response.
%
%   5) Compute the ideal energy-optimal relative-velocity reference
%      using complex-conjugate matching.
%
%   6) Compare the SD relative velocity and optimal reference in:
%         - velocity amplitude
%         - phase
%         - mean absorbed mechanical power
%         - normalized velocity
%         - normalized power
%
% Relative-velocity convention used in the analytical model:
%       v_rel = v_float - v_spar
%
% INPUTS:
%   omega0      angular wave frequency [rad/s]
%   H           WEC-Sim full wave height [m]
%   T           wave period [s]
%   sdWorkspace optional path to saved SD WEC-Sim workspace
%
% EXAMPLE to run:
%
%   [Results, ref] = Reference_velocity_calculation( ...
%       4.0, 0.32, 2*pi/4);
% ------------------------------------------------------------


% 1. BASIC INPUT CHECKS
clc;
rho = 1025;       % kg/m^3
g   = 9.81;       % m/s^2
eta = H/2;        % wave amplitude [m]
omega_from_T = 2*pi/T;

fprintf(' REFERENCE VELOCITY CALCULATION\n');
fprintf('\nRequested operating condition:\n');
fprintf('omega0 = %.6f rad/s\n',omega0);
fprintf('T      = %.6f s\n',T);
fprintf('H      = %.6f m\n',H);
fprintf('eta    = %.6f m\n',eta);

% Frequency-period consistency check
tolOmega = 1e-4;
if abs(omega0 - omega_from_T) > tolOmega
    error(['Inconsistent operating point:\n' ...
           'omega0     = %.6f rad/s\n' ...
           '2*pi/T     = %.6f rad/s\n' ...
           'Check omega0 and T before continuing.'], ...
           omega0,omega_from_T);
end
% ------------------------------------------------------------


% 2. GET SD WEC-SIM DATA
%
% Preferred source:
%   Use the WEC-Sim simulation that was just run and is still
%   available in the MATLAB base workspace.
%
% Optional fallback:
%   A saved MATLAB workspace can still be supplied as the 4th input.

useLiveWorkspace = false;

% Check whether a completed WEC-Sim run exists in base workspace

hasOutput = evalin('base','exist(''output'',''var'') == 1');
hasWaves  = evalin('base','exist(''waves'',''var'')  == 1');
hasPto    = evalin('base','exist(''pto'',''var'')    == 1');
hasBody   = evalin('base','exist(''body'',''var'')   == 1');
hasSimu   = evalin('base','exist(''simu'',''var'')   == 1');

if hasOutput
    fprintf('\nUsing completed WEC-Sim run from MATLAB base workspace.\n');
    S = struct;
    S.output = evalin('base','output');
    if hasWaves
        S.waves = evalin('base','waves');
    end
    if hasPto
        S.pto = evalin('base','pto');
    end
    if hasBody
        S.body = evalin('base','body');
    end
    if hasSimu
        S.simu = evalin('base','simu');
    end
    useLiveWorkspace = true;
else

    % No live WEC-Sim output found.
    % Try an explicitly supplied saved workspace.
    if nargin < 4 || isempty(sdWorkspace)
        error([ ...
            'No completed WEC-Sim output was found in the MATLAB base workspace.\n' ...
            'Run the passive SD WEC-Sim case first, then call this function.\n' ...
            'Alternatively, provide a saved WEC-Sim workspace as the 4th input.']);
    end

    assert(isfile(sdWorkspace), ...
        'Could not find saved SD workspace: %s',sdWorkspace);
    fprintf('\nLoading saved SD workspace:\n%s\n',sdWorkspace);
    S = load(sdWorkspace);
end
% ------------------------------------------------------------


% 3. CHECK THAT WEC-SIM OUTPUT EXISTS

assert(isfield(S,'output'), ...
    'The WEC-Sim data do not contain variable "output".');

outputSD = S.output;
% ------------------------------------------------------------


% 4. CHECK THAT SD RUN MATCHES REQUESTED WAVE CONDITION
if isfield(S,'waves')
    T_SD = S.waves.period;
    H_SD = S.waves.height;

    fprintf('\nSaved SD operating condition:\n');
    fprintf('T_SD = %.6f s\n',T_SD);
    fprintf('H_SD = %.6f m\n',H_SD);

    tolT = 1e-3;   % [s]

    if abs(T_SD - T) > tolT

        error([ ...
         'The SD simulation period does not match the requested period.\n' ...
         'Requested T = %.6f s\n' ...
         'Saved SD T = %.6f s\n' ...
         'Difference = %.6f s'], ...
         T,T_SD,abs(T_SD-T));

    end

    if abs(H_SD - H) > 1e-5
        error([ ...
            'The SD simulation wave height does not match the requested H.\n' ...
            'Requested H = %.6f m\n' ...
            'Saved SD H = %.6f m'], ...
            H,H_SD);
    end
else

    warning(['Saved workspace does not contain "waves". ' ...
             'Wave-condition consistency could not be checked.']);

end
% ------------------------------------------------------------


% 5. IMPORT ACTUAL PTO RELATIVE VELOCITY FROM WEC-SIM
% This is the same PTO quantity used by the original SD analysis:
%       output.ptos(1).velocity(:,3)
% It is already dimensional [m/s].
% DO NOT divide by eta here.

t_SD = outputSD.ptos(1).time(:);
v_PTO_raw = outputSD.ptos(1).velocity(:,3);
p_PTO_raw = outputSD.ptos(1).powerInternalMechanics(:,3);
v_PTO_raw = v_PTO_raw(:);
p_PTO_raw = p_PTO_raw(:);
% ------------------------------------------------------------


% 6. GET SD PTO DAMPING
if isfield(S,'pto')
    B_SD = S.pto(1).damping;
else
    B_SD = 540;
    warning([ ...
        'Saved workspace does not contain "pto". ' ...
        'Using B_SD = 540 N*s/m.']);
end

fprintf('\nPassive SD PTO damping:\n');
fprintf('B_SD = %.6f N*s/m\n',B_SD);
% ------------------------------------------------------------


% 7. GET BODY MASSES AND VISCOUS DAMPING FROM SD WORKSPACE
if isfield(S,'body')

    m1 = S.body(1).mass;
    m2 = S.body(2).mass;
    Bv1 = S.body(1).linearDamping(3,3);
    Bv2 = S.body(2).linearDamping(3,3);

else

    warning([ ...
        'Saved workspace does not contain "body". ' ...
        'Using known baseline mass/damping values.']);

    m1 = 35.1977;
    m2 = 70.7086;
    Bv1 = 0;
    Bv2 = 257.64;

end

M = [m1 0;
     0  m2];

Bv = [Bv1 0;
      0    Bv2];
% ------------------------------------------------------------


% 8. LOCATE HYDRODYNAMIC H5 FILE
if useLiveWorkspace

    % Most likely location:
    % hydroData folder in the directory from which WEC-Sim was run

    candidate1 = fullfile( ...
        pwd, ...
        'hydroData', ...
        'ECO_RM_WEC.h5');

    % Second possibility:
    % hydroData folder beside this function

    functionFolder = fileparts(mfilename('fullpath'));
    candidate2 = fullfile( ...
        functionFolder, ...
        'hydroData', ...
        'ECO_RM_WEC.h5');

    % Third possibility:
    % function is one level above an SD folder

    candidate3 = fullfile( ...
        functionFolder, ...
        'SD', ...
        'hydroData', ...
        'ECO_RM_WEC.h5');

    if isfile(candidate1)

        h5File = candidate1;

    elseif isfile(candidate2)

        h5File = candidate2;

    elseif isfile(candidate3)

        h5File = candidate3;

    else

        error([ ...
            'Could not locate ECO_RM_WEC.h5.\n\n' ...
            'Checked:\n%s\n%s\n%s'], ...
            candidate1, ...
            candidate2, ...
            candidate3);

    end

else

    % Saved-workspace fallback

    outputFolder = fileparts(sdWorkspace);
    sdFolder     = fileparts(outputFolder);

    h5File = fullfile( ...
        sdFolder, ...
        'hydroData', ...
        'ECO_RM_WEC.h5');

end

assert(isfile(h5File), ...
    'Could not find hydrodynamic H5 file: %s',h5File);

fprintf('\nHydrodynamic file:\n%s\n',h5File);
% ------------------------------------------------------------


% 9. ADDED MASS MATRIX A(omega)
%
% Body-1 heave -> global DOF 3
% Body-2 heave -> global DOF 9
%
% BEMIO normalization:
%       A_bar = A/rho
%
% therefore:
%       A = rho*A_bar

A11raw = h5read(h5File, ...
    '/body1/hydro_coeffs/added_mass/components/3_3');

A12raw = h5read(h5File, ...
    '/body1/hydro_coeffs/added_mass/components/3_9');

A21raw = h5read(h5File, ...
    '/body2/hydro_coeffs/added_mass/components/3_3');

A22raw = h5read(h5File, ...
    '/body2/hydro_coeffs/added_mass/components/3_9');

% Check interpolation range

wMin = max([ ...
    min(A11raw(1,:)), ...
    min(A12raw(1,:)), ...
    min(A21raw(1,:)), ...
    min(A22raw(1,:))]);

wMax = min([ ...
    max(A11raw(1,:)), ...
    max(A12raw(1,:)), ...
    max(A21raw(1,:)), ...
    max(A22raw(1,:))]);

if omega0 < wMin || omega0 > wMax

    error([ ...
        'omega0 = %.6f rad/s lies outside the H5 frequency range ' ...
        '[%.6f, %.6f] rad/s.'], ...
        omega0,wMin,wMax);

end

A11 = rho*interp1( ...
    A11raw(1,:),A11raw(2,:),omega0,'linear');

A12 = rho*interp1( ...
    A12raw(1,:),A12raw(2,:),omega0,'linear');

A21 = rho*interp1( ...
    A21raw(1,:),A21raw(2,:),omega0,'linear');

A22 = rho*interp1( ...
    A22raw(1,:),A22raw(2,:),omega0,'linear');

A = [A11 A12;
     A21 A22];
% ------------------------------------------------------------


% 10. RADIATION DAMPING MATRIX B(omega)
%
% BEMIO normalization:
%       B_bar = B/(rho*omega)
% therefore:
%       B = rho*omega*B_bar

B11raw = h5read(h5File, ...
    '/body1/hydro_coeffs/radiation_damping/components/3_3');

B12raw = h5read(h5File, ...
    '/body1/hydro_coeffs/radiation_damping/components/3_9');

B21raw = h5read(h5File, ...
    '/body2/hydro_coeffs/radiation_damping/components/3_3');

B22raw = h5read(h5File, ...
    '/body2/hydro_coeffs/radiation_damping/components/3_9');

B11 = rho*omega0*interp1( ...
    B11raw(1,:),B11raw(2,:),omega0,'linear');

B12 = rho*omega0*interp1( ...
    B12raw(1,:),B12raw(2,:),omega0,'linear');

B21 = rho*omega0*interp1( ...
    B21raw(1,:),B21raw(2,:),omega0,'linear');

B22 = rho*omega0*interp1( ...
    B22raw(1,:),B22raw(2,:),omega0,'linear');

B = [B11 B12;
     B21 B22];
% ------------------------------------------------------------


% 11. HYDROSTATIC RESTORING MATRIX C
%
% BEMIO normalization:
%       C_bar = C/(rho*g)
% therefore:
%       C = rho*g*C_bar

C1raw = h5read(h5File, ...
    '/body1/hydro_coeffs/linear_restoring_stiffness');

C2raw = h5read(h5File, ...
    '/body2/hydro_coeffs/linear_restoring_stiffness');

C11 = rho*g*C1raw(3,3);
C22 = rho*g*C2raw(3,3);

C = [C11 0;
     0   C22];
% ------------------------------------------------------------


% 12. FULL TWO-BODY INTRINSIC IMPEDANCE
% Velocity-domain form:
% Z = B + Bv + j*[omega*(M+A) - C/omega]

Z = (B + Bv) ...
    + 1i*(omega0*(M + A) - C/omega0);
% ------------------------------------------------------------


% 13. RELATIVE-MOTION PORT IMPEDANCE
% Analytical convention:
%       v_rel = v_float - v_spar
%       d = [1; -1]
%       Z_rel = 1/(d^T Z^-1 d)

d = [1;
    -1];

Y_rel = d.'*(Z\d);

Z_rel = 1/Y_rel;

R_rel = real(Z_rel);
X_rel = imag(Z_rel);

if R_rel <= 0

    error([ ...
        'Equivalent relative resistance is not positive.\n' ...
        'R_rel = %.6f N*s/m'],R_rel);

end

fprintf('\nEquivalent relative-port impedance:\n');
fprintf('Z_rel = %.6f %+.6fj N*s/m\n', ...
    R_rel,X_rel);
% ------------------------------------------------------------


% 14. DEFINE STEADY-STATE WINDOW
%
% Original SD analysis uses data after the WEC-Sim ramp.

if isfield(S,'simu')
    rampTime = S.simu.rampTime;
else

    rampTime = 30;
    warning([ ...
        'Saved workspace does not contain "simu". ' ...
        'Using rampTime = 30 s.']);

end

idxSS = t_SD >= rampTime;

if nnz(idxSS) < 20

    error('Not enough samples after the ramp time.');

end

t_SS = t_SD(idxSS);
v_SS = v_PTO_raw(idxSS);
p_SS = p_PTO_raw(idxSS);
% ------------------------------------------------------------


% 15. HARMONIC FIT OF ACTUAL WEC-SIM PTO RELATIVE VELOCITY
% Fit:
%       v(t) = a*cos(wt) + b*sin(wt) + c
% Phasor convention:
%       v(t) = Re{V*exp(j*w*t)}
% therefore:
%       V = a - j*b

Xfit = [ ...
    cos(omega0*t_SS), ...
    sin(omega0*t_SS), ...
    ones(size(t_SS))];

beta = Xfit\v_SS;

a_cos = beta(1);
b_sin = beta(2);
v_mean = beta(3);

V_SD_PTO = a_cos - 1i*b_sin;
A_SD_fit = abs(V_SD_PTO);
phi_SD_PTO = angle(V_SD_PTO);
phi_SD_PTO_deg = rad2deg(phi_SD_PTO);
% ------------------------------------------------------------


% 16. REPRODUCE ORIGINAL PAPER-STYLE AMPLITUDE ESTIMATE
%
% Original analysis:
%
%       signal = abs(v)
%       pks = findpeaks(signal)
%       amplitude = mean(pks)
%
% This is retained only as a check.

if exist('findpeaks','file') == 2
    [pks,~] = findpeaks(abs(v_SS));
    if ~isempty(pks)
        A_SD_peak = mean(pks);
    else
        A_SD_peak = max(abs(v_SS));
    end

else
    A_SD_peak = max(abs(v_SS));
end
% ------------------------------------------------------------


% 17. ACTUAL WEC-SIM SD MECHANICAL POWER
% Same convention as original SD analysis:
%       P_SD = abs(mean(powerInternalMechanics))

P_SD = abs(mean(p_SS));

% Independent sinusoidal damper check
P_SD_fromAmplitude = ...
    0.5*B_SD*A_SD_fit^2;
% ------------------------------------------------------------


% 18. VERIFY PTO VELOCITY SIGN AGAINST BODY RELATIVE VELOCITY
% Analytical convention:
%       v_rel = v_float - v_spar
% WEC-Sim PTO output sign can be checked directly.

ptoSign = 1;
signCheckAvailable = false;

try
    v_float = outputSD.bodies(1).velocity(:,3);
    v_spar  = outputSD.bodies(2).velocity(:,3);

    v_body_diff = v_float(:) - v_spar(:);

    if numel(v_body_diff) == numel(v_PTO_raw)

        err_same = sqrt(mean( ...
            (v_PTO_raw(idxSS) ...
            - v_body_diff(idxSS)).^2));

        err_flip = sqrt(mean( ...
            (v_PTO_raw(idxSS) ...
            + v_body_diff(idxSS)).^2));

        signCheckAvailable = true;

        if err_same <= err_flip

            ptoSign = 1;

            fprintf('\nPTO sign check:\n');
            fprintf('output.ptos velocity matches v_float - v_spar.\n');

        else
            ptoSign = -1;

            fprintf('\nPTO sign check:\n');
            fprintf(['output.ptos velocity matches ' ...
                     '-(v_float - v_spar).\n']);
        end
    end

catch
    warning([ ...
        'Could not verify PTO velocity sign using body velocities. ' ...
        'Assuming output.ptos uses v_float - v_spar.']);
end
% ------------------------------------------------------------


% 19. CONVERT MEASURED PTO PHASOR TO ANALYTICAL SIGN CONVENTION

V_SD_model = ptoSign*V_SD_PTO;
% ------------------------------------------------------------


% 20. IDENTIFY EFFECTIVE RELATIVE-PORT EXCITATION FROM SD RESPONSE
% Relative-port equation:
%       Z_rel*V = F_eff + F_PTO
% Passive SD PTO:
%       F_PTO = -B_SD*V
% therefore:
%       (Z_rel + B_SD)*V_SD = F_eff
% This anchors F_eff to the ACTUAL validated SD WEC-Sim response.

F_eff_SD = ...
    (Z_rel + B_SD)*V_SD_model;
% ------------------------------------------------------------


% 21. ENERGY-OPTIMAL REFERENCE
% Complex-conjugate optimum:
%       Z_PTO,opt = conj(Z_rel)
% therefore:
%
%                     F_eff
%       V_ref = -------------------
%                 2*Re{Z_rel}
%

Z_PTO_opt = conj(Z_rel);

V_ref_model = ...
    F_eff_SD/(2*R_rel);

% Convert reference back to WEC-Sim PTO sign convention
% so the plotted reference can be compared directly with
% output.ptos(1).velocity(:,3).

V_ref = ptoSign*V_ref_model;
A_ref = abs(V_ref);
phi_ref = angle(V_ref);
phi_ref_deg = rad2deg(phi_ref);
% ------------------------------------------------------------


% 22. IDEAL OPTIMAL MECHANICAL POWER
% Only the resistive part of the optimal PTO absorbs mean power.
%       Re{Z_PTO,opt} = R_rel
% therefore:
%       P_opt = 0.5*R_rel*|V_ref|^2

P_ref = ...
    0.5*R_rel*A_ref^2;

% Equivalent analytical check
P_ref_check = ...
    abs(F_eff_SD)^2/(8*R_rel);
% ------------------------------------------------------------


% 23. NORMALIZED QUANTITIES
% The original RAO plots normalize velocity by wave amplitude eta:
%       |V|/eta     [1/s]
% and power by eta^2:
%       P/eta^2     [W/m^2]
%
% Keep all primary calculations DIMENSIONAL.
% Normalize only here for comparison/plotting.

A_SD_norm = ...
    A_SD_fit/eta;

A_ref_norm = ...
    A_ref/eta;

P_SD_norm = ...
    P_SD/(eta^2);

P_ref_norm = ...
    P_ref/(eta^2);
% ------------------------------------------------------------


% 24. PHASE DIFFERENCE
phaseDifference_deg = ...
    rad2deg(angle(V_ref/V_SD_PTO));

phaseDifference_deg = ...
    mod(phaseDifference_deg + 180,360) - 180;
% ------------------------------------------------------------


% 25. PRINT DIAGNOSTICS
fprintf('\n====================================================\n');
fprintf(' ACTUAL SD PTO RESPONSE\n');
fprintf('====================================================\n');

fprintf('Steady-state window starts at t = %.3f s\n', ...
    rampTime);

fprintf('\nHarmonic-fit relative velocity:\n');

fprintf('V_SD = %.6f %+.6fj m/s\n', ...
    real(V_SD_PTO),imag(V_SD_PTO));

fprintf('|V_SD| = %.6f m/s\n',A_SD_fit);

fprintf('phase(V_SD) = %.6f deg\n', ...
    phi_SD_PTO_deg);

fprintf('\nOriginal peak-method amplitude check:\n');

fprintf('A_SD_peak = %.6f m/s\n', ...
    A_SD_peak);

fprintf('Difference fit vs peak = %.3f %%\n', ...
    100*(A_SD_fit - A_SD_peak)/A_SD_peak);

fprintf('\nSD power:\n');

fprintf('WEC-Sim mean absorbed power = %.6f W\n', ...
    P_SD);

fprintf('0.5*B_SD*|V_SD|^2          = %.6f W\n', ...
    P_SD_fromAmplitude);


fprintf('\n====================================================\n');
fprintf(' IDENTIFIED RELATIVE-PORT FORCING\n');
fprintf('====================================================\n');

fprintf('F_eff_SD = %.6f %+.6fj N\n', ...
    real(F_eff_SD),imag(F_eff_SD));

fprintf('|F_eff_SD| = %.6f N\n', ...
    abs(F_eff_SD));

fprintf('phase(F_eff_SD) = %.6f deg\n', ...
    rad2deg(angle(F_eff_SD)));


fprintf('\n====================================================\n');
fprintf(' ENERGY-OPTIMAL REFERENCE\n');
fprintf('====================================================\n');

fprintf('Z_PTO,opt = %.6f %+.6fj N*s/m\n', ...
    real(Z_PTO_opt),imag(Z_PTO_opt));

fprintf('\nV_ref = %.6f %+.6fj m/s\n', ...
    real(V_ref),imag(V_ref));

fprintf('|V_ref| = %.6f m/s\n', ...
    A_ref);

fprintf('phase(V_ref) = %.6f deg\n', ...
    phi_ref_deg);

fprintf('phase shift relative to SD = %.6f deg\n', ...
    phaseDifference_deg);

fprintf('\nIdeal optimal absorbed mechanical power:\n');

fprintf('P_ref       = %.6f W\n', ...
    P_ref);

fprintf('P_ref check = %.6f W\n', ...
    P_ref_check);
% ------------------------------------------------------------


% 26. RESULTS TABLE

Case = [
    "SD PTO relative velocity"
    "Energy-optimal reference"
    ];

VelocityAmplitude_mps = [
    A_SD_fit
    A_ref
    ];

Phase_deg = [
    phi_SD_PTO_deg
    phi_ref_deg
    ];

MeanAbsorbedPower_W = [
    P_SD
    P_ref
    ];

VelocityOverEta_1ps = [
    A_SD_norm
    A_ref_norm
    ];

PowerOverEta2_Wpm2 = [
    P_SD_norm
    P_ref_norm
    ];

Results = table( ...
    Case, ...
    VelocityAmplitude_mps, ...
    Phase_deg, ...
    MeanAbsorbedPower_W, ...
    VelocityOverEta_1ps, ...
    PowerOverEta2_Wpm2);

fprintf('\n====================================================\n');
fprintf(' COMPARISON TABLE\n');
fprintf('====================================================\n\n');

disp(Results);
% ------------------------------------------------------------


% 27. TIME-DOMAIN PLOT
%
% Plot ACTUAL WEC-Sim PTO velocity and ideal reference.
% Use the final five wave cycles for clarity.

nPlotCycles = 5;

tPlotStart = ...
    max(rampTime, t_SD(end) - nPlotCycles*T);

idxPlot = ...
    t_SD >= tPlotStart;

t_plot = ...
    t_SD(idxPlot);

v_SD_plot = ...
    v_PTO_raw(idxPlot);

v_ref_plot = ...
    real(V_ref.*exp(1i*omega0*t_plot));

figure;

plot( ...
    t_plot, ...
    v_SD_plot, ...
    'LineWidth',1.4);

hold on;

plot( ...
    t_plot, ...
    v_ref_plot, ...
    '--', ...
    'LineWidth',1.8);

yline(0,'k:');

grid on;

xlabel('Time [s]');

ylabel('Relative velocity [m/s]');

title(sprintf( ...
    ['PTO Relative Velocity vs Energy-Optimal Reference\n' ...
     '\\omega = %.3f rad/s, H = %.3f m, \\eta = %.3f m'], ...
    omega0,H,eta));

legend( ...
    'WEC-Sim SD PTO relative velocity', ...
    'Energy-optimal reference', ...
    'Location','best');
% ------------------------------------------------------------


% 28. ABSORBED MECHANICAL POWER COMPARISON
%
% Compare the passive SD absorbed power with the ideal
% energy-optimal absorbed mechanical power.

figure;
P_compare = [P_SD P_ref];
bar(P_compare);

grid on;

set(gca, ...
    'XTick',1:2, ...
    'XTickLabel',{'SD PTO','Optimal reference'});

ylabel('Mean absorbed mechanical power [W]');

title(sprintf( ...
    'Absorbed Mechanical Power Comparison, \\omega = %.1f rad/s', ...
    omega0));

% Add numerical value above each bar
for k = 1:length(P_compare)

    text( ...
        k, ...
        P_compare(k) + 0.03*max(P_compare), ...
        sprintf('%.2f W',P_compare(k)), ...
        'HorizontalAlignment','center', ...
        'FontWeight','bold');

end

ylim([0 1.15*max(P_compare)]);
% ------------------------------------------------------------


% 29. PACKAGE OUTPUT VARIABLES

ref = struct;

ref.omega0 = omega0;
ref.T      = T;
ref.H      = H;
ref.eta    = eta;

ref.Z_rel = Z_rel;
ref.R_rel = R_rel;
ref.X_rel = X_rel;

ref.Z_PTO_opt = Z_PTO_opt;

ref.B_SD = B_SD;

ref.V_SD = V_SD_PTO;
ref.A_SD = A_SD_fit;
ref.phi_SD = phi_SD_PTO;
ref.phi_SD_deg = phi_SD_PTO_deg;

ref.V_ref = V_ref;
ref.A_ref = A_ref;
ref.phi_ref = phi_ref;
ref.phi_ref_deg = phi_ref_deg;

ref.F_eff = F_eff_SD;

ref.P_SD = P_SD;
ref.P_ref = P_ref;

ref.A_SD_norm = A_SD_norm;
ref.A_ref_norm = A_ref_norm;

ref.P_SD_norm = P_SD_norm;
ref.P_ref_norm = P_ref_norm;

ref.phaseDifference_deg = phaseDifference_deg;

ref.t_plot = t_plot;
ref.v_SD_plot = v_SD_plot;
ref.v_ref_plot = v_ref_plot;

ref.Results = Results;

fprintf('\nFinished.\n');

end