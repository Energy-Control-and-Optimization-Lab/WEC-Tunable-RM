%% WEC PI controller - Step 2: plant frequency-domain analysis
% Purpose:
%   1) Reconstruct the verified two-body WEC control plant from the
%      frequency-frozen mechanical matrices at omega0 = 4 rad/s.
%   2) Reproduce the documented transfer function Gu(s) = Vrel/FPI.
%   3) Print poles, zeros, controllability, and Gu(j*omega0).
%   4) Generate publication-ready pole-zero, Bode, and root-locus figures.
%
% IMPORTANT:
%   - The passive PTO damping b0 = 540 N*s/m remains inside the plant.
%   - The figures in this script analyze the PLANT before final PI gains
%     are selected.
%   - Poles and zeros are properties of Gu(s); they are not "at" one
%     frequency. Gu(j4) is the plant evaluated at the design frequency.
%
% Requires: Control System Toolbox

clearvars
clc
close all

%% 1) Design operating point
omega0 = 4;                 % rad/s
T0     = 2*pi/omega0;       % s
H      = 0.32;              % m
eta    = H/2;               % m
b0     = 540;               % N*s/m

d = [1; -1];               % v_rel = d^T qdot

fprintf('============================================================\n');
fprintf(' WEC PI CONTROLLER - STEP 2: PLANT ANALYSIS\n');
fprintf('============================================================\n');
fprintf('omega0 = %.6f rad/s\n', omega0);
fprintf('T0     = %.6f s\n', T0);
fprintf('H      = %.6f m\n', H);
fprintf('eta    = %.6f m\n', eta);
fprintf('b0     = %.6f N*s/m\n\n', b0);

%% 2) Frequency-frozen mechanical matrices at omega0 = 4 rad/s
Me = [ 122.8885,  -9.8486;
       -10.4742, 411.7028 ];                     % kg

Bhv = [ 64.5294,   0.6716;
        -2.4767, 257.5742 ];                     % N*s/m

C = [3539.2206,   0;
        0,      314.5995];                       % N/m

% Retained passive PTO damping
BPTO = b0*(d*d.');
Bp   = Bhv + BPTO;

%% 3) Reconstruct the state-space plant
% x = [x1; x2; v1; v2]
% xdot = Ap*x + Bu*F_PI + Be*Fe
% y = Cy*x = v_rel

Ap = [zeros(2), eye(2);
      -(Me\C), -(Me\Bp)];

Bu = [zeros(2,1);
      Me\d];

Be = [zeros(2,2);
      Me\eye(2)];

Cy = [0 0 1 -1];
Dy = 0;

sys_u = ss(Ap, Bu, Cy, Dy);
Gu_ss = minreal(tf(sys_u), 1e-10);

%% 4) Documented analytical transfer function
% This is the rounded analytical form used in the design paper.
s = tf('s');
Gu_doc = (0.0101855*s^3 + 0.00634375*s^2 + 0.0763278*s) / ...
         (s^4 + 6.65290*s^3 + 33.37965*s^2 + 59.67423*s + 22.05245);

fprintf('State-space-derived Gu(s):\n');
Gu_ss

fprintf('Documented analytical Gu(s):\n');
Gu_doc

% Use the documented analytical form for the plots below so the exact
% zero at s = 0 is preserved rather than represented by floating-point
% roundoff from ss -> tf conversion.
Gu = Gu_doc;

%% 5) Verify state-space and analytical transfer functions agree
[num_ss, den_ss] = tfdata(Gu_ss, 'v');
[num_doc, den_doc] = tfdata(Gu_doc, 'v');

% Normalize to monic denominator
num_ss  = num_ss/den_ss(1);
den_ss  = den_ss/den_ss(1);
num_doc = num_doc/den_doc(1);
den_doc = den_doc/den_doc(1);

% Trim tiny leading numerator coefficients if present
while numel(num_ss) > numel(num_doc) && abs(num_ss(1)) < 1e-10
    num_ss(1) = [];
end

% Pad if MATLAB represents an exact zero constant term differently
if numel(num_ss) < numel(num_doc)
    num_ss = [num_ss zeros(1, numel(num_doc)-numel(num_ss))];
elseif numel(num_doc) < numel(num_ss)
    num_doc = [num_doc zeros(1, numel(num_ss)-numel(num_doc))];
end

num_err = max(abs(num_ss-num_doc));
den_err = max(abs(den_ss-den_doc));

fprintf('\nTransfer-function comparison:\n');
fprintf('  max numerator coefficient error   = %.6e\n', num_err);
fprintf('  max denominator coefficient error = %.6e\n', den_err);

%% 6) Poles, zeros, stability, controllability
p = pole(Gu);
z = zero(Gu);

fprintf('\n------------------------------------------------------------\n');
fprintf('PLANT POLES\n');
fprintf('------------------------------------------------------------\n');
for k = 1:numel(p)
    fprintf('p%d = %+.8f %+.8fj\n', k, real(p(k)), imag(p(k)));
end

fprintf('\n------------------------------------------------------------\n');
fprintf('PLANT ZEROS\n');
fprintf('------------------------------------------------------------\n');
for k = 1:numel(z)
    fprintf('z%d = %+.8f %+.8fj\n', k, real(z(k)), imag(z(k)));
end

fprintf('\nOpen-loop plant stable? %d  (1 = yes, 0 = no)\n', isstable(Gu));

Co = ctrb(Ap, Bu);
rankCo = rank(Co);
fprintf('Controllability rank = %d out of %d states\n', rankCo, size(Ap,1));

%% 7) Evaluate Gu(j*omega0)
G0 = evalfr(Gu, 1j*omega0);
G0_mag = abs(G0);
G0_dB = 20*log10(G0_mag);
G0_phase_deg = angle(G0)*180/pi;

fprintf('\n------------------------------------------------------------\n');
fprintf('PLANT RESPONSE AT DESIGN FREQUENCY omega0 = %.6f rad/s\n', omega0);
fprintf('------------------------------------------------------------\n');
fprintf('Gu(j*omega0) = %.10e %+.10ej\n', real(G0), imag(G0));
fprintf('|Gu(j*omega0)| = %.10e (m/s)/N\n', G0_mag);
fprintf('Magnitude       = %.6f dB\n', G0_dB);
fprintf('Phase           = %.6f deg\n', G0_phase_deg);

%% 8) Compare plant phase to passive-to-optimal phase shift
% Values already derived/validated in the paper.
phi_SD  = 58.777;            % deg
phi_ref = 21.259;            % deg
Delta_phi = phi_ref - phi_SD;

fprintf('\nPassive velocity phase = %.6f deg\n', phi_SD);
fprintf('Reference phase        = %.6f deg\n', phi_ref);
fprintf('Required phase change  = %.6f deg\n', Delta_phi);
fprintf('Plant phase at omega0  = %.6f deg\n', G0_phase_deg);
fprintf('Plant phase + required phase change = %.6f deg\n', ...
        G0_phase_deg + Delta_phi);

% Preliminary PI phase-shape relation, NOT final gains.
% For C_PI(jw) = Kp - j Ki/w,
% angle(C_PI(jw0)) = -atan(Ki/(w0*Kp)).
% Setting angle(C_PI(jw0)) = -angle(Gu(jw0)) gives:
alpha = omega0*tand(G0_phase_deg);   % alpha = Ki/Kp [1/s]

fprintf('\nPreliminary PI phase-shape condition:\n');
fprintf('  Ki/Kp = omega0*tan(angle Gu(jw0))\n');
fprintf('  Ki/Kp = %.8f 1/s\n', alpha);
fprintf('  Corresponding PI zero location = -Ki/Kp = %.8f rad/s\n', -alpha);
fprintf('  NOTE: this fixes only the PI zero/phase shape, not the final gain level.\n');

%% 9) Output folder for paper figures
figDir = fullfile(pwd, 'PI_Controller_Analysis_Figures');
if ~exist(figDir, 'dir')
    mkdir(figDir);
end

%% 10) Figure 1 - Plant pole-zero map
fig1 = figure('Name','WEC Plant Pole-Zero Map','Color','w');
pzmap(Gu)
grid on
xlabel('Real Axis [s^{-1}]')
ylabel('Imaginary Axis [s^{-1}]')
title('Pole-Zero Map of WEC Control Plant G_u(s)')

exportgraphics(fig1, fullfile(figDir,'Gu_pole_zero_map.png'), 'Resolution', 300);
exportgraphics(fig1, fullfile(figDir,'Gu_pole_zero_map.pdf'), 'ContentType','vector');

%% 11) Figure 2 - Bode plot with omega0 = 4 rad/s marked
w = logspace(-2, 2, 2500);              % rad/s
Gj = squeeze(freqresp(Gu, w));
mag_dB = 20*log10(abs(Gj));
phase_deg = unwrap(angle(Gj))*180/pi;

fig2 = figure('Name','WEC Plant Bode Plot','Color','w');
t = tiledlayout(2,1,'TileSpacing','compact','Padding','compact');

ax1 = nexttile;
semilogx(w, mag_dB, 'LineWidth', 1.5)
grid on
hold on
xline(omega0, '--', '\omega_0 = 4 rad/s', ...
    'LabelVerticalAlignment','bottom');
plot(omega0, G0_dB, 'o', 'MarkerSize', 7, 'LineWidth', 1.3)
ylabel('Magnitude [dB]')
title('Bode Response of WEC Control Plant G_u(s)')

ax2 = nexttile;
semilogx(w, phase_deg, 'LineWidth', 1.5)
grid on
hold on
xline(omega0, '--', '\omega_0 = 4 rad/s', ...
    'LabelVerticalAlignment','bottom');
plot(omega0, G0_phase_deg, 'o', 'MarkerSize', 7, 'LineWidth', 1.3)
ylabel('Phase [deg]')
xlabel('Frequency [rad/s]')

linkaxes([ax1 ax2], 'x')

exportgraphics(fig2, fullfile(figDir,'Gu_bode_omega4.png'), 'Resolution', 300);
exportgraphics(fig2, fullfile(figDir,'Gu_bode_omega4.pdf'), 'ContentType','vector');

%% 12) Figure 3 - Uncompensated plant root locus
% This is the root locus of K*Gu(s), i.e. proportional gain only.
% It is useful for understanding the plant before the PI zero is added.
fig3 = figure('Name','Uncompensated WEC Plant Root Locus','Color','w');
rlocus(Gu)
grid on
xlabel('Real Axis [s^{-1}]')
ylabel('Imaginary Axis [s^{-1}]')
title('Root Locus of Uncompensated WEC Plant K G_u(s)')

exportgraphics(fig3, fullfile(figDir,'Gu_root_locus_uncompensated.png'), 'Resolution', 300);
exportgraphics(fig3, fullfile(figDir,'Gu_root_locus_uncompensated.pdf'), 'ContentType','vector');

%% 13) Summary
fprintf('\n============================================================\n');
fprintf('STEP 2 PLANT ANALYSIS COMPLETE\n');
fprintf('============================================================\n');
fprintf('Figures saved in:\n  %s\n', figDir);
fprintf('\nGenerated figures:\n');
fprintf('  1) Gu_pole_zero_map\n');
fprintf('  2) Gu_bode_omega4\n');
fprintf('  3) Gu_root_locus_uncompensated\n');
fprintf('\nNext design step:\n');
fprintf(['  Use the omega0 phase condition to define the PI zero, then analyze\n', ...
         '  the compensated loop L(s)=C_PI(s)G_u(s) with the full augmented\n', ...
         '  state-space model before selecting the final Kp and Ki.\n']);
