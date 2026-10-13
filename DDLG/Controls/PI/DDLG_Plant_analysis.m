%% DDLG_Plant_analysis.m
% Purpose:
%   1) Build the uncompensated DDLG outer-control plant at omega0 = 4 rad/s.
%   2) Retain the passive DDLG inside the plant.
%   3) Keep the DDLG electrical inductance as one additional state.
%   4) Compare the resulting plant directly with the previously validated
%      SD plant used for the PI-controller design.
%   5) Print poles, zeros, controllability, and Gu_DDLG(j*omega0).
%   6) Generate PNG figures for design documentation.
%
% Control input:
%       F_PI  [N]
%
% Output:
%       v_rel [m/s]
%
% State vector:
%       X = [x1; x2; v1; v2; i]
%
% IMPORTANT:
%   - This is a frequency-frozen design model at omega0 = 4 rad/s,
%     exactly analogous to the SD PI plant-analysis approach.
%   - Hydrodynamic added mass/radiation damping are frozen at omega0.
%   - The original passive DDLG remains in the plant through its RL
%     electrical dynamics and electromagnetic force.
%   - No active/current controller is included inside the DDLG.
%   - Wave excitation is a disturbance and is omitted from Gu_DDLG.
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
Bpto_target = 540;          % N*s/m, common passive baseline

d = [1; -1];               % v_rel = d^T v

fprintf('============================================================\n');
fprintf(' DDLG PI CONTROLLER - UNCOMPENSATED PLANT ANALYSIS\n');
fprintf('============================================================\n');
fprintf('omega0      = %.6f rad/s\n', omega0);
fprintf('T0          = %.6f s\n', T0);
fprintf('H           = %.6f m\n', H);
fprintf('eta         = %.6f m\n', eta);
fprintf('Bpto target = %.6f N*s/m\n\n', Bpto_target);

%% 2) Frequency-frozen WEC mechanical matrices at omega0 = 4 rad/s
% These are the same validated matrices used in the SD PI design.
% Bhv contains hydrodynamic radiation damping at omega0 plus the physical
% linear viscous damping. It does NOT contain PTO damping.

Me = [ 122.8885,  -9.8486;
       -10.4742, 411.7028 ];                     % kg

Bhv = [ 64.5294,   0.6716;
        -2.4767, 257.5742 ];                     % N*s/m

C = [3539.2206,   0;
        0,      314.5995];                       % N/m

%% 3) Passive DDLG parameters
% Same parameterization as the original uncompensated DDLG input file.

tau_p  = 0.0152;        % m
N      = 560;           % turns
Br     = 1.48;          % T
Dcoil  = 5.74/100;      % m
Dmag   = 50.8/1000;     % m
R_wind = 1.1;           % ohm
L_wind = 0.013;         % H

R_int  = Dcoil/2;
R_iman = Dmag/2;
g_gap  = R_int - R_iman;

k_g       = R_iman/(R_iman + g_gap);
A_pole    = pi*R_iman^2;
k_fringe  = 0.75;
Phi_max   = k_fringe*k_g*Br*A_pole;
lambda_fd = N*Phi_max;

% In SI units for this linear-generator model, the force constant and
% back-EMF constant are numerically equal.
Kf = (pi/tau_p)*lambda_fd;     % N/A
Ke = Kf;                       % V/(m/s)

% Choose passive load so that the quasi-static equivalent damping is 540.
Rload = Kf*Ke/Bpto_target - R_wind;
Rtot  = R_wind + Rload;

Bpto_calc = Kf*Ke/Rtot;
omega_e   = Rtot/L_wind;
tau_e     = L_wind/Rtot;

fprintf('Passive DDLG parameters:\n');
fprintf('  Kf          = %.9f N/A\n', Kf);
fprintf('  Ke          = %.9f V/(m/s)\n', Ke);
fprintf('  R_wind      = %.9f ohm\n', R_wind);
fprintf('  R_load      = %.9f ohm\n', Rload);
fprintf('  R_total     = %.9f ohm\n', Rtot);
fprintf('  L_wind      = %.9f H\n', L_wind);
fprintf('  Bpto_calc   = %.9f N*s/m\n', Bpto_calc);
fprintf('  tau_e       = %.9e s\n', tau_e);
fprintf('  omega_e     = %.6f rad/s\n\n', omega_e);

%% 4) Build the DDLG-augmented state-space plant
%
% Mechanical equation:
%
%   Me*v_dot + Bhv*v + C*x = d*F_PI - d*Kf*i
%
% Passive DDLG electrical equation:
%
%   L*i_dot + Rtot*i = Ke*v_rel
%   v_rel = d^T*v
%
% Therefore:
%
%   i_dot = (Ke/L)*d^T*v - (Rtot/L)*i
%
% The control input F_PI is an ADDITIONAL external mechanical force.

A = [ zeros(2),          eye(2),                 zeros(2,1);
     -(Me\C),           -(Me\Bhv),              -(Me\d)*Kf;
      zeros(1,2),        (Ke/L_wind)*d.',        -Rtot/L_wind ];

B = [ zeros(2,1);
      Me\d;
      0 ];

Cy = [0 0 1 -1 0];
Dy = 0;

sys_DDLG = ss(A,B,Cy,Dy);
sys_DDLG.InputName  = 'F_PI';
sys_DDLG.OutputName = 'v_rel';

Gu_DDLG_ss = minreal(sys_DDLG,1e-10);
Gu_DDLG    = minreal(tf(Gu_DDLG_ss),1e-10);

fprintf('Uncompensated DDLG plant:\n');
fprintf('  Gu_DDLG(s) = V_rel(s) / F_PI(s)\n');
fprintf('  Full state order = %d\n', order(sys_DDLG));
fprintf('  Minimal order    = %d\n\n', order(Gu_DDLG_ss));

Gu_DDLG

%% 5) Rebuild the validated SD plant for direct comparison
% This is the exact analytical form used in the completed SD controller
% analysis, with passive Bpto = 540 N*s/m already retained in the plant.

s = tf('s');
Gu_SD = (0.0101855*s^3 + 0.00634375*s^2 + 0.0763278*s) / ...
        (s^4 + 6.65290*s^3 + 33.37965*s^2 + 59.67423*s + 22.05245);

%% 6) Poles, zeros, stability, controllability
p_DDLG = pole(Gu_DDLG);
z_DDLG = zero(Gu_DDLG);

fprintf('\n------------------------------------------------------------\n');
fprintf('DDLG PLANT POLES\n');
fprintf('------------------------------------------------------------\n');
for k = 1:numel(p_DDLG)
    fprintf('p%d = %+.10f %+.10fj\n', k, real(p_DDLG(k)), imag(p_DDLG(k)));
end

fprintf('\n------------------------------------------------------------\n');
fprintf('DDLG PLANT ZEROS\n');
fprintf('------------------------------------------------------------\n');
for k = 1:numel(z_DDLG)
    fprintf('z%d = %+.10f %+.10fj\n', k, real(z_DDLG(k)), imag(z_DDLG(k)));
end

fprintf('\nOpen-loop DDLG plant stable? %d  (1 = yes, 0 = no)\n', ...
        isstable(Gu_DDLG));

Co = ctrb(A,B);
rankCo = rank(Co);
fprintf('Controllability rank = %d out of %d states\n', rankCo, size(A,1));

%% 7) Evaluate both plants at the design frequency
G0_DDLG = evalfr(Gu_DDLG,1j*omega0);
G0_SD   = evalfr(Gu_SD,1j*omega0);

mag_DDLG = abs(G0_DDLG);
dB_DDLG  = 20*log10(mag_DDLG);
ph_DDLG  = angle(G0_DDLG)*180/pi;

mag_SD = abs(G0_SD);
dB_SD  = 20*log10(mag_SD);
ph_SD  = angle(G0_SD)*180/pi;

mag_diff_pct = 100*(mag_DDLG-mag_SD)/mag_SD;
phase_diff_deg = ph_DDLG-ph_SD;

fprintf('\n------------------------------------------------------------\n');
fprintf('RESPONSE AT DESIGN FREQUENCY omega0 = %.6f rad/s\n', omega0);
fprintf('------------------------------------------------------------\n');
fprintf('DDLG:\n');
fprintf('  Gu_DDLG(jw0) = %.10e %+.10ej\n', real(G0_DDLG), imag(G0_DDLG));
fprintf('  |Gu_DDLG|     = %.10e (m/s)/N\n', mag_DDLG);
fprintf('  magnitude     = %.6f dB\n', dB_DDLG);
fprintf('  phase         = %.6f deg\n\n', ph_DDLG);

fprintf('SD:\n');
fprintf('  |Gu_SD|       = %.10e (m/s)/N\n', mag_SD);
fprintf('  magnitude     = %.6f dB\n', dB_SD);
fprintf('  phase         = %.6f deg\n\n', ph_SD);

fprintf('DDLG - SD comparison at omega0:\n');
fprintf('  magnitude difference = %+.6f %%\n', mag_diff_pct);
fprintf('  phase difference     = %+.6f deg\n', phase_diff_deg);

%% 8) Inspect the passive DDLG damping at omega0
B_DDLG_0 = Kf*Ke/(Rtot + 1j*omega0*L_wind);

fprintf('\nPassive DDLG equivalent damping at omega0:\n');
fprintf('  B_DDLG(jw0) = %.10e %+.10ej N*s/m\n', ...
        real(B_DDLG_0), imag(B_DDLG_0));
fprintf('  |B_DDLG|     = %.9f N*s/m\n', abs(B_DDLG_0));
fprintf('  phase        = %.9f deg\n', angle(B_DDLG_0)*180/pi);

%% 9) Output folder
figDir = fullfile(pwd,'PI_Controller_Analysis_Figures');
if ~exist(figDir,'dir')
    mkdir(figDir);
end

%% 10) Figure 1 - DDLG pole-zero map
fig1 = figure('Name','DDLG Plant Pole-Zero Map','Color','w');
pzmap(Gu_DDLG)
grid on
xlabel('Real Axis [s^{-1}]')
ylabel('Imaginary Axis [s^{-1}]')
title('Pole-Zero Map of DDLG Control Plant G_u^{DDLG}(s)')

exportgraphics(fig1,fullfile(figDir,'DDLG_Gu_pole_zero_map.png'), ...
               'Resolution',300);

%% 11) Figure 2 - DDLG Bode plot with omega0 marked
wplot = logspace(-2,5,3500);
Gj = squeeze(freqresp(Gu_DDLG,wplot));
mag_dB = 20*log10(abs(Gj));
phase_deg = unwrap(angle(Gj))*180/pi;

fig2 = figure('Name','DDLG Plant Bode Plot','Color','w');
tiledlayout(2,1,'TileSpacing','compact','Padding','compact');

nexttile
semilogx(wplot,mag_dB,'LineWidth',1.5)
grid on
hold on
xline(omega0,'--','\omega_0 = 4 rad/s', ...
    'LabelVerticalAlignment','bottom');
plot(omega0,dB_DDLG,'o','MarkerSize',7,'LineWidth',1.3)
ylabel('Magnitude [dB]')
title('Bode Response of DDLG Control Plant G_u^{DDLG}(s)')

nexttile
semilogx(wplot,phase_deg,'LineWidth',1.5)
grid on
hold on
xline(omega0,'--','\omega_0 = 4 rad/s', ...
    'LabelVerticalAlignment','bottom');
plot(omega0,ph_DDLG,'o','MarkerSize',7,'LineWidth',1.3)
ylabel('Phase [deg]')
xlabel('Frequency [rad/s]')

exportgraphics(fig2,fullfile(figDir,'DDLG_Gu_bode_omega4.png'), ...
               'Resolution',300);

%% 12) Figure 3 - Uncompensated DDLG root locus
fig3 = figure('Name','Uncompensated DDLG Root Locus','Color','w');
rlocus(Gu_DDLG)
grid on
xlabel('Real Axis [s^{-1}]')
ylabel('Imaginary Axis [s^{-1}]')
title('Root Locus of Uncompensated DDLG Plant K G_u^{DDLG}(s)')

exportgraphics(fig3,fullfile(figDir,'DDLG_Gu_root_locus_uncompensated.png'), ...
               'Resolution',300);

%% 13) Figure 4 - Direct DDLG versus SD comparison
wcmp = logspace(-2,5,3500);
Gd = squeeze(freqresp(Gu_DDLG,wcmp));
Gs = squeeze(freqresp(Gu_SD,wcmp));

fig4 = figure('Name','DDLG vs SD Uncompensated Plants','Color','w');
tiledlayout(2,1,'TileSpacing','compact','Padding','compact');

nexttile
semilogx(wcmp,20*log10(abs(Gd)),'LineWidth',1.5)
hold on
semilogx(wcmp,20*log10(abs(Gs)),'--','LineWidth',1.5)
xline(omega0,':','\omega_0 = 4 rad/s');
grid on
ylabel('Magnitude [dB]')
title('Uncompensated Plant Comparison: DDLG vs SD')
legend('DDLG','SD','Location','best')

nexttile
semilogx(wcmp,unwrap(angle(Gd))*180/pi,'LineWidth',1.5)
hold on
semilogx(wcmp,unwrap(angle(Gs))*180/pi,'--','LineWidth',1.5)
xline(omega0,':','\omega_0 = 4 rad/s');
grid on
ylabel('Phase [deg]')
xlabel('Frequency [rad/s]')
legend('DDLG','SD','Location','best')

exportgraphics(fig4,fullfile(figDir,'DDLG_vs_SD_uncompensated_bode.png'), ...
               'Resolution',300);

%% 14) Summary
fprintf('\n============================================================\n');
fprintf('DDLG UNCOMPENSATED PLANT ANALYSIS COMPLETE\n');
fprintf('============================================================\n');
fprintf('Figures saved in:\n  %s\n',figDir);
fprintf('\nGenerated figures:\n');
fprintf('  1) DDLG_Gu_pole_zero_map.png\n');
fprintf('  2) DDLG_Gu_bode_omega4.png\n');
fprintf('  3) DDLG_Gu_root_locus_uncompensated.png\n');
fprintf('  4) DDLG_vs_SD_uncompensated_bode.png\n');
fprintf('\nNext step:\n');
fprintf(['  Compare Gu_DDLG(j4) with Gu_SD(j4), then test the existing SD PI\n', ...
         '  gains on the DDLG plant before deriving any new gains.\n']);
