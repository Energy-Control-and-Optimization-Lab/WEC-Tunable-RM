%% DDLG_PI_compensated_analysis.m
% Purpose:
%   1) Reconstruct the uncompensated DDLG plant Gu_DDLG(s) = Vrel/F_PI
%      at the nominal design point omega0 = 4 rad/s.
%   2) Design the PI controller independently for the DDLG using the same
%      criterion used for the SD case:
%           |S(j*omega0)| = 0.30
%      i.e. 70% reduction of the passive-to-reference phasor error.
%   3) Fix the PI zero from the plant phase at omega0.
%   4) Compare the independently derived DDLG gains against the completed
%      SD gains.
%   5) Verify the compensated loop using root locus, Bode, Nyquist,
%      sensitivity, complementary sensitivity, margins, and closed-loop
%      poles.
%
% IMPORTANT:
%   - The passive DDLG remains inside Gu_DDLG.
%   - The PI force F_PI is an additional external mechanical force.
%   - The controller is designed at ONE frequency: omega0 = 4 rad/s.
%   - Robustness quantities are used as verification metrics, not as the
%     primary synthesis objective.
%
% Requires: Control System Toolbox

clearvars
clc
close all

fprintf('============================================================\n');
fprintf(' DDLG PI CONTROLLER - COMPENSATED LOOP ANALYSIS\n');
fprintf('============================================================\n');

%% 1) Nominal design point
omega0 = 4;                     % rad/s
S_target = 0.30;                % 70% error reduction target
Bpto_target = 540;              % N*s/m
s = tf('s');

fprintf('omega0       = %.6f rad/s\n',omega0);
fprintf('|S(jw0)| tgt = %.6f\n',S_target);
fprintf('Bpto target  = %.6f N*s/m\n\n',Bpto_target);

%% 2) Frequency-frozen WEC mechanical matrices at omega0 = 4 rad/s
Me = [ 122.8885,  -9.8486;
       -10.4742, 411.7028 ];

Bhv = [ 64.5294,   0.6716;
        -2.4767, 257.5742 ];

Cmat = [3539.2206,   0;
           0,      314.5995];

d = [1; -1];

%% 3) Original passive DDLG parameters
tau_p  = 0.0152;        % m
N      = 560;
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

Kf = (pi/tau_p)*lambda_fd;      % N/A
Ke = Kf;                        % V/(m/s)

Rload = Kf*Ke/Bpto_target - R_wind;
Rtot  = R_wind + Rload;

fprintf('DDLG parameters:\n');
fprintf('  Kf      = %.9f N/A\n',Kf);
fprintf('  Ke      = %.9f V/(m/s)\n',Ke);
fprintf('  Rload   = %.9f ohm\n',Rload);
fprintf('  Rtotal  = %.9f ohm\n',Rtot);
fprintf('  Lwind   = %.9f H\n\n',L_wind);

%% 4) Build Gu_DDLG(s) = Vrel/F_PI
% States:
%   X = [x1 x2 v1 v2 i]^T
%
% Mechanical dynamics:
%   Me*v_dot + Bhv*v + C*x = d*F_PI - d*Kf*i
%
% Electrical dynamics:
%   L*i_dot + Rtot*i = Ke*v_rel

A = [ zeros(2),          eye(2),                 zeros(2,1);
     -(Me\Cmat),        -(Me\Bhv),              -(Me\d)*Kf;
      zeros(1,2),        (Ke/L_wind)*d.',        -Rtot/L_wind ];

B = [ zeros(2,1);
      Me\d;
      0 ];

Cy = [0 0 1 -1 0];
Dy = 0;

sys_DDLG = ss(A,B,Cy,Dy);
Gu_DDLG  = minreal(tf(sys_DDLG),1e-10);

%% 5) Plant response at omega0
G0 = evalfr(Gu_DDLG,1j*omega0);
G0mag = abs(G0);
G0phase = rad2deg(angle(G0));

fprintf('------------------------------------------------------------\n');
fprintf('UNCOMPENSATED DDLG PLANT AT omega0\n');
fprintf('------------------------------------------------------------\n');
fprintf('Gu_DDLG(jw0) = %.10e %+.10ej\n',real(G0),imag(G0));
fprintf('|Gu_DDLG|     = %.10e (m/s)/N\n',G0mag);
fprintf('Magnitude     = %.6f dB\n',20*log10(G0mag));
fprintf('Phase         = %.6f deg\n\n',G0phase);

%% 6) PI zero from phase condition
% C_PI(s) = Kp + Ki/s = Kp*(s + alpha)/s
% alpha = Ki/Kp
%
% Choose the PI phase so that the loop is approximately real positive
% at omega0:
%   angle C_PI(jw0) = -angle Gu_DDLG(jw0)

alpha = omega0*tand(G0phase);       % Ki/Kp [1/s]
Cshape = (s + alpha)/s;

Lshape0 = evalfr(Cshape*Gu_DDLG,1j*omega0);

fprintf('------------------------------------------------------------\n');
fprintf('PI PHASE-SHAPE DESIGN\n');
fprintf('------------------------------------------------------------\n');
fprintf('alpha = Ki/Kp = %.9f 1/s\n',alpha);
fprintf('PI zero       = %.9f rad/s\n',-alpha);
fprintf('angle Cshape(jw0) = %.6f deg\n',rad2deg(angle(evalfr(Cshape,1j*omega0))));
fprintf('angle Lshape(jw0) = %.6f deg\n\n',rad2deg(angle(Lshape0)));

%% 7) Gain level from the 70% error-reduction target
% With angle L(jw0) approximately zero:
%   |S(jw0)| = 1/(1 + |L(jw0)|)
%
% Therefore:
%   rho = |L(jw0)| = 1/S_target - 1

rho = 1/S_target - 1;

Kp_DDLG = rho/abs(Lshape0);
Ki_DDLG = alpha*Kp_DDLG;

fprintf('------------------------------------------------------------\n');
fprintf('INDEPENDENT DDLG PI DESIGN\n');
fprintf('------------------------------------------------------------\n');
fprintf('|S(jw0)| target = %.6f\n',S_target);
fprintf('|L(jw0)| target = %.6f\n',rho);
fprintf('Kp_DDLG = %.9f N*s/m\n',Kp_DDLG);
fprintf('Ki_DDLG = %.9f N/m\n',Ki_DDLG);
fprintf('Ki/Kp   = %.9f 1/s\n\n',Ki_DDLG/Kp_DDLG);

%% 8) Compare against completed SD gains
Kp_SD = 1625.088066;
Ki_SD = 4991.194750;

fprintf('------------------------------------------------------------\n');
fprintf('COMPARISON WITH COMPLETED SD CONTROLLER\n');
fprintf('------------------------------------------------------------\n');
fprintf('SD:   Kp = %.9f, Ki = %.9f\n',Kp_SD,Ki_SD);
fprintf('DDLG: Kp = %.9f, Ki = %.9f\n',Kp_DDLG,Ki_DDLG);
fprintf('Kp difference = %+.6f %%\n',100*(Kp_DDLG-Kp_SD)/Kp_SD);
fprintf('Ki difference = %+.6f %%\n\n',100*(Ki_DDLG-Ki_SD)/Ki_SD);

%% 9) Build compensated loop
Cpi_DDLG = Kp_DDLG + Ki_DDLG/s;

L_DDLG = minreal(Cpi_DDLG*Gu_DDLG,1e-10);
S_DDLG = minreal(feedback(1,L_DDLG),1e-10);
T_DDLG = minreal(feedback(L_DDLG,1),1e-10);

L0 = evalfr(L_DDLG,1j*omega0);
S0 = evalfr(S_DDLG,1j*omega0);
T0 = evalfr(T_DDLG,1j*omega0);

fprintf('------------------------------------------------------------\n');
fprintf('COMPENSATED LOOP AT omega0\n');
fprintf('------------------------------------------------------------\n');
fprintf('L(jw0) = %.9f %+.9fj\n',real(L0),imag(L0));
fprintf('|L|     = %.9f\n',abs(L0));
fprintf('angle L = %.6f deg\n',rad2deg(angle(L0)));
fprintf('|S|     = %.9f\n',abs(S0));
fprintf('|T|     = %.9f\n',abs(T0));
fprintf('angle T = %.6f deg\n\n',rad2deg(angle(T0)));

%% 10) Closed-loop poles
p_cl = pole(T_DDLG);

fprintf('------------------------------------------------------------\n');
fprintf('REDUCED CLOSED-LOOP POLES\n');
fprintf('------------------------------------------------------------\n');
for k = 1:numel(p_cl)
    fprintf('p%d = %+.10f %+.10fj\n',k,real(p_cl(k)),imag(p_cl(k)));
end
fprintf('Reduced closed-loop stable? %d\n\n',isstable(T_DDLG));

%% 11) Full augmented PI state-space eigenvalue check
% Integral state:
%   z_dot = r - Cy*x
%
% Controller:
%   F_PI = Kp*(r-Cy*x) + Ki*z
%
% For eigenvalue analysis, set r = 0.

Aaug = [A - B*Kp_DDLG*Cy, B*Ki_DDLG;
        -Cy,               0];

p_aug = eig(Aaug);

fprintf('------------------------------------------------------------\n');
fprintf('FULL 6-STATE AUGMENTED PI EIGENVALUES\n');
fprintf('------------------------------------------------------------\n');
for k = 1:numel(p_aug)
    fprintf('lambda%d = %+.10f %+.10fj\n',k,real(p_aug(k)),imag(p_aug(k)));
end

%% 12) Classical and all-margin checks
[Gm,Pm,Wcg,Wcp] = margin(L_DDLG);

fprintf('\n------------------------------------------------------------\n');
fprintf('CLASSICAL STABILITY MARGINS\n');
fprintf('------------------------------------------------------------\n');
if isinf(Gm)
    fprintf('Gain margin  = Inf\n');
else
    fprintf('Gain margin  = %.9f (%.6f dB) at %.9f rad/s\n', ...
        Gm,20*log10(Gm),Wcg);
end
fprintf('Phase margin = %.9f deg at %.9f rad/s\n',Pm,Wcp);

try
    AM = allmargin(L_DDLG);
    fprintf('\nallmargin() completed successfully.\n');
    disp(AM)
catch
    fprintf('\nallmargin() not available in this MATLAB release; skipped.\n');
end

%% 13) Sensitivity peaks
w = logspace(-2,5,5000);

respS = squeeze(freqresp(S_DDLG,w));
respT = squeeze(freqresp(T_DDLG,w));

[Ms,idxMs] = max(abs(respS));
[Mt,idxMt] = max(abs(respT));

fprintf('\n------------------------------------------------------------\n');
fprintf('SENSITIVITY METRICS\n');
fprintf('------------------------------------------------------------\n');
fprintf('Ms = max|S| = %.9f (%.6f dB) at %.9f rad/s\n', ...
    Ms,20*log10(Ms),w(idxMs));
fprintf('Mt = max|T| = %.9f (%.6f dB) at %.9f rad/s\n', ...
    Mt,20*log10(Mt),w(idxMt));

%% 14) Build SD compensated loop for direct comparison
Gu_SD = (0.0101855*s^3 + 0.00634375*s^2 + 0.0763278*s) / ...
        (s^4 + 6.65290*s^3 + 33.37965*s^2 + 59.67423*s + 22.05245);

Cpi_SD = Kp_SD + Ki_SD/s;
L_SD = minreal(Cpi_SD*Gu_SD,1e-10);
S_SD = minreal(feedback(1,L_SD),1e-10);
T_SD = minreal(feedback(L_SD,1),1e-10);

%% 15) Figure directory
figDir = fullfile(pwd,'PI_Controller_Analysis_Figures');
if ~exist(figDir,'dir')
    mkdir(figDir);
end

%% 16) Compensated root locus - mechanical-region zoom
Gshape = minreal(Cshape*Gu_DDLG,1e-10);

fig1 = figure('Name','DDLG PI-Compensated Root Locus','Color','w');
rlocus(Gshape)
grid on
hold on
plot(real(p_cl),imag(p_cl),'ks','MarkerSize',8,'LineWidth',1.5)
xline(-alpha,'--','PI zero')
xlim([-8 2])
ylim([-8 8])
xlabel('Real Axis [s^{-1}]')
ylabel('Imaginary Axis [s^{-1}]')
title(sprintf('DDLG PI-Compensated Root Locus, z_{PI} = %.3f rad/s',-alpha))

exportgraphics(fig1,fullfile(figDir,'DDLG_PI_compensated_root_locus.png'), ...
    'Resolution',300);

%% 17) Compensated open-loop Bode
wDesign = logspace(-2,2.5,2500);
respL = squeeze(freqresp(L_DDLG,wDesign));
magL_dB = 20*log10(abs(respL));
phaseL_deg = unwrap(angle(respL))*180/pi;

fig2 = figure('Name','DDLG PI Compensated Bode','Color','w');
tiledlayout(2,1,'TileSpacing','compact','Padding','compact');

nexttile
semilogx(wDesign,magL_dB,'LineWidth',1.5)
grid on
hold on
xline(omega0,'--','\omega_0 = 4 rad/s')
yline(0,':')
plot(omega0,20*log10(abs(L0)),'o','MarkerSize',7,'LineWidth',1.3)
ylabel('Magnitude [dB]')
title(sprintf('DDLG Compensated Open Loop: K_p = %.2f, K_i = %.2f', ...
    Kp_DDLG,Ki_DDLG))

nexttile
semilogx(wDesign,phaseL_deg,'LineWidth',1.5)
grid on
hold on
xline(omega0,'--','\omega_0 = 4 rad/s')
plot(omega0,rad2deg(angle(L0)),'o','MarkerSize',7,'LineWidth',1.3)
ylabel('Phase [deg]')
xlabel('Frequency [rad/s]')

exportgraphics(fig2,fullfile(figDir,'DDLG_PI_compensated_bode.png'), ...
    'Resolution',300);

%% 18) Nyquist
fig3 = figure('Name','DDLG PI Nyquist','Color','w');
nyquist(L_DDLG)
grid on
hold on
plot(-1,0,'rx','MarkerSize',10,'LineWidth',2)
title('Nyquist Plot of DDLG Compensated Loop L(s)')

exportgraphics(fig3,fullfile(figDir,'DDLG_PI_nyquist.png'), ...
    'Resolution',300);

%% 19) Sensitivity and complementary sensitivity
fig4 = figure('Name','DDLG Sensitivity Functions','Color','w');

semilogx(wDesign,20*log10(abs(squeeze(freqresp(S_DDLG,wDesign)))), ...
    'LineWidth',1.5)
hold on
semilogx(wDesign,20*log10(abs(squeeze(freqresp(T_DDLG,wDesign)))), ...
    'LineWidth',1.5)
grid on
xline(omega0,'--','\omega_0 = 4 rad/s')
yline(0,':')
xlabel('Frequency [rad/s]')
ylabel('Magnitude [dB]')
legend('|S(j\omega)|','|T(j\omega)|','Location','best')
title('DDLG Closed-Loop Sensitivity and Complementary Sensitivity')

exportgraphics(fig4,fullfile(figDir,'DDLG_PI_sensitivity_T.png'), ...
    'Resolution',300);

%% 20) DDLG vs SD compensated-loop comparison
respL_DDLG = squeeze(freqresp(L_DDLG,wDesign));
respL_SD   = squeeze(freqresp(L_SD,wDesign));

fig5 = figure('Name','DDLG vs SD Compensated Loops','Color','w');
tiledlayout(2,1,'TileSpacing','compact','Padding','compact');

nexttile
semilogx(wDesign,20*log10(abs(respL_DDLG)),'LineWidth',1.5)
hold on
semilogx(wDesign,20*log10(abs(respL_SD)),'--','LineWidth',1.5)
grid on
xline(omega0,':','\omega_0 = 4 rad/s')
yline(0,':')
ylabel('Magnitude [dB]')
title('Compensated Open-Loop Comparison: DDLG vs SD')
legend('DDLG','SD','Location','best')

nexttile
semilogx(wDesign,unwrap(angle(respL_DDLG))*180/pi,'LineWidth',1.5)
hold on
semilogx(wDesign,unwrap(angle(respL_SD))*180/pi,'--','LineWidth',1.5)
grid on
xline(omega0,':','\omega_0 = 4 rad/s')
ylabel('Phase [deg]')
xlabel('Frequency [rad/s]')
legend('DDLG','SD','Location','best')

exportgraphics(fig5,fullfile(figDir,'DDLG_vs_SD_compensated_loop.png'), ...
    'Resolution',300);

%% 21) Summary
fprintf('\n============================================================\n');
fprintf('DDLG PI COMPENSATED-LOOP ANALYSIS COMPLETE\n');
fprintf('============================================================\n');
fprintf('Independent DDLG gains:\n');
fprintf('  Kp_DDLG = %.9f\n',Kp_DDLG);
fprintf('  Ki_DDLG = %.9f\n',Ki_DDLG);
fprintf('  PI zero = %.9f rad/s\n',-alpha);
fprintf('\nFigures saved in:\n  %s\n',figDir);
fprintf('\nGenerated figures:\n');
fprintf('  1) DDLG_PI_compensated_root_locus.png\n');
fprintf('  2) DDLG_PI_compensated_bode.png\n');
fprintf('  3) DDLG_PI_nyquist.png\n');
fprintf('  4) DDLG_PI_sensitivity_T.png\n');
fprintf('  5) DDLG_vs_SD_compensated_loop.png\n');
fprintf('\nNext step after reviewing these results:\n');
fprintf(['  If stability and sensitivity checks are satisfactory, assign the\n', ...
         '  independently derived Kp_DDLG and Ki_DDLG to the Simulink PI\n', ...
         '  block and perform the first WEC-Sim tracking simulation.\n']);
