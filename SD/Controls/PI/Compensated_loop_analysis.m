%% WEC PI controller - Step 3: compensated-loop analysis and gain screening
% Purpose:
%   1) Use the verified plant G_u(s)
%   2) Fix the PI zero from the phase condition at omega0 = 4 rad/s
%   3) Screen PI gain levels using an explicit sensitivity criterion
%   4) Plot compensated root locus, Bode, and Nyquist diagrams
%   5) Check closed-loop poles and the full augmented PI state-space model
%
% IMPORTANT:
%   The gains generated here are screening candidates, not yet final WEC-Sim
%   gains. Final selection must also satisfy PTO force, stroke, saturation,
%   absorbed-power, and nonlinear WEC-Sim constraints.

clearvars -except output
clc
close all

fprintf('============================================================\n');
fprintf(' WEC PI CONTROLLER - STEP 3: COMPENSATED LOOP ANALYSIS\n');
fprintf('============================================================\n');

%% 1) Verified operating point and plant
omega0 = 4;                  % rad/s
b0     = 540;                % N*s/m, retained passive PTO damping
s      = tf('s');

Gu = (0.0101855*s^3 + 0.00634375*s^2 + 0.0763278*s) / ...
     (s^4 + 6.65290*s^3 + 33.37965*s^2 + 59.67423*s + 22.05245);

% State-space realization from the verified mechanical matrices
Me = [ 122.8885,  -9.8486;
       -10.4742, 411.7028 ];
Bhv = [ 64.5294,   0.6716;
        -2.4767, 257.5742 ];
Cmat = [3539.2206, 0;
        0, 314.5995];
d = [1; -1];
BPTO = b0*(d*d.');
Bp = Bhv + BPTO;

Ap = [zeros(2), eye(2);
      -(Me\Cmat), -(Me\Bp)];
Bu = [zeros(2,1);
      Me\d];
Cy = [0 0 1 -1];

%% 2) Plant frequency response at omega0
G0 = evalfr(Gu,1j*omega0);
G0mag = abs(G0);
G0phase = rad2deg(angle(G0));

fprintf('\nVerified plant response at omega0 = %.4f rad/s:\n',omega0);
fprintf('  Gu(jw0) = %.10e %+.10ej\n',real(G0),imag(G0));
fprintf('  |Gu(jw0)| = %.10e (m/s)/N\n',G0mag);
fprintf('  angle Gu(jw0) = %.6f deg\n',G0phase);

%% 3) Fix the PI zero from the phase condition
% C_PI(s) = Kp + Ki/s = Kp*(s + alpha)/s,
% where alpha = Ki/Kp.
%
% At omega0, choose the PI phase approximately equal to -angle Gu(jw0),
% so that angle L(jw0) = angle[C_PI(jw0)Gu(jw0)] ~ 0 deg.

alpha = omega0*tan(deg2rad(G0phase));   % alpha = Ki/Kp [1/s]
Cshape = (s + alpha)/s;                 % PI shape with Kp factored out
Gshape = minreal(Cshape*Gu,1e-10);      % compensated plant before Kp

fprintf('\nPI phase-shape condition:\n');
fprintf('  alpha = Ki/Kp = %.8f 1/s\n',alpha);
fprintf('  PI zero z_PI = -alpha = %.8f rad/s\n',-alpha);

Cshape0 = evalfr(Cshape,1j*omega0);
Lshape0 = Cshape0*G0;
fprintf('  angle Cshape(jw0) = %.6f deg\n',rad2deg(angle(Cshape0)));
fprintf('  angle [Cshape*Gu](jw0) = %.6f deg\n',rad2deg(angle(Lshape0)));
fprintf('  |Cshape*Gu|(jw0) = %.10e per unit Kp\n',abs(Lshape0));

%% 4) Gain screening from a sensitivity target at omega0
% Closed-loop sensitivity:
%       S = 1/(1+L)
% With the phase-shaped loop approximately real and positive at omega0,
%       |S(jw0)| ~= 1/(1 + |L(jw0)|).
%
% We screen three explicit targets, with 30%% remaining error selected
% as the nominal design target:
%   |S| = 0.30  -> 70%% reduction of passive-to-reference phasor error
%   |S| = 0.20  -> 80%% reduction
%   |S| = 0.10  -> 90%% reduction

S_targets = [0.30 0.20 0.10];
nCand = numel(S_targets);

Kp_vec = zeros(1,nCand);
Ki_vec = zeros(1,nCand);
rho_vec = zeros(1,nCand);

% Validated phasors at omega0 from the previous derivation
VSD  = 0.192033 + 1j*0.316802;       % m/s
VREF = 0.968756 + 1j*0.376907;       % m/s

fprintf('\n------------------------------------------------------------\n');
fprintf('GAIN SCREENING AT omega0 = %.4f rad/s\n',omega0);
fprintf('------------------------------------------------------------\n');
fprintf('Target |S|   |L|=rho      Kp          Ki        |Y| [m/s]   phase Y [deg]   |E| [m/s]   |F_PI| [N]\n');

for k = 1:nCand
    Starget = S_targets(k);
    rho = 1/Starget - 1;
    Kp = rho/abs(Lshape0);
    Ki = alpha*Kp;

    Kp_vec(k) = Kp;
    Ki_vec(k) = Ki;
    rho_vec(k) = rho;

    Cpi = Kp + Ki/s;
    L = minreal(Cpi*Gu,1e-10);

    L0 = evalfr(L,1j*omega0);
    T0 = L0/(1+L0);
    S0 = 1/(1+L0);

    % At omega0 the passive wave-driven output is VSD = Ge*Fe.
    % Therefore the controlled response is:
    %     Y = T*VREF + S*VSD
    Y0 = T0*VREF + S0*VSD;
    E0 = VREF - Y0;
    U0 = evalfr(Cpi,1j*omega0)*E0;

    fprintf('   %.3f      %7.3f   %10.3f  %10.3f    %8.5f       %8.3f      %8.5f    %8.2f\n', ...
        Starget,rho,Kp,Ki,abs(Y0),rad2deg(angle(Y0)),abs(E0),abs(U0));
end

%% 5) Select a nominal screening candidate
% Use |S(jw0)| = 0.30 as the nominal candidate for detailed analysis.
% This means 30%% of the original passive-to-reference phasor error remains,
% i.e. a 70%% reduction in that tracking-error phasor at omega0.
% This is NOT yet the final controller; WEC-Sim force, stroke, saturation,
% absorbed-power, and nonlinear behavior still determine final acceptance.

idxNom = 1;
Kp_nom = Kp_vec(idxNom);
Ki_nom = Ki_vec(idxNom);
Cpi_nom = Kp_nom + Ki_nom/s;
L_nom = minreal(Cpi_nom*Gu,1e-10);
T_nom = minreal(feedback(L_nom,1),1e-10);
S_nom = minreal(feedback(1,L_nom),1e-10);

fprintf('\nNominal screening candidate (|S(jw0)| target = %.2f):\n',S_targets(idxNom));
fprintf('  Kp = %.6f N/(m/s)\n',Kp_nom);
fprintf('  Ki = %.6f N/m\n',Ki_nom);
fprintf('  Ki/Kp = %.8f 1/s\n',Ki_nom/Kp_nom);

L0_nom = evalfr(L_nom,1j*omega0);
T0_nom = evalfr(T_nom,1j*omega0);
S0_nom = evalfr(S_nom,1j*omega0);

fprintf('  L(jw0) = %.6f %+.6fj\n',real(L0_nom),imag(L0_nom));
fprintf('  |L(jw0)| = %.6f, angle L(jw0) = %.6f deg\n',abs(L0_nom),rad2deg(angle(L0_nom)));
fprintf('  |T(jw0)| = %.6f, angle T(jw0) = %.6f deg\n',abs(T0_nom),rad2deg(angle(T0_nom)));
fprintf('  |S(jw0)| = %.6f\n',abs(S0_nom));

%% 6) Closed-loop poles and classical stability margins
p_cl = pole(T_nom);

fprintf('\nClosed-loop poles of the reduced input-output model:\n');
for k = 1:numel(p_cl)
    fprintf('  p%d = %+.8f %+.8fj\n',k,real(p_cl(k)),imag(p_cl(k)));
end

[Gm,Pm,Wcg,Wcp] = margin(L_nom);

fprintf('\nClassical margins for nominal compensated loop:\n');
if isinf(Gm)
    fprintf('  Gain margin  = Inf\n');
else
    fprintf('  Gain margin  = %.6f (%.3f dB) at %.6f rad/s\n',Gm,20*log10(Gm),Wcg);
end
fprintf('  Phase margin = %.6f deg at %.6f rad/s\n',Pm,Wcp);

%% 7) Full augmented PI state-space eigenvalue check
% Integral state z_dot = r - Cy*x
% u = Kp(r-Cy*x) + Ki*z
Aaug = [Ap - Bu*Kp_nom*Cy, Bu*Ki_nom;
        -Cy,               0];

p_aug = eig(Aaug);
fprintf('\nFull 5-state augmented PI eigenvalues:\n');
for k = 1:numel(p_aug)
    fprintf('  lambda%d = %+.10f %+.10fj\n',k,real(p_aug(k)),imag(p_aug(k)));
end

fprintf(['\nIMPORTANT: because Gu(s) has an exact zero at s=0 and the PI has ', ...
         'an integrator pole at s=0, the input-output loop contains an ', ...
         'origin pole-zero cancellation. The reduced transfer function can ', ...
         'be stable while the full augmented realization retains a neutral ', ...
         'mode near zero. This must be handled carefully in Simulink and ', ...
         'with saturation/anti-windup.\n']);

%% 8) Figure directory
figDir = fullfile(pwd,'PI_Controller_Analysis_Figures');
if ~exist(figDir,'dir')
    mkdir(figDir);
end

%% 9) Figure 4 - compensated root locus with PI zero fixed
f1 = figure('Name','PI-compensated root locus','Color','w');
rlocus(Gshape)
grid on
hold on

% Mark nominal reduced closed-loop poles
plot(real(p_cl),imag(p_cl),'ks','MarkerSize',8,'LineWidth',1.5)

xline(-alpha,'--','PI zero')
xlim([-22 2])
ylim([-5 5])
title(sprintf('PI-Compensated Root Locus, z_{PI} = %.3f rad/s',-alpha))
xlabel('Real Axis (s^{-1})')
ylabel('Imaginary Axis (s^{-1})')

set(f1,'PaperPositionMode','auto')
print(f1,fullfile(figDir,'PI_compensated_root_locus.png'),'-dpng','-r300')
print(f1,fullfile(figDir,'PI_compensated_root_locus.pdf'),'-dpdf','-painters')

%% 10) Figure 5 - nominal compensated-loop Bode plot
w = logspace(-2,2.5,1800);
respL = squeeze(freqresp(L_nom,w));
magL_dB = 20*log10(abs(respL));
phaseL_deg = unwrap(angle(respL))*180/pi;

f2 = figure('Name','Nominal compensated loop Bode','Color','w');
tiledlayout(2,1)

nexttile
semilogx(w,magL_dB,'LineWidth',1.5)
grid on
hold on
xline(omega0,'--',sprintf('\\omega_0 = %.1f rad/s',omega0))
yline(0,':')
plot(omega0,20*log10(abs(L0_nom)),'ko','MarkerFaceColor','k')
ylabel('Magnitude (dB)')
title(sprintf('Compensated Open-Loop Bode Plot: K_p = %.1f, K_i = %.1f',Kp_nom,Ki_nom))

nexttile
semilogx(w,phaseL_deg,'LineWidth',1.5)
grid on
hold on
xline(omega0,'--',sprintf('\\omega_0 = %.1f rad/s',omega0))
plot(omega0,rad2deg(angle(L0_nom)),'ko','MarkerFaceColor','k')
ylabel('Phase (deg)')
xlabel('Frequency (rad/s)')

set(f2,'PaperPositionMode','auto')
print(f2,fullfile(figDir,'PI_compensated_bode_nominal.png'),'-dpng','-r300')
print(f2,fullfile(figDir,'PI_compensated_bode_nominal.pdf'),'-dpdf','-painters')

%% 11) Figure 6 - Nyquist plot of nominal compensated loop
f3 = figure('Name','Nominal compensated loop Nyquist','Color','w');
nyquist(L_nom)
grid on
hold on
plot(-1,0,'rx','MarkerSize',10,'LineWidth',2)
title(sprintf('Nyquist Plot of L(s), K_p = %.1f, K_i = %.1f',Kp_nom,Ki_nom))

set(f3,'PaperPositionMode','auto')
print(f3,fullfile(figDir,'PI_compensated_nyquist_nominal.png'),'-dpng','-r300')
print(f3,fullfile(figDir,'PI_compensated_nyquist_nominal.pdf'),'-dpdf','-painters')

%% 12) Figure 7 - sensitivity and complementary sensitivity
respS = squeeze(freqresp(S_nom,w));
respT = squeeze(freqresp(T_nom,w));

f4 = figure('Name','Sensitivity functions','Color','w');
semilogx(w,20*log10(abs(respS)),'LineWidth',1.5)
hold on
semilogx(w,20*log10(abs(respT)),'LineWidth',1.5)
grid on
xline(omega0,'--',sprintf('\\omega_0 = %.1f rad/s',omega0))
yline(0,':')
xlabel('Frequency (rad/s)')
ylabel('Magnitude (dB)')
legend('|S(j\\omega)|','|T(j\\omega)|','Location','best')
title('Closed-Loop Sensitivity and Complementary Sensitivity')

set(f4,'PaperPositionMode','auto')
print(f4,fullfile(figDir,'PI_sensitivity_complementary_sensitivity.png'),'-dpng','-r300')
print(f4,fullfile(figDir,'PI_sensitivity_complementary_sensitivity.pdf'),'-dpdf','-painters')

%% 13) Summary
fprintf('\n============================================================\n');
fprintf('STEP 3 COMPENSATED-LOOP ANALYSIS COMPLETE\n');
fprintf('============================================================\n');
fprintf('Figures saved in:\n  %s\n',figDir);
fprintf('\nGenerated figures:\n');
fprintf('  4) PI_compensated_root_locus\n');
fprintf('  5) PI_compensated_bode_nominal\n');
fprintf('  6) PI_compensated_nyquist_nominal\n');
fprintf('  7) PI_sensitivity_complementary_sensitivity\n');
fprintf(['\nNext step after reviewing these results: select the controller gain ', ...
         'level using tracking error, stability margins, predicted F_PI, ', ...
         'and then implement the chosen PI in WEC-Sim/Simulink.\n']);
