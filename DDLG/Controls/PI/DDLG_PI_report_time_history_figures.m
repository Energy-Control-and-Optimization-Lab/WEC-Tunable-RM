%% DDLG_PI_report_time_history_figures.m
% Publication-ready time-history figures for the DDLG PI controller.
%
% Required files in the current folder:
%   DDLG_passive_baseline.mat
%   DDLG_PI_controlled_final.mat
%
% The script:
%   1) Loads the passive and controlled WEC-Sim runs
%   2) Fits v_ref, v_rel(passive), and v_rel(PI) at omega0 = 4 rad/s
%   3) Computes the actual passive-to-controlled phasor-error reduction
%   4) Plots the final 5 wave cycles:
%        - v_ref vs passive v_rel vs PI-controlled v_rel
%        - F_PI, F_DDLG, and total PTO force for the controlled run
%   5) Exports PNG + PDF versions for the report

clear
clc
close all

%% 1) Load saved WEC-Sim runs
passiveFile    = 'DDLG_passive_baseline.mat';
controlledFile = 'DDLG_PI_controlled_final.mat';

assert(isfile(passiveFile), ...
    'Cannot find %s in the current folder.', passiveFile);
assert(isfile(controlledFile), ...
    'Cannot find %s in the current folder.', controlledFile);

P = load(passiveFile);
C = load(controlledFile);

%% 2) Design point and display window
omega0  = 4;                  % rad/s
T0      = 2*pi/omega0;        % s
Nplot   = 5;                  % final 5 complete wave cycles
Nfit    = 20;                 % final 20 complete cycles for phasor fitting

%% 3) Extract velocity logs
tp = P.v_rel_log.Time(:);
vp = squeeze(P.v_rel_log.Data);
vp = vp(:);

tc = C.v_rel_log.Time(:);
vc = squeeze(C.v_rel_log.Data);
vc = vc(:);

vrefc = squeeze(C.v_ref_log.Data);
vrefc = vrefc(:);

% Put passive velocity on the controlled-run time base.
vp_on_c = interp1(tp, vp, tc, 'linear', 'extrap');

%% 4) Phasor fits over the final 20 complete cycles
tEndFit   = tc(end);
tStartFit = tEndFit - Nfit*T0;
idxFit    = tc >= tStartFit;

[VREF,Aref,phiref] = fitTone(tc(idxFit),vrefc(idxFit),omega0);
[VP,Ap,phip]       = fitTone(tc(idxFit),vp_on_c(idxFit),omega0);
[VC,Ac,phic]       = fitTone(tc(idxFit),vc(idxFit),omega0);

Epass = VREF - VP;
Ectrl = VREF - VC;

errorReduction = 100*(1 - abs(Ectrl)/abs(Epass));

wrap180 = @(x) mod(x+180,360)-180;

fprintf('============================================================\n');
fprintf(' DDLG REPORT FIGURE SUMMARY\n');
fprintf('============================================================\n');
fprintf('Reference:  |Vref| = %.6f m/s, phase = %.6f deg\n',Aref,phiref);
fprintf('Passive:    |Vrel| = %.6f m/s, phase = %.6f deg\n',Ap,phip);
fprintf('Controlled: |Vrel| = %.6f m/s, phase = %.6f deg\n',Ac,phic);
fprintf('|E_passive| = %.6f m/s\n',abs(Epass));
fprintf('|E_PI|      = %.6f m/s\n',abs(Ectrl));
fprintf('Error reduction = %.6f %%\n',errorReduction);
fprintf('Passive phase error    = %+.6f deg\n',wrap180(phip-phiref));
fprintf('Controlled phase error = %+.6f deg\n\n',wrap180(phic-phiref));

%% 5) Output directory
figDir = fullfile(pwd,'PI_Controller_Analysis_Figures');
if ~exist(figDir,'dir')
    mkdir(figDir);
end

%% 6) Figure 1 - Passive vs PI-controlled velocity tracking
tEndPlot   = tc(end);
tStartPlot = tEndPlot - Nplot*T0;
idxPlot    = tc >= tStartPlot;

fig1 = figure('Name','DDLG Passive vs PI Velocity Tracking','Color','w', ...
              'Position',[100 100 1200 650]);

plot(tc(idxPlot),vrefc(idxPlot),'-','LineWidth',1.8)
hold on
plot(tc(idxPlot),vp_on_c(idxPlot),'--','LineWidth',1.7)
plot(tc(idxPlot),vc(idxPlot),'-.','LineWidth',1.7)
grid on
box on

xlabel('Time [s]')
ylabel('Relative Velocity [m/s]')
title('DDLG Velocity Tracking: Passive vs PI-Controlled Response')

legend('Reference v_{ref}', ...
       'Passive DDLG v_{rel}', ...
       'PI-Controlled DDLG v_{rel}', ...
       'Location','best')

xlim([tStartPlot tEndPlot])

% Report-ready metric box
metricText = sprintf([ ...
    'Passive: |V| = %.3f m/s, phase = %.2f^\\circ\n', ...
    'PI: |V| = %.3f m/s, phase = %.2f^\\circ\n', ...
    '|E_{passive}| = %.3f m/s, |E_{PI}| = %.3f m/s\n', ...
    'Tracking-error reduction = %.2f%%'], ...
    Ap,phip,Ac,phic,abs(Epass),abs(Ectrl),errorReduction);

annotation(fig1,'textbox',[0.15 0.70 0.34 0.16], ...
    'String',metricText, ...
    'FitBoxToText','on', ...
    'BackgroundColor','white', ...
    'EdgeColor',[0.55 0.55 0.55], ...
    'FontSize',10);

exportgraphics(fig1, ...
    fullfile(figDir,'DDLG_velocity_passive_vs_PI.png'), ...
    'Resolution',300);
exportgraphics(fig1, ...
    fullfile(figDir,'DDLG_velocity_passive_vs_PI.pdf'), ...
    'ContentType','vector');

%% 7) Figure 2 - Controlled force components
Fpi = squeeze(C.F_PI_log.Data);
Fpi = Fpi(:);

tPTO  = C.output.ptoSim.time(:);
Fddlg = interp1(tPTO,C.output.ptoSim.force(:),tc,'linear','extrap');
Ftot  = Fddlg + Fpi;

fig2 = figure('Name','DDLG Controlled PTO Force Components','Color','w', ...
              'Position',[100 100 1200 650]);

plot(tc(idxPlot),Fpi(idxPlot),'-','LineWidth',1.7)
hold on
plot(tc(idxPlot),Fddlg(idxPlot),'--','LineWidth',1.7)
plot(tc(idxPlot),Ftot(idxPlot),':','LineWidth',2.0)
grid on
box on

xlabel('Time [s]')
ylabel('Force [N]')
title('DDLG Controlled PTO Force Components')

legend('F_{PI}', ...
       'F_{DDLG}', ...
       'F_{PTO,total}', ...
       'Location','best')

xlim([tStartPlot tEndPlot])

[FPI,AFpi,phiFpi]       = fitTone(tc(idxFit),Fpi(idxFit),omega0);
[FDDLG,AFddlg,phiFddlg] = fitTone(tc(idxFit),Fddlg(idxFit),omega0);
[FTOT,AFtot,phiFtot]    = fitTone(tc(idxFit),Ftot(idxFit),omega0);

forceText = sprintf([ ...
    '|F_{PI}| = %.1f N, phase = %.1f^\\circ\n', ...
    '|F_{DDLG}| = %.1f N, phase = %.1f^\\circ\n', ...
    '|F_{PTO,total}| = %.1f N, phase = %.1f^\\circ'], ...
    AFpi,phiFpi,AFddlg,phiFddlg,AFtot,phiFtot);

annotation(fig2,'textbox',[0.15 0.73 0.31 0.13], ...
    'String',forceText, ...
    'FitBoxToText','on', ...
    'BackgroundColor','white', ...
    'EdgeColor',[0.55 0.55 0.55], ...
    'FontSize',10);

exportgraphics(fig2, ...
    fullfile(figDir,'DDLG_PI_force_components_report.png'), ...
    'Resolution',300);
exportgraphics(fig2, ...
    fullfile(figDir,'DDLG_PI_force_components_report.pdf'), ...
    'ContentType','vector');

%% 8) Summary
fprintf('Generated report figures:\n');
fprintf('  %s\n',fullfile(figDir,'DDLG_velocity_passive_vs_PI.png'));
fprintf('  %s\n',fullfile(figDir,'DDLG_velocity_passive_vs_PI.pdf'));
fprintf('  %s\n',fullfile(figDir,'DDLG_PI_force_components_report.png'));
fprintf('  %s\n',fullfile(figDir,'DDLG_PI_force_components_report.pdf'));

%% Local function
function [P,A,phiDeg] = fitTone(t,y,w)
    t = t(:);
    y = y(:);

    X = [cos(w*t), sin(w*t), ones(size(t))];
    beta = X\y;

    % y = Re{P exp(jwt)} + DC
    P = beta(1) - 1j*beta(2);

    A = abs(P);
    phiDeg = rad2deg(angle(P));
end
