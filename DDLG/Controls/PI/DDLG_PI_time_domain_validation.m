%% DDLG_PI_time_domain_validation.m
% Validate the controlled DDLG WEC-Sim run in steady state.
% Required workspace variables after wecSim:
%   output, v_ref_log, v_rel_log, F_PI_log
%
% Mechanical-power sign convention used below:
%   P_absorbed = -mean(F .* v_rel)
% Positive => absorbed from the WEC; negative => injected into the WEC.

clc

%% 1) Design point and fitting window
omega0  = 4;                  % rad/s
T0      = 2*pi/omega0;        % s
Ncycles = 20;                 % final 20 complete cycles

%% 2) Put signals on one common time base
t    = v_rel_log.Time(:);
vrel = squeeze(v_rel_log.Data);
vrel = vrel(:);

vref = interp1(v_ref_log.Time(:), squeeze(v_ref_log.Data(:)), ...
               t, 'linear', 'extrap');

Fpi = interp1(F_PI_log.Time(:), squeeze(F_PI_log.Data(:)), ...
              t, 'linear', 'extrap');

tPTO  = output.ptoSim.time(:);
Fddlg = interp1(tPTO, output.ptoSim.force(:), t, 'linear', 'extrap');
vPTO  = interp1(tPTO, output.ptoSim.vel(:),   t, 'linear', 'extrap');

Ftot = Fddlg + Fpi;

hasAbsPower = isfield(output.ptoSim,'absPower') && ~isempty(output.ptoSim.absPower);
if hasAbsPower
    Pabs_ws = interp1(tPTO, output.ptoSim.absPower(:), t, 'linear', 'extrap');
end

%% 3) Final 20 complete cycles
tEnd   = t(end);
tStart = tEnd - Ncycles*T0;
idx = (t >= tStart) & (t <= tEnd);

ts     = t(idx);
vrefs  = vref(idx);
vrels  = vrel(idx);
Fpis   = Fpi(idx);
Fddlgs = Fddlg(idx);
Ftots  = Ftot(idx);
vPTOs  = vPTO(idx);

fprintf('============================================================\n');
fprintf(' DDLG PI - WEC-SIM TIME-DOMAIN VALIDATION\n');
fprintf('============================================================\n');
fprintf('omega0      = %.6f rad/s\n',omega0);
fprintf('Fit window  = %.6f to %.6f s\n',tStart,tEnd);
fprintf('Cycles used = %d\n\n',Ncycles);

%% 4) Velocity-channel consistency check
vel_rms_diff = sqrt(mean((vrels-vPTOs).^2));
fprintf('Velocity consistency:\n');
fprintf('  RMS(v_rel_log - output.ptoSim.vel) = %.6e m/s\n\n',vel_rms_diff);

%% 5) Known-frequency least-squares sinusoidal fits
[VREF,Aref,phiref,DCref,RMSEref]       = fitTone(ts,vrefs,omega0);
[VREL,Arel,phirel,DCrel,RMSErel]       = fitTone(ts,vrels,omega0);
[E,Aerr,phierr,DCerr,RMSEerr]          = fitTone(ts,vrefs-vrels,omega0);
[FPI,AFpi,phiFpi,DCFpi,RMSEFpi]        = fitTone(ts,Fpis,omega0);
[FDDLG,AFddlg,phiFddlg,DCFddlg,RMSEF]  = fitTone(ts,Fddlgs,omega0);
[FTOT,AFtot,phiFtot,DCFtot,RMSEFtot]   = fitTone(ts,Ftots,omega0);

wrap180 = @(x) mod(x+180,360)-180;

fprintf('------------------------------------------------------------\n');
fprintf('STEADY-STATE PHASORS AT omega0 = 4 rad/s\n');
fprintf('------------------------------------------------------------\n');
fprintf('Signal       Amplitude        Phase [deg]       DC offset\n');
fprintf('v_ref     %12.6f      %12.6f      %+12.6e\n',Aref,phiref,DCref);
fprintf('v_rel     %12.6f      %12.6f      %+12.6e\n',Arel,phirel,DCrel);
fprintf('error     %12.6f      %12.6f      %+12.6e\n',Aerr,phierr,DCerr);
fprintf('F_PI      %12.6f      %12.6f      %+12.6e\n',AFpi,phiFpi,DCFpi);
fprintf('F_DDLG    %12.6f      %12.6f      %+12.6e\n',AFddlg,phiFddlg,DCFddlg);
fprintf('F_total   %12.6f      %12.6f      %+12.6e\n\n',AFtot,phiFtot,DCFtot);

fprintf('Reference check:\n');
fprintf('  expected |Vref| = 1.039494 m/s\n');
fprintf('  fitted   |Vref| = %.6f m/s\n',Aref);
fprintf('  expected phase  = 21.259000 deg\n');
fprintf('  fitted phase    = %.6f deg\n\n',phiref);

fprintf('Tracking metrics:\n');
fprintf('  |Vrel|            = %.6f m/s\n',Arel);
fprintf('  phase(Vrel)       = %.6f deg\n',phirel);
fprintf('  Vrel-Vref phase   = %+.6f deg\n',wrap180(phirel-phiref));
fprintf('  |E|               = %.6f m/s\n',Aerr);
fprintf('  |E|/|Vref|        = %.6f\n',Aerr/Aref);
fprintf('  |F_PI|            = %.6f N\n',AFpi);
fprintf('  |F_DDLG|          = %.6f N\n',AFddlg);
fprintf('  |F_total|         = %.6f N\n\n',AFtot);

E_from_phasors = VREF - VREL;
fprintf('Phasor consistency:\n');
fprintf('  |E_fit - (Vref-Vrel)| = %.6e\n\n',abs(E-E_from_phasors));

%% 6) Fit quality
fprintf('Fit RMSE over final %d cycles:\n',Ncycles);
fprintf('  v_ref   = %.6e m/s\n',RMSEref);
fprintf('  v_rel   = %.6e m/s\n',RMSErel);
fprintf('  error   = %.6e m/s\n',RMSEerr);
fprintf('  F_PI    = %.6e N\n',RMSEFpi);
fprintf('  F_DDLG  = %.6e N\n',RMSEF);
fprintf('  F_total = %.6e N\n\n',RMSEFtot);

%% 7) Peak steady-state forces
fprintf('Steady-state peak magnitudes:\n');
fprintf('  peak |F_PI|    = %.6f N\n',max(abs(Fpis)));
fprintf('  peak |F_DDLG|  = %.6f N\n',max(abs(Fddlgs)));
fprintf('  peak |F_total| = %.6f N\n\n',max(abs(Ftots)));

%% 8) Preliminary mechanical-power bookkeeping
P_DDLG_mech = -mean(Fddlgs.*vrels);
P_PI_mech   = -mean(Fpis.*vrels);
P_total     = -mean(Ftots.*vrels);

fprintf('------------------------------------------------------------\n');
fprintf('PRELIMINARY MECHANICAL POWER CHECK\n');
fprintf('positive = absorbed from WEC; negative = injected into WEC\n');
fprintf('------------------------------------------------------------\n');
fprintf('DDLG mechanical channel = %+12.6f W\n',P_DDLG_mech);
fprintf('PI mechanical channel   = %+12.6f W\n',P_PI_mech);
fprintf('Total mechanical PTO    = %+12.6f W\n',P_total);

if hasAbsPower
    Pws = mean(Pabs_ws(idx));
    fprintf('WEC-Sim ptoSim.absPower = %+12.6f W\n',Pws);
    fprintf('DDLG channel - absPower = %+12.6f W\n',P_DDLG_mech-Pws);
end

fprintf('\nNOTE:\n');
fprintf(['  The actual 70%% passive-to-reference phasor-error reduction\n', ...
         '  requires the passive DDLG velocity phasor from the same WEC-Sim\n', ...
         '  model (or a previously saved passive DDLG run).\n']);

%% Local function
function [P,A,phiDeg,dc,rmseFit] = fitTone(t,y,w)
    t = t(:);
    y = y(:);
    X = [cos(w*t), sin(w*t), ones(size(t))];
    beta = X\y;

    % y = a*cos(wt) + b*sin(wt) + dc
    % phasor convention: y = Re{P*exp(jwt)} + dc
    P = beta(1) - 1j*beta(2);

    A = abs(P);
    phiDeg = rad2deg(angle(P));
    dc = beta(3);

    yhat = X*beta;
    rmseFit = sqrt(mean((y-yhat).^2));
end
