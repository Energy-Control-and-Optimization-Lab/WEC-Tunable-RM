% userDefinedFunctions_PI.m
% WEC-Sim post-processing for the PI-controlled SD case.
% Retains the standard WEC-Sim plots and adds controller / PTO figures
% used for the PI-controller report results.

%% Standard WEC-Sim plots

% Plot waves
waves.plotElevation(simu.rampTime);
try
    waves.plotSpectrum();
catch
end

% Plot heave response for body 1 (float)
output.plotResponse(1,3);

% Plot heave response for body 2 (spar)
output.plotResponse(2,3);

% Plot heave forces for body 1 (float)
output.plotForces(1,3);

% Plot heave forces for body 2 (spar)
output.plotForces(2,3);

%% PI-controller signals from Simulink To Workspace blocks
% Expected To Workspace variable names:
%   v_ref, v_rel, error, F_PI
% Saved format: Timeseries

try
    % Read the Simulink To Workspace signals from the base workspace.
    vRefSig = evalin('base','v_ref');
    vRelSig = evalin('base','v_rel');
    errSig  = evalin('base','error');
    fPiSig  = evalin('base','F_PI');

    %% Figure: PI controller signals
    figure('Name','PI Controller Signals','Color','w');
    tl = tiledlayout(4,1,'TileSpacing','compact','Padding','compact');
    title(tl,'PI Controller Signals - Ramped Reference','FontWeight','bold');

    nexttile
    plot(vRefSig.Time, squeeze(vRefSig.Data), 'LineWidth',1.25)
    grid on
    ylabel('v_{ref} (m/s)')
    set(gca,'FontSize',11)

    nexttile
    plot(vRelSig.Time, squeeze(vRelSig.Data), 'LineWidth',1.25)
    grid on
    ylabel('v_{rel} (m/s)')
    set(gca,'FontSize',11)

    nexttile
    plot(errSig.Time, squeeze(errSig.Data), 'LineWidth',1.25)
    grid on
    ylabel('e (m/s)')
    set(gca,'FontSize',11)

    nexttile
    plot(fPiSig.Time, squeeze(fPiSig.Data), 'LineWidth',1.25)
    grid on
    ylabel('F_{PI} (N)')
    xlabel('Time (s)')
    set(gca,'FontSize',11)

    %% Figure: reference tracking
    figure('Name','PI Velocity Tracking','Color','w');
    plot(vRefSig.Time, squeeze(vRefSig.Data), 'LineWidth',1.35)
    hold on
    plot(vRelSig.Time, squeeze(vRelSig.Data), 'LineWidth',1.35)
    grid on
    xlabel('Time (s)')
    ylabel('Relative velocity (m/s)')
    legend('v_{ref}','v_{rel}','Location','best')
    title('PI Velocity Tracking - Ramped Reference')
    set(gca,'FontSize',11)

    %% PTO power analysis
    % Use the physical PTO damping already defined in the WEC-Sim model.
    try
        Bpto = pto(1).damping;
    catch
        Bpto = 540; % fallback for this SD case [N s/m]
    end

    % Wave period from the WEC-Sim regular-wave definition.
    try
        T = waves.period;
        if numel(T) > 1
            T = T(1);
        end
    catch
        T = 2*pi/4; % fallback for omega0 = 4 rad/s
    end

    % Analyze up to the last 30 complete steady-state wave periods.
    t2 = min(vRelSig.Time(end), fPiSig.Time(end));
    availableSteadyTime = max(t2 - simu.rampTime, T);
    Ncycles = min(30, max(1, floor(availableSteadyTime/T)));
    t1 = t2 - Ncycles*T;

    % Common high-resolution time vector for power calculations.
    nPowerSamples = max(5000, 1000*Ncycles);
    tPower = linspace(t1,t2,nPowerSamples).';

    vData = squeeze(vRelSig.Data);
    fData = squeeze(fPiSig.Data);

    v = interp1(vRelSig.Time, vData, tPower, 'pchip');
    Fpi = interp1(fPiSig.Time, fData, tPower, 'pchip');

    % PTO force applied to the WEC.
    Fdamp = -Bpto*v;
    Ftotal = Fdamp + Fpi;

    % Positive sign denotes mechanical power absorbed by the PTO.
    pDamp  = Bpto*v.^2;
    pPI    = -Fpi.*v;
    pTotal = pDamp + pPI;

    % Time-average over an integer number of wave periods.
    P_damp  = trapz(tPower,pDamp)/(tPower(end)-tPower(1));
    P_PI    = trapz(tPower,pPI)/(tPower(end)-tPower(1));
    P_total = trapz(tPower,pTotal)/(tPower(end)-tPower(1));

    fprintf('\nPI-controlled PTO power analysis\n');
    fprintf('Analysis window: %.4f to %.4f s (%d periods)\n',t1,t2,Ncycles);
    fprintf('Average passive damping power = %10.4f W\n',P_damp);
    fprintf('Average PI-channel power      = %10.4f W\n',P_PI);
    fprintf('Average NET PTO power         = %10.4f W\n',P_total);
    fprintf('Peak |F_damp|                 = %10.4f N\n',max(abs(Fdamp)));
    fprintf('Peak |F_PI|                   = %10.4f N\n',max(abs(Fpi)));
    fprintf('Peak |F_total|                = %10.4f N\n\n',max(abs(Ftotal)));

    %% Figure: instantaneous PTO power over the last five wave periods
    nPlotCycles = min(5,Ncycles);
    idxPlot = tPower >= (t2 - nPlotCycles*T);

    figure('Name','PI-Controlled PTO Instantaneous Power','Color','w');
    plot(tPower(idxPlot),pDamp(idxPlot),'LineWidth',1.30)
    hold on
    plot(tPower(idxPlot),pPI(idxPlot),'LineWidth',1.30)
    plot(tPower(idxPlot),pTotal(idxPlot),'LineWidth',1.50)
    grid on
    xlabel('Time (s)')
    ylabel('Mechanical power (W)')
    legend('Passive damping','PI channel','Net PTO','Location','best')
    title('PI-Controlled PTO Instantaneous Power')
    set(gca,'FontSize',11)

    %% Figure: average PTO power decomposition
    figure('Name','PI-Controlled PTO Power Decomposition','Color','w');
    bar(categorical({'Passive damping','PI channel','Net PTO'}), ...
        [P_damp P_PI P_total])
    grid on
    ylabel('Average mechanical power (W)')
    title('PI-Controlled PTO Power Decomposition')
    set(gca,'FontSize',11)

catch ME
    warning('PI report plots were not generated: %s', ME.message);
end
