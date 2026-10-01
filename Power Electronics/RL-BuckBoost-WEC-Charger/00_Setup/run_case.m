function S = run_case(wave, caseNo, varargin)
%RUN_CASE  Run one controller on one sea state, plot it, and print its metrics.
%
%   RUN_CASE('regular',   1)   fixed duty cycle,  regular 6 V / 0.6667 Hz wave
%   RUN_CASE('regular',   2)   single voltage PI
%   RUN_CASE('regular',   3)   cascaded (double) PI
%   RUN_CASE('regular',   4)   PPO reinforcement-learning controller
%   RUN_CASE('irregular', 1..4) the same four controllers on the JONSWAP wave
%
%   S = RUN_CASE(...) also returns a struct with the time vector, the battery
%   voltage and current, the inductor current, the duty cycle, the SOC, and
%   the computed metrics.
%
%   Name-value options
%     'Decimation' (default 10)  log every Nth solver step. The solver always
%                                runs at 1 us; decimation only thins what is
%                                SAVED. 10 -> 100 kHz logs (5x the 20 kHz
%                                switching frequency, so the ripple is still
%                                fully resolved) and about 100 MB of RAM.
%                                Set 1 to log at the full 1 MHz exactly as the
%                                paper did; that needs roughly 1 GB of RAM.
%     'StopTime'   (default 10)  simulation length in seconds.
%     'Plot'       (default true) draw the 5-panel summary figure.
%
%   Everything this function needs is inside this repository. It requires no
%   files, variables or paths from outside it.
%
%   Requires: MATLAB R2024b, Simulink, Simscape, Simscape Electrical,
%             and for Case 4 only: Reinforcement Learning Toolbox and
%             Deep Learning Toolbox.
%
%   NOTE ON MATLAB VERSION: these models are built with Simscape Electrical.
%   They were developed and validated on R2024b. Release R2026a removed the
%   Specialized Power Systems library, so the models will not compile there.

% ------------------------------------------------------------------ options
p = inputParser;
p.addParameter('Decimation', 10, @(x) isnumeric(x) && x >= 1);
p.addParameter('StopTime',   10, @isnumeric);
p.addParameter('Plot',     true, @islogical);
p.parse(varargin{:});
dec  = round(p.Results.Decimation);
tEnd = p.Results.StopTime;

% ------------------------------------------------- locate the model to run
root = fileparts(fileparts(mfilename('fullpath')));   % repository root
addpath(fullfile(root, '00_Setup'));

wave = lower(string(wave));
switch wave
    case "regular",   wdir = '01_Regular_Wave';   wtag = 'Regular';
    case "irregular", wdir = '02_Irregular_Wave'; wtag = 'Irregular';
    otherwise, error('run_case: wave must be ''regular'' or ''irregular''.');
end

cdirs = {'Case1_Fixed_Duty','Case2_Voltage_PI','Case3_Cascaded_PI','Case4_PPO_RL'};
cname = {'Fixed duty cycle','Voltage PI','Cascaded PI','PPO (reinforcement learning)'};
if ~ismember(caseNo, 1:4), error('run_case: caseNo must be 1, 2, 3 or 4.'); end

cdir  = fullfile(root, wdir, cdirs{caseNo});
mdl   = sprintf('Case%d_%s', caseNo, wtag);
mfile = fullfile(cdir, [mdl '.slx']);
if ~isfile(mfile)
    error('run_case: model not found:\n  %s', mfile);
end

fprintf('\n==============================================================\n');
fprintf(' %s wave  |  Case %d: %s\n', wtag, caseNo, cname{caseNo});
fprintf('==============================================================\n');

% --------------------------------------------- sea state (source of the EMF)
% Regular wave: the EMF is an ideal 6 V / 0.6667 Hz AC source inside the model,
% so nothing has to be prepared. Irregular wave: the model reads the base
% workspace variable emf_ts, which make_irregular_wave regenerates from a
% fixed random seed.
if wave == "irregular"
    make_irregular_wave();
end

% ------------------------------------------------- trained policy (Case 4 only)
if caseNo == 4
    agfile = fullfile(cdir, sprintf('ppo_agent_%s.mat', lower(wtag)));
    if ~isfile(agfile)
        error('run_case: trained agent not found:\n  %s', agfile);
    end
    A  = load(agfile);
    fn = fieldnames(A);
    agentObj = [];
    for k = 1:numel(fn)
        if isa(A.(fn{k}), 'rl.agent.AbstractAgent'), agentObj = A.(fn{k}); break; end
    end
    if isempty(agentObj)
        error('run_case: %s contains no reinforcement-learning agent.', agfile);
    end
    % Deterministic evaluation: use the learned mean action, no exploration noise.
    agentObj.UseExplorationPolicy = false;
    assignin('base', 'agentObj', agentObj);
    fprintf('Trained PPO policy loaded (exploration disabled).\n');
end

% --------------------------------------------------------------- run the model
cleanupModel = onCleanup(@() closeIfOpen(mdl));
load_system(mfile);
set_param(mdl, 'StopTime', num2str(tEnd), 'ReturnWorkspaceOutputs', 'on');

% Thin the logged data. This changes only what is stored, never the solver,
% which always advances at the 1 us fixed step.
tw = find_system(mdl, 'BlockType', 'ToWorkspace');
for k = 1:numel(tw), set_param(tw{k}, 'Decimation', num2str(dec)); end

fprintf('Simulating %g s at a 1 us fixed step (ode4), logging every %d steps ...\n', tEnd, dec);
tic; out = sim(mdl); wall = toc;
fprintf('Done in %.0f s of wall-clock time.\n', wall);

% --------------------------------------------------------- collect the signals
% Each logging block carries its own time vector, because the controller
% signals and the circuit signals are not all sampled at the same rate. Keep
% them separate rather than assuming one common time base.
S.t     = getsig(out, 'log_BBV', 'time');
S.v_bat = getsig(out, 'log_BBV');
S.i_bat = getsig(out, 'log_BBI');
S.i_L   = getsig(out, 'log_IL');
S.v_dc  = getsig(out, 'log_Vout_rect');
S.v_gen = getsig(out, 'log_Vin_gen');

% The battery block logs CHARGE in coulombs. The cell holds 2 mAh = 7.2 C, so
% dividing by 7.2 converts it to a state-of-charge fraction (0.5 at t = 0).
S.soc    = getsig(out, 'log_SOC') / 7.2;
S.t_soc  = getsig(out, 'log_SOC', 'time');

S.duty   = getsig(out, 'log_Duty');
S.t_duty = getsig(out, 'log_Duty', 'time');
if isempty(S.duty)                       % Case 1 is open loop: no duty log
    S.duty = 0.74*ones(size(S.t));  S.t_duty = S.t;
end
S.wave = wtag;  S.caseNo = caseNo;  S.name = cname{caseNo};

% ----------------------------------------------------------------- metrics
% Regulation window 4-10 s: the DC link has settled by 4 s, so this window
% measures steady-state regulation rather than start-up. The peak window skips
% the first 50 ms, which is the capacitor pre-charge transient of the circuit
% and not a property of the controller.
%
% These are the definitions used for the table in the paper. In particular the
% error figure is the BIAS of the regulated voltage, |mean(v_bat) - 12 V|, not
% the mean of the absolute error; the ripple is reported separately as the
% standard deviation.
t0 = 4;
if S.t(end) < t0            % short run (a quick test): use the last 60 %
    t0 = 0.4*S.t(end);
    fprintf('\n  NOTE: run is shorter than 4 s, so the regulation window is %.2f-%.2f s.\n', t0, S.t(end));
end
reg = S.t >= t0;  pk = S.t > 0.05;
v   = S.v_bat(reg);

S.metrics = struct( ...
    'window_s',          [t0 S.t(end)], ...
    'bias_mV',           abs(mean(v) - 12)*1e3, ...
    'ripple_std_mV',     std(v)*1e3, ...
    'max_dev_mV',        max(abs(v - 12))*1e3, ...
    'in_band_pct',       100*mean(abs(S.v_bat(S.t >= min(1, 0.1*S.t(end))) - 12) < 0.05), ...
    'max_V',             max(S.v_bat(pk)), ...
    'min_V',             min(S.v_bat(pk)), ...
    'peak_to_peak_mV',  (max(S.v_bat(pk)) - min(S.v_bat(pk)))*1e3, ...
    'peak_iL_mA',        max(S.i_L)*1e3, ...
    'mean_duty',         mean(S.duty(S.t_duty >= t0)), ...
    'final_SOC_pct',     S.soc(end)*100);

fprintf('\n  Regulation window %.2f-%.2f s\n', t0, S.t(end));
fprintf('    |mean(v_bat) - 12 V|     %9.2f mV\n', S.metrics.bias_mV);
fprintf('    ripple (std of v_bat)    %9.2f mV\n', S.metrics.ripple_std_mV);
fprintf('    largest deviation        %9.2f mV\n', S.metrics.max_dev_mV);
fprintf('    mean duty cycle          %9.3f\n',    S.metrics.mean_duty);
fprintf('  Whole run after 50 ms\n');
fprintf('    highest v_bat            %9.3f V\n',  S.metrics.max_V);
fprintf('    lowest  v_bat            %9.3f V\n',  S.metrics.min_V);
fprintf('    peak-to-peak excursion   %9.0f mV\n', S.metrics.peak_to_peak_mV);
fprintf('    peak inductor current    %9.0f mA\n', S.metrics.peak_iL_mA);
fprintf('  Whole run after 1 s\n');
fprintf('    time within 12 V +/-50 mV %8.1f %%\n', S.metrics.in_band_pct);
fprintf('    final battery SOC        %9.2f %%\n', S.metrics.final_SOC_pct);

% ------------------------------------------------------------------- figure
if p.Results.Plot, plot_case(S); end
end

% =========================================================================
function v = getsig(out, name, what)
%GETSIG  Pull one logged signal out of the simulation output by its name.
if nargin < 3, what = 'data'; end
v = [];
lg = out.who;
i  = find(strcmp(lg, name), 1);
if isempty(i), return; end
x = out.get(lg{i});
if isa(x, 'timeseries')
    if strcmp(what, 'time'), v = x.Time; else, v = squeeze(x.Data); end
else
    if strcmp(what, 'time'), v = x.time; else, v = squeeze(x.signals.values); end
end
v = double(v(:));
end

% =========================================================================
function plot_case(S)
%PLOT_CASE  Five-panel summary: wave, battery voltage, battery current,
%           duty cycle, state of charge.
f = figure('Color', 'w', 'Position', [60 40 1100 900], ...
           'Name', sprintf('Case %d - %s wave', S.caseNo, S.wave));
ttl = sprintf('%s wave  -  Case %d: %s', S.wave, S.caseNo, S.name);

ax(1) = subplot(5,1,1);
plot(S.t, S.v_gen, 'Color', [0 0.35 0.7]); grid on
ylabel('v_{gen} [V]'); title(ttl, 'FontWeight', 'bold');

ax(2) = subplot(5,1,2);
plot(S.t, S.v_bat, 'k'); grid on; hold on
yline(12, 'r--', 'LineWidth', 1);
ylabel('v_{bat} [V]');
legend('v_{bat}', 'reference 12 V', 'Location', 'southeast');

ax(3) = subplot(5,1,3);
plot(S.t, S.i_bat, 'Color', [0.85 0.33 0.1]); grid on
ylabel('i_{bat} [A]');

ax(4) = subplot(5,1,4);
plot(S.t_duty, S.duty, 'Color', [0.47 0.67 0.19]); grid on
ylabel('duty [-]'); ylim([0 1]);

ax(5) = subplot(5,1,5);
plot(S.t_soc, S.soc*100, 'Color', [0.49 0.18 0.56]); grid on
ylabel('SOC [%]'); xlabel('time [s]');

linkaxes(ax, 'x'); xlim(ax(1), [0 S.t(end)]);
drawnow;
end

% =========================================================================
function closeIfOpen(mdl)
if bdIsLoaded(mdl), close_system(mdl, 0); end   % never save over the shipped model
end
