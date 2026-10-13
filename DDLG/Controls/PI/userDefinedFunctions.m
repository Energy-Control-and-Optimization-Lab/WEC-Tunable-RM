%% Plot waves and responses
waves.plotElevation(simu.rampTime);
try
    waves.plotSpectrum();
catch
end

% Plot heave response for body 1
output.plotResponse(1,3);

% Plot heave response for body 2
output.plotResponse(2,3);

% Plot heave forces for body 1
output.plotForces(1,3);

% Plot heave forces for body 2
output.plotForces(2,3);

%% Power post-processing — DDLG
t   = output.ptoSim.time;                   % time vector [s]
idx = t > simu.rampTime;                    % logical index: exclude ramp-up period

% --- Time series ---
P_mec       = output.ptoSim.absPower(:,end);   % mechanical power [W]
P_elec      = output.ptoSim.elecPower(:,end);  % electrical power [W]
P_elec_conv = P_elec * eta_conv;               % electrical power after converter [W]

% --- Mean values (post-ramp only) ---
mean_P_mec       = mean(P_mec(idx));
mean_P_elec      = mean(P_elec(idx));
mean_P_elec_conv = mean(P_elec_conv(idx));

fprintf('\n--- Power Results ---\n');
fprintf('mean_P_mec       = %.4f W\n', mean_P_mec);
fprintf('mean_P_elec      = %.4f W\n', mean_P_elec);
fprintf('mean_P_elec_conv = %.4f W  (eta_conv = %.2f)\n', mean_P_elec_conv, eta_conv);

%% Save to output folder
% Note: folder named 'output_data' to avoid conflict with WEC-Sim's 'output' object
if ~exist('output_data', 'dir')
    mkdir('output_data');
end

save('output_data/power_results.mat', ...
    't', ...
    'P_mec', 'P_elec', 'P_elec_conv', ...
    'mean_P_mec', 'mean_P_elec', 'mean_P_elec_conv', ...
    'eta_conv');

fprintf('Results saved to output_data/power_results.mat\n');
