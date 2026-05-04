%% =========================================================
%  userDefinedFunctions.m
%  Post-processing for WEC-Sim + PTOSim (EGEC)
%  Motor: Akribis ADR175-B143
% =========================================================

%% -- Standard WEC-Sim plots ----------------------------------
waves.plotElevation(simu.rampTime);
try
    waves.plotSpectrum();
catch
end

output.plotResponse(1,3);
output.plotResponse(2,3);
output.plotForces(1,3);
output.plotForces(2,3);

%% -- PTOSim Power Analysis -----------------------------------

% Time vector
t   = output.ptoSim.time;
idx = t >= simu.rampTime;

% Electrical signals — from "To Workspace" Simulink blocks
V = voltage_out;   % [V]
I = current_out;   % [A]

% -- Mechanical power from motion -----------------------------
v_rel = output.bodies(1).velocity(:,3) - output.bodies(2).velocity(:,3);
F_pto = output.ptos(1).forceActuation(:,3);   % [N]
P_mec = -F_pto .* v_rel;                      % [W] positive = absorbed

% -- Damping verification from motion -------------------------
% B_pto = F_rms / v_rms  (valid for sinusoidal steady-state)
B_pto_eff = rms(F_pto(idx)) / rms(v_rel(idx));

% -- Electrical power -----------------------------------------
P_elec = V .* I;
P_loss = P_mec - P_elec;

% Save into output struct
output.ptoSim.voltage = V;
output.ptoSim.current = I;
output.ptoSim.P_mec   = P_mec;
output.ptoSim.P_elec  = P_elec;

% Efficiency
eta          = mean(P_elec(idx)) / mean(P_mec(idx)) * 100;
eta_expected = R_load / (Ra + R_load) * 100;

%% -- Console summary -----------------------------------------
fprintf('\n======= PTOSim Power Summary (SS: t >= %.0f s) =======\n', simu.rampTime);
fprintf('  v_rel RMS     : %8.4f m/s\n', rms(v_rel(idx)));
fprintf('  F_pto RMS     : %8.2f N\n',   rms(F_pto(idx)));
fprintf('  B_pto efectivo: %8.1f N·s/m  (target: %.0f)\n', B_pto_eff, B_pto_target);
fprintf('  ------------------------------------------------\n');
fprintf('  P_mec   mean  : %8.2f W\n',   mean(P_mec(idx)));
fprintf('  P_elec  mean  : %8.2f W\n',   mean(P_elec(idx)));
fprintf('  P_loss  mean  : %8.2f W\n',   mean(P_loss(idx)));
fprintf('  Voltage RMS   : %8.4f V\n',   rms(V(idx)));
fprintf('  Current RMS   : %8.4f A\n',   rms(I(idx)));
fprintf('  Efficiency    : %7.2f %%  (expected: %.2f %%)\n', eta, eta_expected);
fprintf('=====================================================\n\n');

%% -- 4-panel figure ------------------------------------------
figure('Name','PTOSim – EGEC Power Analysis', ...
       'Position',[80 80 1200 780], 'Color','white');

tl = tiledlayout(2,2,'TileSpacing','compact','Padding','compact');
title(tl,'Electric Generator Equivalent Circuit — Power Analysis', ...
     'FontSize',13,'FontWeight','normal');

% 1. Mechanical Power
nexttile;
plot(t, P_mec, 'Color',[0.20 0.45 0.75],'LineWidth',1);
xline(simu.rampTime,'--k','LineWidth',0.8);
xlabel('Time (s)'); ylabel('P_{mec} (W)');
title('Mechanical Power Absorbed');
text(0.98,0.05, sprintf('Mean SS = %.2f W', mean(P_mec(idx))), ...
    'Units','normalized','HorizontalAlignment','right','FontSize',9, ...
    'Color',[0.20 0.45 0.75]);
text(0.98,0.12, sprintf('B_{pto} = %.1f N·s/m', B_pto_eff), ...
    'Units','normalized','HorizontalAlignment','right','FontSize',9, ...
    'Color',[0.20 0.45 0.75]);
grid on; box off;

% 2. Terminal Voltage
nexttile;
plot(t, V, 'Color',[0.85 0.45 0.10],'LineWidth',1);
xline(simu.rampTime,'--k','LineWidth',0.8);
xlabel('Time (s)'); ylabel('V (V)');
title('Terminal Voltage');
text(0.98,0.05, sprintf('RMS SS = %.4f V', rms(V(idx))), ...
    'Units','normalized','HorizontalAlignment','right','FontSize',9, ...
    'Color',[0.85 0.45 0.10]);
grid on; box off;

% 3. Armature Current
nexttile;
plot(t, I, 'Color',[0.15 0.65 0.45],'LineWidth',1);
xline(simu.rampTime,'--k','LineWidth',0.8);
xlabel('Time (s)'); ylabel('I (A)');
title('Armature Current');
text(0.98,0.05, sprintf('RMS SS = %.4f A', rms(I(idx))), ...
    'Units','normalized','HorizontalAlignment','right','FontSize',9, ...
    'Color',[0.15 0.65 0.45]);
grid on; box off;

% 4. Electrical Power
nexttile;
plot(t, P_elec, 'Color',[0.60 0.20 0.65],'LineWidth',1);
xline(simu.rampTime,'--k','LineWidth',0.8);
xlabel('Time (s)'); ylabel('P_{elec} (W)');
title('Electrical Power Output');
text(0.98,0.05, sprintf('Mean SS = %.2f W', mean(P_elec(idx))), ...
    'Units','normalized','HorizontalAlignment','right','FontSize',9, ...
    'Color',[0.60 0.20 0.65]);
grid on; box off;