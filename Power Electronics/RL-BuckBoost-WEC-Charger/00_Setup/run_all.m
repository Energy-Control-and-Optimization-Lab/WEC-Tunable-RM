% RUN_ALL - Run all eight simulations and print the comparison table.
%
% Four controllers x two sea states. Each run simulates 10 s of the switched
% circuit at a 1 us fixed step, so expect this script to take a while: on a
% normal laptop roughly 10-25 minutes per run, so a few hours in total. To see
% a single result quickly, open any case folder and press Run on its RUN_ME.m
% instead.
%
% A Results folder is created next to this README, holding a text table and one
% PNG per run.

addpath(fileparts(mfilename('fullpath')));
root = fileparts(fileparts(mfilename('fullpath')));
outdir = fullfile(root, 'Results');
if ~isfolder(outdir), mkdir(outdir); end

waves = {'regular', 'irregular'};
M = cell(2, 4);

for w = 1:2
    for c = 1:4
        S = run_case(waves{w}, c);
        M{w, c} = S;
        exportgraphics(gcf, fullfile(outdir, ...
            sprintf('Case%d_%s.png', c, S.wave)), 'Resolution', 150);
        close(gcf);
    end
end

% ----------------------------------------------------------------- the table
fid = fopen(fullfile(outdir, 'metrics_table.txt'), 'w');
names = {'Fixed duty cycle', 'Voltage PI', 'Cascaded PI', 'PPO (RL)'};
for f = [1 fid]
    fprintf(f, ['\n================================================' ...
                '================================================\n']);
    fprintf(f, ' COMPARISON TABLE   (regulation window 4-10 s; peaks over the whole run after 50 ms)\n');
    fprintf(f, ['================================================' ...
                '================================================\n']);
    for w = 1:2
        fprintf(f, '\n%s wave\n', M{w,1}.wave);
        fprintf(f, '%-20s %8s %8s %8s %8s %8s %9s %8s %7s\n', 'controller', ...
                'bias', 'ripple', 'maxdev', 'max v', 'min v', 'peak-peak', 'iL pk', 'duty');
        fprintf(f, '%-20s %8s %8s %8s %8s %8s %9s %8s %7s\n', '', ...
                '[mV]', '[mV]', '[mV]', '[V]', '[V]', '[mV]', '[mA]', '[-]');
        for c = 1:4
            m = M{w,c}.metrics;
            fprintf(f, '%-20s %8.2f %8.2f %8.0f %8.3f %8.3f %9.0f %8.0f %7.3f\n', names{c}, ...
                m.bias_mV, m.ripple_std_mV, m.max_dev_mV, m.max_V, m.min_V, ...
                m.peak_to_peak_mV, m.peak_iL_mA, m.mean_duty);
        end
    end
    fprintf(f, '\n');
end
fclose(fid);

save(fullfile(outdir, 'all_metrics.mat'), 'M');
fprintf('Wrote %s\n', fullfile(outdir, 'metrics_table.txt'));
