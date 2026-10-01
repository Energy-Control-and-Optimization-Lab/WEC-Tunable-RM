% RUN_ME - Case 1, fixed duty cycle, regular wave.
%
% Press Run (F5). Nothing else is needed: this script finds the repository,
% prepares the sea state, loads the controller, simulates 10 s, prints the
% metrics and draws the summary figure.
%
% Everything it uses lives inside this repository.

addpath(fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))), '00_Setup'));
S = run_case('regular', 1);
