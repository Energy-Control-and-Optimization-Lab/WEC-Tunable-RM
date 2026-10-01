function emf_ts = make_irregular_wave()
%MAKE_IRREGULAR_WAVE  Generate the JONSWAP generator EMF used by every
%                     irregular-wave simulation in this repository.
%
%   emf_ts = MAKE_IRREGULAR_WAVE() returns a timeseries of the open-circuit
%   generator EMF [V] over 0-10 s and also assigns it to the base workspace
%   as the variable "emf_ts", which is the variable the From Workspace block
%   SRC_emf inside each irregular-wave model reads.
%
%   The random phases are drawn from a FIXED seed (rng(7)), so this function
%   is deterministic: it regenerates bit-for-bit the same wave record that
%   produced every irregular-wave result in the paper. No data file needed.
%
%   Sea state (Table 6 of the paper):
%       significant wave height   Hs    = 0.25 m
%       peak period               Tp    = 1.5  s
%       peak enhancement factor   gamma = 3.3
%       spectral components       N     = 300   over 0.25-3 Hz
%
%   The surface elevation is built from the JONSWAP spectrum, differentiated
%   to a velocity (the linear-generator EMF is proportional to relative
%   velocity), and the slowly varying Hilbert envelope of that velocity is
%   mapped onto crest amplitudes of 3-6 V. The record is then scaled so that
%   its absolute peak is exactly 6 V, i.e. the irregular wave never exceeds
%   the 6 V peak of the regular-wave case. That makes the two sea states
%   directly comparable: same peak forcing, different time distribution.

% ---- JONSWAP spectrum -----------------------------------------------------
Tp = 1.5; fp = 1/Tp; gam = 3.3; Hs = 0.25; g = 9.81; N = 300;
f  = linspace(0.25, 3, N);  df = f(2) - f(1);

sig = 0.07*(f <= fp) + 0.09*(f > fp);
r   = exp(-(f - fp).^2 ./ (2*sig.^2*fp^2));
S   = 0.0081 * g^2 * (2*pi)^-4 .* f.^-5 .* exp(-1.25*(fp./f).^4) .* gam.^r;
S   = S * (Hs^2/16) / trapz(f, S);          % scale to the target Hs

% ---- random-phase realisation (FIXED SEED -> reproducible) ---------------
rng(7);
ph = 2*pi*rand(1, N);
A  = sqrt(2*S*df);

dt = 1e-5;  t = (0:dt:10)';
eta = zeros(size(t));  vel = zeros(size(t));
for k = 1:N
    eta = eta + A(k)*cos(2*pi*f(k)*t + ph(k));
    vel = vel - 2*pi*f(k)*A(k)*sin(2*pi*f(k)*t + ph(k));
end

% ---- envelope -> 3-6 V crest amplitudes, hard 6 V ceiling ---------------
env = abs(hilbert(vel));
env = movmean(env, round(0.75/dt));         % half a peak period
e0  = min(env);  e1 = max(env);
Amp = 3 + 3*(env - e0)/(e1 - e0);
emf = vel ./ max(env, 1e-3) .* Amp;
emf = emf * 6/max(abs(emf));                % exact 6 V ceiling

emf_ts = timeseries(emf, t);
assignin('base', 'emf_ts', emf_ts);

fprintf('Irregular wave ready: peak %.2f V, rms %.2f V (Hs %.2f m, Tp %.1f s, gamma %.1f, seed 7)\n', ...
        max(abs(emf)), rms(emf), Hs, Tp, gam);
end
