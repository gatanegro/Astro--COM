% Doppler Effect with Fixed HQS Coupling
f_source = 2; % Source frequency (Hz)
c = 3e8; % Speed of light (m/s)
HQS = 0.235; % Fixed curvature coupling (23.5%)
v = [50, 30, 20]; % Velocities (m/s)

% Relativistic Doppler formula scaled by HQS
beta = v ./ c; % Velocity ratio (v/c)
f_observed = f_source * sqrt((1 + beta) ./ (1 - beta)) * HQS;

disp('Observed Frequencies (Hz):');
disp(f_observed);