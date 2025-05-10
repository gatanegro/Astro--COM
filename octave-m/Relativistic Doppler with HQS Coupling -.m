% Relativistic Doppler with HQS Coupling
f_source = 2; % Hz
c = 3e8; % m/s
HQS = 0.235; % Fixed coupling
v = [50, 30, 20]; % m/s

beta = v ./ c;
f_observed = f_source * sqrt((1 + beta) ./ (1 - beta)) * HQS;

disp('Observed Frequencies (Hz):');
disp(f_observed);