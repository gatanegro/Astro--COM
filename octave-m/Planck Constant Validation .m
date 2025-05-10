% Planck Constant Validation
h = 6.626e-34; % J·s
lambda = [1e-9, 1e-3, 1e2]; % Wavelengths (m)
frequency = 3e8 ./ lambda; % Frequency (Hz)
energy = h .* frequency; % Energy per layer (J)

nodes = [100, 50, 10];
energy_per_node = energy ./ nodes;

disp('Energy per Node (J):');
disp(energy_per_node);