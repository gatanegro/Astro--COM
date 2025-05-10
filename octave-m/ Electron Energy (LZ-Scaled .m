% Save as beta_decay.m
LZ = 1.23498;  % Scalable amplitude
HQS = 0.235;   % Curvature coupling
energy_total = 1.0 * LZ; % Total energy scaled by LZ

% Simulate electron and antineutrino energies
electron_energy = rand(1, 10000) * energy_total;
antineutrino_energy = energy_total - electron_energy;

% Apply HQS-modulated spectrum weights
spectrum_weights = HQS * electron_energy .* (energy_total - electron_energy).^2;
valid_indices = rand(1, 10000) < spectrum_weights / max(spectrum_weights);

% Plot histogram
hist(electron_energy(valid_indices), 50);
xlabel('Electron Energy (LZ-Scaled)');
ylabel('Frequency (HQS-Modulated)');