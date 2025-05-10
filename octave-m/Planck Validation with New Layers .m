% Planck Validation with New Layers
h = 6.626e-34; 
K = h / (1e-9 * 1e-6); 
A = [1e-15, 1e-9, 1e-3, 1e2, 1e6]; % Amplitudes (m)
lambda = [1e-12, 1e-6, 1e-1, 1e5, 1e20]; % Wavelengths (m)
nodes = [500, 100, 50, 10, 2]; % Nodes per layer
E_layers = K .* (A .* lambda);
V = lambda.^3; % Volume per layer
rho = E_layers ./ V; % Energy density (J/m³)
disp('Energy Density per Layer (J/m³):');
disp(rho);