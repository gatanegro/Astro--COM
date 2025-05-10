% Gravitational Constant with LZ and HQS
h = 6.626e-34;
LZ = 1.23498; % Scalable energy factor
HQS = 0.235; % Fixed curvature coupling
beta = 1.007e8; % Empirical scaling for G

% Energy density calculation (LZ scales energy)
lambda_ref = 1e-6;
K = h / lambda_ref^2;
lambda = [1e-6, 1e-1, 1e5];
E_layers = LZ * K .* lambda; % LZ scales energy per layer
V = lambda.^3; % Volume (m³)
rho = E_layers ./ V; % Energy density (J/m³)

% Gravitational constant (G = beta * rho)
G = beta * mean(rho);
disp('G (m³/kg/s²):');
disp(G);