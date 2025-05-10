beta = [1.0e-10, 1.0, 1.0e10]; % FIELD scaling
rho = [1e30, 1e10, 1e-5]; % Node density
grad_E = [1e5, 1e2, 1e-3]; % Energy gradient
HQS = 0.235; % Curvature coupling (23.5%)

G_dynamic = HQS .* beta .* rho .* grad_E; % HQS modulates G
bar(G_dynamic);
ylabel('Gravitational Constant (G)');
title('HQS-Coupled Gravitational Constant');