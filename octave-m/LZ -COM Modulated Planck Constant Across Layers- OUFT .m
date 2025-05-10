alpha = [6.626e-34, 1.0e-20, 1.0e-10]; % FIELD proportionality
amplitude = [1e-9, 1e-3, 1.0]; % Oscillatory amplitude
LZ = 1.23498; % Scalable amplitude

h_dynamic = LZ .* alpha .* amplitude; % LZ scales h across layers
bar(h_dynamic);
ylabel('Planck Constant (h)');
title('LZ-Modulated Planck Constant Across Layers');