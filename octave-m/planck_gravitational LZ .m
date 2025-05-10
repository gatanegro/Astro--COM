% Save as planck_gravitational.m
h = 6.626e-34;
LZ = 1.23498;
lambda = [1e-9, 1e-3, 1e2];
energy = LZ * h * 3e8 ./ lambda;
disp('Energy (J):'), disp(energy);