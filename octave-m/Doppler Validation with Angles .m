% Doppler Validation with Angles
f_source = 2; 
c = 300; 
v = [50, 30, 20]; 
theta = [0, 45, 90]; % Angles in degrees
beta = v ./ c;
f_observed = f_source * sqrt((1 + beta .* cosd(theta)) ./ (1 - beta));
disp('Observed Frequencies (Hz):');
disp(f_observed);