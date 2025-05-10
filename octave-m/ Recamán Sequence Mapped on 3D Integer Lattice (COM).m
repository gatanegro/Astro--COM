% Parameters
N = 100;    % Number of Recamán terms
a = 10;     % Lattice width (x)
b = 10;     % Lattice height (y)

% Generate Recamán sequence
recaman = zeros(1, N);
used = false(1, N*10); % Over-allocate for safety
used(1) = true;

for n = 2:N
    prev = recaman(n-1);
    candidate = prev - (n-1);
    if candidate > 0 && ~used(candidate+1)
        recaman(n) = candidate;
    else
        recaman(n) = prev + (n-1);
    end
    used(recaman(n)+1) = true;
end

% COM 3D mapping for each Recamán number
x = mod(recaman, a);
y = mod(floor(recaman/a), b);
z = floor(recaman/(a*b));

% Plot all integer lattice points in a region for context (optional)
[X,Y,Z] = ndgrid(0:a-1, 0:b-1, 0:max(z));
figure;
scatter3(X(:), Y(:), Z(:), 10, [0.8 0.8 0.8], 'filled'); hold on;

% Plot Recamán sequence nodes
scatter3(x, y, z, 80, linspace(1,10,N), 'filled');
plot3(x, y, z, 'r-', 'LineWidth', 1.5);

xlabel('X (mod a)');
ylabel('Y (mod b)');
zlabel('Z (integer layers)');
title('Recamán Sequence Mapped on 3D Integer Lattice (COM)');
grid on;
colorbar;
view(45,30);

legend('Lattice points','Recamán nodes','Recamán path');

