N = 100; % Number of Recamán terms
recaman = zeros(1, N);
used = false(1, N*5); % Over-allocate for safety
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

% Digital root function (mod 9, treating 0 as 9)
digital_root = @(x) mod(x-1,9)+1;

% Prepare 3D data
X = 1:N;
Y = recaman;
Z = arrayfun(digital_root, recaman);

% 3D scatter plot
scatter3(X, Y, Z, 50, Z, 'filled');
xlabel('Sequence Index (n)');
ylabel('Recamán Value');
zlabel('Digital Root / Energy Layer');
title('COM 3D Map of Recamán Sequence Nodes');
colorbar;
grid on;
