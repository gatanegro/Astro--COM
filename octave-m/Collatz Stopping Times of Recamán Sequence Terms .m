% Number of Recamán terms to generate
N = 100;

% Generate Recamán sequence
recaman = zeros(1, N);
used = zeros(1, N*2); % Preallocate for speed
used(1) = 1; % Mark 0 as used (MATLAB index 1)

for n = 2:N
    prev = recaman(n-1);
    candidate = prev - (n-1);
    if candidate > 0 && ~used(candidate+1)
        recaman(n) = candidate;
    else
        recaman(n) = prev + (n-1);
    end
    used(recaman(n)+1) = 1;
end

% Function to compute Collatz stopping time
collatz_steps = @(x) ...
    sum(arrayfun(@(n) collatz_single(n), x));

function steps = collatz_single(n)
    steps = 0;
    while n ~= 1
        if mod(n,2) == 0
            n = n/2;
        else
            n = 3*n + 1;
        end
        steps = steps + 1;
    end
end

% Compute Collatz stopping times for Recamán sequence (ignoring 0)
collatz_times = zeros(1, N-1);
for k = 2:N
    collatz_times(k-1) = collatz_single(recaman(k));
end

% Plotting
figure;
stem(2:N, collatz_times, 'filled','LineWidth',1.2);
xlabel('Recamán Sequence Index');
ylabel('Collatz Stopping Time');
title('Collatz Stopping Times of Recamán Sequence Terms');
grid on;
