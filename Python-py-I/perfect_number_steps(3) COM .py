def collatz_steps(n):
    steps = 0
    while n != 1:
        n = n // 2 if n % 2 == 0 else 3 * n + 1
        steps += 1
    return steps

def perfect_number_steps(p):
    # Calculate steps for the Mersenne prime M_p
    mersenne = (1 << p) - 1  # Equivalent to 2^p - 1
    mersenne_steps = collatz_steps(mersenne)
    return (p - 1) + mersenne_steps

# Example: Steps for the perfect number with p=3 (28)
print(perfect_number_steps(3))  # Output: 18