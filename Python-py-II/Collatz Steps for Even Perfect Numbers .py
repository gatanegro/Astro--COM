def collatz_steps(n):
    steps = 0
    while n != 1:
        n = n // 2 if n % 2 == 0 else 3 * n + 1
        steps += 1
    return steps

even_perfect_numbers = [6, 28, 496, 8128, 33550336, 8589869056] # Extend as needed
results = {}

for num in even_perfect_numbers:
    results[num] = collatz_steps(num)

print("Collatz Steps for Even Perfect Numbers:")
for num, steps in results.items():
    print(f"{num}: {steps}")
