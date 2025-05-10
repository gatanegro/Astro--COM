def digital_root(n):
    if n == 0:
        return 0
    return 1 + ((n - 1) % 9)

def collatz_steps(n):
    steps = 0
    while n != 1:
        n = n // 2 if n % 2 == 0 else 3 * n + 1
        steps += 1
    return steps

even_perfect_numbers = [6, 28, 496, 8128, 33550336, 8589869056]
results = {}

for num in even_perfect_numbers:
    steps = collatz_steps(num)
    root = digital_root(num)
    results[num] = (steps, root)

print("Collatz Steps and Digital Roots for Even Perfect Numbers:")
for num, (steps, root) in results.items():
    print(f"{num}: Steps = {steps}, Digital Root = {root}")

