def digital_root(n):
    if n == 0:
        return 0
    return 1 + ((n - 1) % 9)

def find_center_numbers(limit):
    center_numbers = []
    for n in range(1, limit + 1):
        if digital_root(n) == 1:
            center_numbers.append(n)
    return center_numbers

# Example usage
limit = 100
center_numbers = find_center_numbers(limit)
print(f"Numbers with digital root 1 up to {limit}:")
print(center_numbers)
