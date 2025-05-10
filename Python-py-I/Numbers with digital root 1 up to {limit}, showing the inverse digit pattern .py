def find_center_numbers_with_pattern(limit):
    center_numbers = []
    for n in range(1, limit + 1):
        if (n - 1) % 9 == 0:
            center_numbers.append(n)
    return center_numbers

# Example usage
limit = 200
center_numbers = find_center_numbers_with_pattern(limit)
print(f"Numbers with digital root 1 up to {limit}, showing the inverse digit pattern:")
for i, num in enumerate(center_numbers):
    print(f"{num:3d}", end=" ")
    if (i + 1) % 10 == 0:  # New line every 10 numbers for readability
        print()
