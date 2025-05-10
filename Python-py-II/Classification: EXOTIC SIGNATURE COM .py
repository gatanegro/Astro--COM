import numpy as np

# =====================
# COLLATZ CORE ANALYSIS
# =====================
def collatz(n):
    sequence = [n]
    while n != 1:
        n = 3*n + 1 if n%2 else n//2
        sequence.append(n)
    return sequence

def analyze(n):
    seq = collatz(n)
    return {
        'length': len(seq),
        'max': max(seq),
        'energy': len(seq) * np.log(len(seq)+0.5772),
        'oscillations': sum(1 if x>y else -1 for x,y in zip(seq[1:], seq[:-1]))
    }

# ====================
# TEXT-BASED INTERFACE
# ====================
print("\nCOLLATZ OCTAVE MODEL ANALYSIS")
print("=============================")

for number in range(1, 21):  # Analyze numbers 1-20
    result = analyze(number)
    print(f"\nNumber: {number}")
    print(f"Sequence length: {result['length']} steps")
    print(f"Maximum value: {result['max']}")
    print(f"Topological energy: {result['energy']:.2f}")
    print(f"Net oscillations: {result['oscillations']:+}")

    # Simple classification
    if result['energy'] > 50:
        print("Classification: EXOTIC SIGNATURE")
    elif result['energy'] > 25:
        print("Classification: TORSION DETECTED")
    else:
        print("Classification: TRIVIAL PATH")

print("\nAnalysis complete. Key:")
print("- Energy <25: Safe numbers")
print("- Energy 25-50: Interesting patterns")
print("- Energy >50: Potential exotic behavior")