def C(Omega):
    return sum(2**k * 3**(Omega - k - 1) for k in range(Omega))

print("C(3) =", C(3))  # Output: C(3) = 1*9 + 2*3 + 4*1 = 19