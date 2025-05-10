import numpy as np
import matplotlib.pyplot as plt

# Example data for regular moons (replace with your actual data)
regular_names = ['Mimas', 'Enceladus', 'Tethys', 'Dione', 'Rhea', 'Titan']
regular_distances = np.array([185539, 238042, 294619, 377396, 527108, 1221870])

# Example COM model function
def com_model(a0, n, phase_func='sin', lz=1.23498, hqs=0.235):
    semi_major_axes = []
    for i in range(n):
        if phase_func == 'sin':
            phase = np.sin(4 * np.pi * i)
        else:
            phase = 0
        a_n = a0 * (lz ** i) * (1 + hqs * phase)
        semi_major_axes.append(a_n)
    return semi_major_axes

best_phase = 'sin'
a0 = regular_distances[0]
extended_n = 12
extended_predictions = com_model(a0, extended_n, best_phase)

plt.figure(figsize=(12, 8))
plt.scatter(range(len(regular_distances)), regular_distances, color='red', s=100, label='Observed')
plt.plot(range(extended_n), extended_predictions, 'o-', label='COM Extended Prediction')
plt.xlabel("Moon Index")
plt.ylabel("Semi-Major Axis (km)")
plt.title("Extended COM Prediction for Saturn's Moons")
plt.yscale('log')
plt.grid(True, alpha=0.3)
plt.legend()

for i, name in enumerate(regular_names):
    plt.annotate(name, (i, regular_distances[i]), textcoords="offset points", xytext=(0,10), ha='center')

for i in range(len(regular_distances), extended_n):
    plt.axhline(y=extended_predictions[i], color='green', linestyle='--', alpha=0.3)
    plt.text(extended_n-1, extended_predictions[i], f"Predicted: {extended_predictions[i]:.0f}", va='center', ha='right', bbox=dict(facecolor='white', alpha=0.7))

plt.tight_layout()
plt.savefig('saturn_moons_extended_prediction.png', dpi=300)
plt.show()

print("\nPredictions for potential additional moons:")
print("-"*60)
print(f"{'Moon':<12} {'Predicted Distance (km)':<25}")
print("-"*60)
for i in range(len(regular_distances), extended_n):
    print(f"Moon-{i+1:<8} {extended_predictions[i]:<25.0f}")
print("="*80)
