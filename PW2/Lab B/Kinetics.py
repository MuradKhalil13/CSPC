"""
PW2 Lab B Part 3 -- Fit a reaction rate constant k.
Read kinetics.csv, define C(t) = C0 * exp(-k*t), and minimise the squared error
using SLSQP to find the optimal reaction rate constant k.
Run: python kinetics.py
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

data = np.loadtxt("kinetics.csv", delimiter=",", skiprows=1)
t = data[:, 0]
C_measured = data[:, 1]

C0 = C_measured[0]

def total_error(k):
    C_model = C0 * np.exp(-k * t)
    error = np.sum((C_measured - C_model) ** 2)
    return error

res = minimize(total_error, x0=[0.5], method="SLSQP", bounds=[(0, 5)])
k_fitted = res.x[0]

print(f"Fitted reaction rate constant k ≈ {k_fitted:.4f}")


t_smooth = np.linspace(t.min(), t.max(), 200)
C_smooth = C0 * np.exp(-k_fitted * t_smooth)

plt.figure(figsize=(8, 5))
plt.scatter(t, C_measured, color="red", label="Measured", zorder=5)
plt.plot(t_smooth, C_smooth, color="blue", label=f"k = {k_fitted:.4f}", linewidth=2)
plt.xlabel("Time (t)")
plt.ylabel("Concentration C(t)")
plt.title("Part 3:Chemical kinetics-fit a reaction rate")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)

plt.savefig("kinetics.png", dpi=300, bbox_inches="tight")
plt.show()