"""
PW2 Lab B Part 5 (bonus) -- find a titration's equivalence point.

titration.csv holds a titration curve: pH versus the volume of base added.
The equivalence point is the volume where the pH changes fastest (the steep
jump). Numerically, that is where the SLOPE of the pH curve is largest.
Run:  python titration.py
"""
import numpy as np
import matplotlib.pyplot as plt

# TODO 1: read titration.csv (columns volume_base, pH) into arrays V, pH.
data = np.loadtxt("titration.csv", delimiter=",", skiprows=1)
volume = data[:, 0]
pH = data[:, 1]

# TODO 2: compute the slope of the pH curve with np.gradient(pH, V), and find
#         the volume where that slope is largest (np.argmax). That is the
#         equivalence point. Print it.
slope = np.gradient(pH, volume)
max_slope_idx = np.argmax(slope)
eq_volume = volume[max_slope_idx]

print(f"Equivalence point is at approximately volume V = {eq_volume:.2f} mL")

# TODO 3: make two plots side by side: (left) pH vs volume with a line at the
#         equivalence point; (right) the slope vs volume, showing it peaks
#         at the equivalence point. Save as titration.png.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(volume, pH, color="purple", linewidth=2)
ax1.axvline(eq_volume, color="red", linestyle="--", label=f"Eq. Point ({eq_volume:.2f} mL)")
ax1.set_xlabel("Base volume (mL)")
ax1.set_ylabel("pH")
ax1.set_title("Titration curve(pH vs Volume)")
ax1.legend()
ax1.grid(True, linestyle="--", alpha=0.6)

ax2.plot(volume, slope, color="teal", linewidth=2)
ax2.axvline(eq_volume, color="red", linestyle="--", label=f"Max Slope ({eq_volume:.2f} mL)")
ax2.set_xlabel("Base volume (mL)")
ax2.set_ylabel("Titration slope(dpH/dV)")
ax2.set_title("pH Change rate (Slope)")
ax2.legend()
ax2.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()

plt.savefig("titration.png", dpi=300, bbox_inches="tight")
plt.show()
