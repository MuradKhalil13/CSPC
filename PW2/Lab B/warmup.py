"""
PW2 Lab B Part 2 -- three routes to a minimum.

Compare gradient descent, Newton, and SLSQP on two functions:
  2A: f(x) = (x-3)**2 + 1          (easy, one minimum at x=3)
  2B: g(x) = x**4 - 3*x**2 + x + 5 (harder, several stationary points)
Run:  python warmup.py
"""
import numpy as np
from scipy.optimize import newton, minimize

# ---------- 2A: easy convex function ----------
def f(x): 
      return (x-3)**2 + 1
def df(x): 
     return 2*(x-3)
def d2f(x):
     return 2.0


print("Part A")

# TODO 2A: minimise f three ways from x0=0 and print each result:
#   (1) gradient descent by hand (loop x = x - lr*df(x) until the step is tiny)

#lr=learning_rate

x=0.0
lr=0.1
for _ in range(100):
     x=x-lr*df(x)
print(f"Gradient result is: x={x:4f}")

#   (2) scipy.optimize.newton(df, x0, fprime=d2f)
x_newton=newton(df, x0=0.0, fprime=d2f)
print(f"Newton method result is: x={x_newton:4f}")

#   (3) scipy.optimize.minimize(f, x0, method="SLSQP")
result_min=minimize(f, x0=0.0, method="SLSQP")
print(f"SLSQP result: x={result_min.x[0]:.4f}")

x

# ---------- 2B: harder landscape ----------
def g(x):   return x**4 - 3*x**2 + x + 5
def dg(x):  return 4*x**3 - 6*x + 1
def d2g(x): return 12*x**2 - 6

print("\nPart B")

for x0 in [0, 2]:
    print(f"\nx0 = {x0}:")
    
    
    x_gd_b =x0
    for _ in range(100):
        x_gd_b = x_gd_b - lr * dg(x_gd_b)
    print(f"Gradient Descent: x ≈ {x_gd_b:.4f}")

    # 2. Newton's method
    x_newton_b = newton(dg, x0=x0, fprime=d2g)
    curv = d2g(x_newton_b)
    if curv>0:
      st_type="minimum"
    else:
      st_type="maximum"
    print(f"Newton: x ≈ {x_newton_b:.4f} (d2g = {curv:.2f} => {st_type})")

    
    x_SLSQP= minimize(g, x0=x0, method="SLSQP")
    print(f"SLSQP: x ≈ {x_SLSQP.x[0]:.4f}")

# TODO 2B: run the same three methods on g, from x0=0 AND from x0=2.
#   For Newton (which solves dg(x)=0), also check the sign of d2g at the answer:
#   d2g > 0 means a minimum, d2g < 0 means a maximum.
#   In your README note: do the methods agree? did Newton find a minimum or
#   another stationary point? how did the starting point change the result?
