import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.constants import c, h
from sbpy.calib import Sun

sun = Sun.from_default()

lam = np.array([
     2852.81, 2853.01,
    3302.37, 3302.98,
    5889.950, 5895.924,
    8183.256, 8194.790,
    8194.824, 11381.45,
    11403.78, 18465.25
]) * u.angstrom

A21 = np.array([
    0.00554,
    0.00554,
    0.0281,
    0.0281,
    0.616,
    0.614,
    0.453,
    0.090,
    0.540,
    0.090,
    0.176,
    0.140
]) * 1e8 / u.s

J_lower = np.array([
    0.5, 0.5,
    0.5, 0.5,
    0.5, 0.5,
    0.5, 1.5,
    1.5, 0.5,
    1.5, 2.5
])

J_upper = np.array([
    1.5, 0.5,
    1.5, 0.5,
    1.5, 0.5,
    1.5, 1.5,
    2.5, 0.5,
    0.5, 3.5
])

w1 = 2 * J_lower + 1
w2 = 2 * J_upper + 1

B12 = (w2 / w1) * A21 * lam.to(u.m) ** 5 / (8 * np.pi * h * c)

plt.figure()
plt.scatter(lam.value, B12.value)
plt.xlabel('Wavelength (Angstrom)')
plt.ylabel('Einstein B Coefficient (m^3 s^-2)')
plt.title('Einstein B Coefficient for Sodium Atom Transitions')

plt.grid(True)
plt.show()