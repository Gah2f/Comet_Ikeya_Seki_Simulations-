import numpy as np
from astropy import units as u
from sbpy.calib import Sun
import matplotlib.pyplot as plt
sun = Sun.from_default()

lam_solar = np.linspace(300,1100,2) * u.nm
# print(lam_solar)
flux = sun(lam_solar) 

plt.figure()
plt.plot(lam_solar, flux.to(u.W / (u.m**2 * u.nm)))
plt.xlabel('Wavelength(nm)')
plt.ylabel('Solar Irradiance')
plt.title('Solar Spectrum')
plt.grid()
plt.show()