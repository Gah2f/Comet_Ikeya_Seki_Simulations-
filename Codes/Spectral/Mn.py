from astropy.table import Table
from astropy import units as u
from astropy.constants import c, h
import numpy as np
import matplotlib.pyplot as plt

t3= Table.read('Spectral/Mn.csv')



# print(t.colnames)
# print(t[:5])
# nt= np.array([
#     float(x.replace('""', 'NAN').replace('"', '').replace('=', ''))
#     for x in t['obs_wl_vac(nm)'].data
# ]) * u.nm

# # wave_lengthv= np.array([])
# print(nt)

def clean_numeric_column(col):
    out = []
    for x in col:
        s = x.replace('"', '').replace('=', '').strip()
        if s == '' or s.lower() == 'nan':
            out.append(np.nan)
            continue
        try:
            out.append(float(s))
        except:
            out.append(np.nan)
    return np.array(out)

def clean_J_column(col):
    out = []
    
    for x in col:
        s = x.replace('"', '').replace('=', '').strip()
        if s == '' or s.lower() == 'nan':
            out.append(np.nan)
            continue
        if '/' in s:
            try:
                num, denom = s.split('/')
                out.append(float(num) / float(denom))
            except:
                out.append(np.nan)
            continue
        try:
            out.append(float(s))
        except:
            out.append(np.nan)    
    return np.array(out)

wave_lengthv = clean_numeric_column(t3['obs_wl_vac(nm)'])
A21 = clean_numeric_column(t3['Aki(s^-1)']) * 1e8 / u.s
Ji = clean_J_column(t3['J_i'])
Jk = clean_J_column(t3['J_k'])

mask = (
    np.isfinite(wave_lengthv) &
    np.isfinite(A21) &
    np.isfinite(Ji) &
    np.isfinite(Jk)
)

wave_lengthv = wave_lengthv[mask]
A21 = A21[mask]
Ji = Ji[mask]
Jk = Jk[mask]

lam = wave_lengthv * u.nm
wi = 2 * Ji + 1
wk = 2 * Jk + 1
 
B12 = (wk / wi) * A21 * lam.to(u.m) ** 5 / (8 * np.pi * h * c)

plt.figure()
plt.scatter(lam.value, B12.value, s=5)
plt.xlabel('Wavelength (nm)')
plt.ylabel('Einstein B Coefficient (m^3 s^-2)')
plt.title('Einstein B Coefficient for Manganese Atom Transitions')
plt.grid(True)
plt.show()