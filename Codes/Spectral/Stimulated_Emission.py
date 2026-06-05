import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy import units as u
from astropy.constants import c, h

t1 = Table.read('Spectral/Na(2).csv')

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

wave_lengthv = clean_numeric_column(t1['obs_wl_vac(nm)'])
A21 = clean_numeric_column(t1['Aki(s^-1)']) * 1e8 / u.s
Ji = clean_J_column(t1['J_i'])
Jk = clean_J_column(t1['J_k'])

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
rho = 1e-9
lam = wave_lengthv * u.nm
wi = 2 * Ji + 1
wk = 2 * Jk + 1
 
B12 = (wk / wi) * A21 * lam.to(u.m) ** 5 / (8 * np.pi * h * c)
R_up = B12 * rho

N_atoms = 10000
atoms = np.zeros(N_atoms, dtype = int)

B21 = (wk / wi) * B12
R_down = A21 + B21 * rho


dt = 1e-9

P_up = 1 - np.exp(-R_up * dt)
P_down = 1 - np.exp(-R_down * dt)

def propagate(atoms, P_up, P_down):
    rand = np.random.random(len(atoms))
    ground = atoms == 0 
    excited = atoms == 1
    
    excite = ground & (rand < P_up)
    decay = excited & (rand < P_down)
    
    atoms[excite] = 1
    atoms[decay] = 0
    
    return atoms

n_steps = 5000

excited_history = []
time_history = []

for step in range(n_steps):
    
    atoms = propagate(atoms, P_up, P_down)
    
    excited_history.append(np.sum(atoms))
    time_history.append(step * dt)

plt.figure()
plt.plot(time_history, excited_history)

plt.xlabel('Time (s)')
plt.ylabel('Number of excited atoms')
plt.title('Two-level Monte Carlo Simulation')
plt.grid()
plt.show()