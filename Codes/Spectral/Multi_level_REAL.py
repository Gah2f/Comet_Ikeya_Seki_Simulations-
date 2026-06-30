import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.constants import h,c, k_B 

cm_inv = 1/u.cm
nm = u.nm
m = u.m
s = u.s
Hz = u.Hz
J = u.J
W = u.W


file_path_1 = 'Spectral/Na_All_Wavelength.csv'
file_path_2 = 'Spectral/Spectre_HR_LATMOS_Meftah_V1.csv'


df = pd.read_csv(file_path_1, skipinitialspace=True)
df.columns = df.columns.str.strip()

df2 = pd.read_csv(file_path_2, sep=r"\s+", engine="python")
df2.columns = df2.columns.str.strip()
# print(df2.columns.tolist())

for col in df.columns:
    df[col] = (
        df[col].astype(str).str.replace('="', '', regex=False)
        .str.replace('"', '', regex=False)
        .str.replace('=', '', regex=False)
        .str.strip()
    )

# print(df.head())
# print(df.columns)
# print("Rows:", len(df))

df["Ei(cm-1)"] = pd.to_numeric(df["Ei(cm-1)"], errors="coerce") 
df["Ek(cm-1)"] = pd.to_numeric(df["Ek(cm-1)"], errors="coerce") 
df["Aki(s^-1)"] = pd.to_numeric(df["Aki(s^-1)"], errors="coerce")
df = df.dropna(subset=["Aki(s^-1)"])

df["level_i"] = (
    df["conf_i"].astype(str).str.strip() + " " +
    df["term_i"].astype(str).str.strip() + " J=" +
    df["J_i"].astype(str).str.strip()
)

df["level_k"] = (
    df["conf_k"].astype(str).str.strip() + " " +
    df["term_k"].astype(str).str.strip() + " J=" +
    df["J_k"].astype(str).str.strip()
)

df2["Lambda[nm]"] = pd.to_numeric(
    df2["Lambda[nm]"],
    errors="coerce"
)

df2["SSI[W.m^(-2).nm^(-1)]"] = pd.to_numeric(
    df2["SSI[W.m^(-2).nm^(-1)]"],
    errors="coerce"
)


N_atoms = 10000
n_steps = 4000

level_energy = {} 
level_J = {}

for _,row in df.iterrows():
    if pd.notna(row['Ei(cm-1)']):
        level_energy[row['level_i']] = row['Ei(cm-1)']
        
        try: 
            level_J[row['level_i']] = float(str(row['J_i']).split(',')[0])
        except:
            level_J[row['level_i']] = 0.5

    if pd.notna(row['Ek(cm-1)']):
        level_energy[row['level_k']] = row['Ek(cm-1)']
        try:
            level_J[row['level_k']] = float(str(row['J_k']).split(',')[0])
        except:
            level_J[row['level_k']] = 0.5

levels = sorted(level_energy.keys(), key=lambda x: level_energy[x])
n_levels = len(levels)

level_to_index = {level: i for i, level in enumerate(levels)}

R = np.zeros((n_levels, n_levels))

for _,row in df.iterrows():
    level_i = row['level_i']
    level_k = row['level_k']
    
    if level_i not in level_to_index:
        continue
    if level_k not in level_to_index:
        continue
    
    Ei = row["Ei(cm-1)"] * cm_inv 
    Ek = row["Ek(cm-1)"] * cm_inv 
    
    if np.isnan(Ei.value) or np.isnan(Ek.value):
        continue
    
    i = level_to_index[level_i]
    k_index = level_to_index[level_k]
    
    if Ek < Ei:
        continue
    
    delta_cm = Ek - Ei
    
    nu = (delta_cm).to(Hz, equivalencies = u.spectral())
    lam_nm = (delta_cm).to(nm, equivalencies = u.spectral())
    lam_m = (delta_cm).to(m, equivalencies = u.spectral())
  
    Aki = row["Aki(s^-1)"] * (1/s)
    
    if np.isnan(Aki):
        continue
    
    gi = 2 * level_J[level_i] + 1
    gk = 2 * level_J[level_k] + 1 
    
    Bki = (
        c**3 / (8*np.pi * h * nu**3)
    ) * Aki
    
    Bik = (gk/gi) * Bki 
    
    SSI = np.interp(
        lam_nm.to_value(nm),
        df2["Lambda[nm]"].to_numpy(),
        df2["SSI[W.m^(-2).nm^(-1)]"].to_numpy()
    ) 
    
    SSI = SSI * (W/(m**2 * nm))

    u_lambda = SSI / c
    
    rho = u_lambda * (lam_m**2 / c )
    try:
        downward_rate = (Aki + Bki * rho).to(1/s).value
        upward_rate = (Bik * rho).to(1/s).value 
    except Exception as e:
        print("Rate conversion error:")
        print("delta_cm =", delta_cm)
        print("nu =", nu)
        print("lam =", lam_nm)
        print("Bki =", Bki.unit)
        print("rho =", rho.unit)
        raise
    
    R[k_index, i] += downward_rate
    R[i, k_index,] += upward_rate
 

# initial_level = min(5, n_levels -1) 
# atoms = np.full(N_atoms, initial_level, dtype=int)

# atoms = np.random.randint(0, n_levels, size=N_atoms)
atoms = np.zeros(N_atoms, dtype=int)
dt = 0.001 / np.max(R)
population_history = np.zeros((n_steps, n_levels))
time_history = np.arange(n_steps) * dt

print("R max =", np.max(R))
print("R min =", np.min(R))
print("Nonzero elements =", np.count_nonzero(R))    
print("Largest row sum =", np.max(np.sum(R, axis=1)))
print("Largest P =", 1 - np.exp(-np.max(np.sum(R, axis=1))*dt))
# print(np.amax(R))


def propagate_mutilevel (atoms, R, dt):
    for atom_index in range(len(atoms)):
        current_state = atoms[atom_index]
        rates = R[:,current_state]
        total_rate = np.sum(rates)
        if total_rate <= 0:
            continue
        P_transition = 1 - np.exp(-total_rate * dt)
        if np.random.random() > P_transition:
            continue
        probabilities = rates / total_rate
        new_state = np.random.choice(len(rates), p=probabilities)
        atoms[atom_index] = new_state
        
    return atoms




counts = np.bincount(atoms, minlength=n_levels)
population_history[0] = counts

for step in range(1, n_steps):
    atoms = propagate_mutilevel(atoms, R, dt)
    counts = np.bincount(atoms, minlength=n_levels)
    population_history[step] = counts
   
# E_list = []
# y_list = [] 

# for i,level in enumerate(levels):
#     N_i = population_history[-1,i]
#     if N_i <= 0:
#         continue 
#     g = 2 * level_J[level] + 1
#     E = level_energy[level] 
    
#     E_list.append(E)
#     y_list.append(np.log(N_i / g) )
    
# E_list = np.array(E_list)
# y_list = np.array(y_list) 

# slope, intercept = np.polyfit(E_list, y_list, 1)
# slope = slope * u.cm
# T = - (100* h * c) / (slope * k_B) 
# T = T.to(u.K)
# print("Estimated Temp: ", T)  

# final_populations = population_history[-1]

# for i, level in enumerate(levels[:20]):

#     g = 2 * level_J[level] + 1

#     print(
#         i,
#         level,
#         "E =", level_energy[level],
#         "N =", final_populations[i],
#         "N/g =", final_populations[i]/g
#     )
        
plt.figure()
n_plot = min(8, n_levels)
for level in range(n_plot):
    plt.plot(time_history, population_history[:, level], label=levels[level])
    
plt.xlabel('Time (s)')   
plt.ylabel('Number of Atoms')
plt.title('NIST Based Multi-Level Monte carlo Simulation For Sodium Atoms')
plt.legend()
plt.grid()
plt.show()


