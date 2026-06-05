import numpy as np
import matplotlib.pyplot as plt

N_atoms = 10000
atoms = np.zeros(N_atoms, dtype = int)

R_up = 1e6
A21 = 5e6

dt = 1e-9

P_up = 1 - np.exp(-R_up * dt)
P_down = 1 - np.exp(-A21 * dt)

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