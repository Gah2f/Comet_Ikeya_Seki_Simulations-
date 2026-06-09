import numpy as np
import matplotlib.pyplot as plt

levels = ["3S", "3P", "4S", "3D"]
n_levels = len(levels)

R = np.array([
    [0.0, 5e6, 0.0 , 0.0],
    [6e7, 0.0, 2e7, 1e7],
    [0.0, 3e7, 0.0, 2e6],
    [0.0, 2e7, 0.0, 0.0]
])

N_atoms = 10000
dt = 1e-9
n_steps = 5000

atoms = np.zeros(N_atoms, dtype=int)

population_history = np.zeros((n_steps, n_levels))
time_history = np.arange(n_steps) * dt

def propagate_multilevel (atoms, R, dt):
    for atom_index in range(len(atoms)):
        current_state = atoms[atom_index]
        rates = R[current_state]
        total_rate = np.sum(rates)
        if total_rate == 0:
            continue
        
        P_transition = 1 - np.exp(-total_rate * dt)
        random_number = np.random.random()
        
        if random_number > P_transition:
            continue
        
        probabilities = rates / total_rate
        
        new_state = np.random.choice(np.arange(n_levels), p=probabilities)
        atoms[atom_index] = new_state
        return atoms
    

for step in range(n_steps):
    atoms = propagate_multilevel(atoms, R, dt)
    for level in range(n_levels):
        population_history[step, level] = np.sum(atoms == level)
        

plt.figure()

for level in range(n_levels):
    plt.plot(time_history, population_history[:, level], label=levels[level])
plt.xlabel("Time (s)")
plt.ylabel("Number of Atoms")
plt.title("Multi-Level Monte Carlo Simulation of Na Atom Populations")
plt.legend()
plt.grid()
plt.show()