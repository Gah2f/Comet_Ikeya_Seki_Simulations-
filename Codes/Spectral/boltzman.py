import matplotlib.pyplot as plt
import numpy as np
k = 1
mgh = 10

N = 3
T = [23,24,25,26,27,28]
E = np.array([0, mgh/2 , mgh])

def bolt(E,T):
    w = np.exp(-E / (k * T))
    return w / np.sum(w)

fig, axes = plt.subplots(1,len(T), figsize=(18,4))
for index, T in enumerate(T):
    prob = bolt(E,T)
    states = np.random.choice([0,1,2], size=N , p=prob)
    counts = [
        np.sum(states == 0),
        np.sum(states == 1),
        np.sum(states == 2)
    ]
     
    axes[index].bar(['Bottom', 'Middle', 'Top'], counts)
    axes[index].set_title(f"T={T}")
    axes[index].set_ylim(0, N)
    
plt.suptitle('Bolt dist')
plt.show()