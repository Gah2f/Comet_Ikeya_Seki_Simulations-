import numpy as np
from sbpy.calib import Sun
from astropy import units as u

sun = Sun.from_default()
x = np.arange(300,1100,2) * u.nm 
# print(x)
dx = 0.000000000000000000000000001 * u.nm 
    
def Area(sun,x,dx):
    values = []
    
    for i in range(len(x)):
        i_v = x[i] 
        values.append(sun(i_v) * dx)
    t = sum(values)
    print(t.to(u.W / u.m**2))           
    
Area(sun,x,dx)
