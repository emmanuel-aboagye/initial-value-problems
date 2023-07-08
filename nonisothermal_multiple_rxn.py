# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 02:53:52 2023

@author: aboag
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

#%% Non-isothermal multi-reaction problem

# define constants
Cao = 4 
Vo  = 240 

# define function 
def reaction_model(X, t):
    
    # define state variables
    Ca = X[0]
    Cb = X[1]
    Cc = X[2]
    T  = X[3]
    
    # define algebraic equations
    V = 100 + (Vo * t)
    k1a = 1.25 * np.exp((9500/1.987) * (1/320 - 1/T))
    k2b = 0.08 * np.exp((7000/1.987) * (1/290 - 1/T))
    ra = -k1a * Ca
    rb = k1a * Ca / 2 - k2b * Cb
    rc = 3 * k2b * Cb
    
    # define differential equations
    dCadt = ra + (Cao - Ca) * Vo / V
    dCbdt = rb - Cb * Vo / V
    dCcdt = rc - Cc * Vo / V
    dTdt  = (35000 * (298 -T) - Cao * Vo * 30 * (T - 305) + ( (-6500) * (-k1a * Ca) + (8000) * (-k2b * Cb)) \
        * V) / ((Ca * 30 + Cb * 60 + Cc * 20) * V + 100 * 35)
    return [dCadt, dCbdt, dCcdt, dTdt]

# Initial conditions
initial_guess = np.array([1., 0., 0., 290.0])

# Time steps 
time = np.linspace(0,1.5,100)

# Solve ODE
solution = odeint(reaction_model, initial_guess, time)

#%%
plt.figure(figsize=(14,10))
plt.subplot(2,1,1)
plt.subplots_adjust(hspace=0.25)
plt.plot(time, solution[:,0], label='Ca')
plt.plot(time, solution[:,1], label='Cb')
plt.plot(time, solution[:,2], label='Cc')
plt.xlim(xmin=0)
plt.ylabel("Concentration, $mol/dm^3$")
plt.xlabel('time, hr')
plt.legend(loc='best')

plt.subplot(2,1,2)
plt.plot(time, solution[:,3], label='T')
plt.xlim(xmin=0)
plt.xlabel('time, hr')
plt.ylabel('Temperature, K')
plt.legend(loc='best')
plt.show()

