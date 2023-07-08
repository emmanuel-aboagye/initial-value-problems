# -*- coding: utf-8 -*-
"""
Created on Wed May 24 01:35:22 2023

@author: aboag
"""

#%%
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from gekko import GEKKO
#%%


'''
Biochemical reactions example:

dS/dt = -Vmax * S / (Km + S)
dP/dt = Vmax * S / (Km + S)
where S is substrate
      P is product
      Vmax is maximum reaction rate
      Km is teh substrate concentration at which the reaction rate is half of Vmax

dy/dt ~ ( y(i) - y(i-1) )/ del_t = f(y(i-1), t(i-1))
y(i) = y(i-1) + del_t * f(y(i-1), t(i-1))

'''
#### Using the forward difference method: y(i) = y(i-1) + del_t * f(y(i-1), t(i-1))

# kinetic parameters
Vmax = 2. # mol/(L.s)
Km = 0.5  # mol/L

# ODE function
def ode_function(s, t):
    dsdt = -Vmax * s / (Km + s)
    return dsdt

# setup time discretization
n = 100   # number of time-steps
t = np.linspace(0, 2., n)
dt = t[1] - t[0]

# initilization of initial condiditions
solution = np.zeros(n)
solution[0] = 1.  # initial S in mol/L
for i in range(1,n):
    solution[i] = solution[i-1] + dt*ode_function(solution[i-1], t[i-1])

# plotting
plt.figure(figsize=(10,7), dpi=500)
plt.plot(t, solution)
plt.xlabel('time (dimensionless)')
plt.ylabel('concentration (dimensionless)')
plt.show()

#%%

#### Using the modified Euler method:
#### y(i) = y(i-1) + 0.5 * del_t * (f(y(i-1), t(i-1)) + f(y*(i), t(i)))
#### wehre y*(i) = y(i-1) + del_t * f(y(i-1), t(i-1))

n1 = 100 
t1 = np.linspace(0,2.,n1)
dt1 = t1[1] - t1[0]
solution1 = np.zeros(n1)
solution1[0] = 1.0
for i in range(1,n):
    slope = ode_function(solution1[i-1], t1[i-1])
    est = solution1[i-1] + dt1 * slope
    solution1[i] = solution1[i-1] + 0.5 * dt1 * (slope + ode_function(est, t1[i]))

# plotting
plt.figure(figsize=(10,7), dpi=500)
plt.plot(t, solution, label='forward_diff_method')
plt.plot(t1, solution1, label='modified_euler_method')
plt.xlabel('time (dimensionless)')
plt.ylabel('concentration (dimensionless)')
plt.legend(loc='best')
plt.show()

#%%

'''
Systems of Equations

it is easier to solve systems of differential equations using the scipy.integrate.odeint() librabry
the odeint() function takes in 
1) a function containing the ODEs
2) initial condition(s)
3) the time span for integration
    
'''
def systems_ode_function(c , t):
    s = c[0]  # substrate concentration
    p = c[1]  # product concentration
    dsdt = -Vmax * s / (Km + s)
    dpdt = Vmax * s / (Km + s)
    return np.array([dsdt, dpdt])

# initial condition and solution
c0 = np.array([1., 0.]) # initial condition for S, P in mol/L
t2 = np.linspace(0,2.,100)
solution2 = odeint(systems_ode_function, c0, t2)

# plotting
plt.figure(figsize=(10,7), dpi=500)
plt.plot(t2, solution2, label=["Substrate", "Product"])
plt.xlabel('time (dimensionless)')
plt.ylabel('concentration (dimensionless)')
plt.legend(loc='best')
plt.show()

#%%

'''
heat transfer problems
'''
h = 8.4     # W/m2-K
m_dot = 0.5 # kg/s
Cp = 1656   # J/kg-K 
Ri = 0.25   # m

def heat_transfer(T, z):
    Tw = (0.121 * z**3) - (3.4899 * z**2) + (23.3 * z) + 426.99
    dTdz = h * 2 * math.pi * Ri * (Tw - T) / (m_dot * Cp)
    return dTdz

T0 = 360 
Z = np.linspace(0., 10., 100)
solution4 = odeint(heat_transfer, T0, Z)

# plotting
plt.figure(figsize=(10,7), dpi=500)
plt.plot(Z, solution4)
plt.xlabel('distance (m)')
plt.ylabel('Tempearture, $^oC$')
plt.show()


#%%

k = 30  # W/m-k

def nuclear_reactor(x, r):
    S = x[0]
    QrR = x[1]
    dQrRdr = S * r
    dTdr = -(S * r) / (k * 2)
    return [dQrRdr, dTdr]

x00 = [9e7, 9e7/2] 
r = np.linspace(0, 0.05/2, 1000)
solution5 = odeint(nuclear_reactor, x00, r)

# plotting
plt.figure(figsize=(10,7), dpi=500)
plt.plot(r, solution5[:,1])
plt.xlabel('distance (m)')
plt.ylabel('Tempearture, $^oC$')
plt.show()

#%%

'''
Reaction problems
'''

def cstr_model(CA, t):
    k = 0.35   # 1/s
    Q1 = 10./60.    # L/min
    CA1 = 2.   # molA/L
    Q2 = 8./60.     # L/min
    CA2 = 5.   # molA/L
    V = 50.    # L
    dCAdt = (((CA1 * Q1) + (CA2 * Q2) - (Q1 + Q2) * CA) / V) - k * CA 
    return dCAdt

CA0 = 0.73   # molA/L
t4 = np.linspace(0, 50, 1000)
sol = odeint(cstr_model, CA0, t4)

# plotting
plt.figure(figsize=(10,7), dpi=500)
plt.plot(t4, sol)
plt.xlabel('time (s)')
plt.ylabel('Concentration, molA/L')
plt.show()

#%%
'''
Ebola Outbreak Prediction
'''

def ebola_outbreak1(X, t):
    # parameters
    C = 2/50000
    d = 0.5
    
    # variables
    S = X[0]
    I = X[1]
    # R = X[2]
    
    # equations
    dSdt = -C * S * I
    dIdt = C * S * I - I - d * I
    dRdt = I
    return [dSdt, dIdt, dRdt]
X0 = [50000, 2.0, 0]
t5 = np.linspace(0, 30, 1000)
sol_lowerC = odeint(ebola_outbreak1, X0, t5)

def ebola_outbreak2(X, t):
    # parameters
    C = 6/50000
    d = 0.5
    
    # variables
    S = X[0]
    I = X[1]
    # R = X[2]
    
    # equations
    dSdt = -C * S * I
    dIdt = C * S * I - I - d * I
    dRdt = I
    return [dSdt, dIdt, dRdt]
sol_middleC = odeint(ebola_outbreak2, X0, t5)

def ebola_outbreak3(X, t):
    # parameters
    C = 10/50000
    d = 0.5
    
    # variables
    S = X[0]
    I = X[1]
    # R = X[2]
    
    # equations
    dSdt = -C * S * I
    dIdt = C * S * I - I - d * I
    dRdt = I
    return [dSdt, dIdt, dRdt]
sol_upperC = odeint(ebola_outbreak3, X0, t5)
    
# plotting
plt.figure(figsize=(14,10), dpi=500)
plt.subplots_adjust(hspace=0.35)
plt.subplot(3,1,1)
plt.plot(t5, sol_lowerC[:,0], label='C = 2/50000')
plt.plot(t5, sol_middleC[:,0], label='C = 6/50000')
plt.plot(t5, sol_upperC[:,0], label='C = 10/50000')
plt.legend(loc='best')
plt.xlabel('time (days)')
plt.ylabel('No of people suceptible')   
    
plt.subplot(3,1,2)
plt.plot(t5, sol_lowerC[:,1], label='C = 2/50000')
plt.plot(t5, sol_middleC[:,1], label='C = 6/50000')
plt.plot(t5, sol_upperC[:,1], label='C = 10/50000')
plt.legend(loc='best')
plt.xlabel('time (days)')
plt.ylabel('No of people infected')

plt.subplot(3,1,3)
plt.plot(t5, sol_lowerC[:,2], label='C = 2/50000')
plt.plot(t5, sol_middleC[:,2], label='C = 6/50000')
plt.plot(t5, sol_upperC[:,2], label='C = 10/50000')
plt.legend(loc='best')
plt.xlabel('time (days)')
plt.ylabel('No of people recovered')
plt.show()

#%%
'''
Minimum distance between earth and mars
'''
G = 1.983979e-29    # Universal gravitational constant AU^3 / year^2-kg
M = 2.0e30          # Mass of the sun

def distance_for_earth(x, t):
    
    # variable declaration
    X = x[0]
    Y = x[1]
    Vx  = x[2]
    Vy  = x[3]
    
    # equations
    dXdt = Vx
    dYdt = Vy
    dVxdt = - (G * M * X) / (np.sqrt(X**2 + Y**2))**3
    dVydt = - (G * M * Y) / (np.sqrt(X**2 + Y**2))**3
    return [dXdt, dYdt, dVxdt, dVydt]

initial_conditions_earth = [0.44503, 0.88106, -5.71113, 2.80924]
sim_time = np.linspace(0,10, 10000)
sol_earth = odeint(distance_for_earth, initial_conditions_earth, sim_time)

def distance_for_mars(x, t):
    
    # variable declaration
    X = x[0]
    Y = x[1]
    Vx  = x[2]
    Vy  = x[3]
    
    # equations
    dXdt = Vx
    dYdt = Vy
    dVxdt = - (G * M * X) / (np.sqrt(X**2 + Y**2))**3
    dVydt = - (G * M * Y) / (np.sqrt(X**2 + Y**2))**3
    return [dXdt, dYdt, dVxdt, dVydt]

initial_conditions_mars = [-0.81449,  1.41483, -4.23729, -2.11473]
sol_mars = odeint(distance_for_mars, initial_conditions_mars, sim_time)

# plotting
plt.figure(figsize=(15,11), dpi=500)
plt.subplots_adjust(hspace=0.35)
plt.subplot(4,1,1)
plt.plot(sim_time, sol_earth[:,0], label='Earth')
plt.plot(sim_time, sol_mars[:,0], label='Mars')
plt.legend(loc='best')
plt.xlabel('Time (Years)')
plt.ylabel('X-Distance (AU)')   

plt.subplot(4,1,2)
plt.plot(sim_time, sol_earth[:,1], label='Earth')
plt.plot(sim_time, sol_mars[:,1], label='Mars')
plt.legend(loc='best')
plt.xlabel('Time (Years)')
plt.ylabel('Y-Distance (AU)')

plt.subplot(4,1,3)
plt.plot(sim_time, sol_earth[:,2], label='Earth')
plt.plot(sim_time, sol_mars[:,2], label='Mars')
plt.legend(loc='best')
plt.xlabel('Time (Years)')
plt.ylabel('Vx-Velocity (AU/year)')

plt.subplot(4,1,4)
plt.plot(sim_time, sol_earth[:,3], label='Earth')
plt.plot(sim_time, sol_mars[:,3], label='Mars')
plt.legend(loc='best')
plt.xlabel('Time (Years)')
plt.ylabel('Vy-Velocity (AU/year)')
plt.show()

#%%

plt.figure(figsize=(10,4), dpi=500)
plt.subplots_adjust(wspace=0.35)
plt.subplot(1,2,1)
plt.plot(sol_earth[:,0], sol_earth[:,1], label='Earth')
plt.plot(sol_mars[:,0], sol_mars[:,1], label='Mars')
plt.legend(loc=2)
plt.xlabel('X-Distance (AU)')
plt.ylabel('Y-Distance (AU)')
plt.text(-0.2,0, r'Sun(0,0)')

plt.subplot(1,2,2)
plt.plot(sol_earth[:,2], sol_earth[:,3], label='Earth')
plt.plot(sol_mars[:,2], sol_mars[:,3], label='Mars')
plt.legend(loc=2)
plt.xlabel('Vx-Velocity (AU/year)')
plt.ylabel('Vy-Velocity (AU/year)')
plt.text(-0.2,0, r'Sun(0,0)')
plt.show()

#%%
distance_btn_mars_and_earth = [None] * len(sim_time)

for i in range(len(sim_time)):
    distance_btn_mars_and_earth[i] = np.sqrt((sol_earth[:,0][i] - sol_mars[:,0][i])**2 + (sol_earth[:,1][i] - sol_mars[:,1][i])**2)

print("minimum distance (AU) is : ", np.array(distance_btn_mars_and_earth).min())
#%%

'''
Halocarbons: Chlorofluorocarbons (CFCs) in bioshpere
'''

Ls = 5  # years
Lt = 1000 # years
tau = 3 # years
f  = 0.15/0.18

def halocarbons(x, t):
    
    # variables
    Ht = x[0]
    Hs = x[1]
    C = x[2]
    
    # equations
    dHtdt = -(Ht/Lt) - (Ht*f/tau) + (Hs/tau)
    dHsdt = -(Hs/Ls) - (Hs/tau) + (Ht*f/tau)
    dCdt = -(C/tau) + (Hs/Ls)
    return np.array([dHtdt, dHsdt, dCdt])

halo_initial_cond = [1., 0., 0.]
t_sim = np.linspace(0,100,10000)
sol_halo = odeint(halocarbons, halo_initial_cond, t_sim)

plt.figure(figsize=(10,7), dpi=500)
plt.plot(t_sim, sol_halo, label=['Troposphere_HC','Stratosphere_HC','Stratosphere_free_Cl'])
plt.legend(loc='best')
plt.xlabel('Time (Years)')
plt.ylabel('Species Concentration (kg)')
plt.show()

#%% 

'''
Reaction Engineering
'''
def reactor_model(x, V):
  # parameters
  Q = 0.5 # L/min
  k1 = 0.05 # 1/min
  k2 = 0.0025 # 1/min
  
  # variables 
  CA = x[0]
  CB = x[1]
  CC = x[2]
  
  # differential equations
  dCAdV = -k1 * CA / Q
  dCBdV = (k1 * CA - k2 * CB) / Q
  dCCdV = k2 * CB / Q
  return np.array([dCAdV, dCBdV, dCCdV])
  
initial_cond = [10., 0., 0.]
V_sim = np.linspace(0, 100, 10000)
solution6 = odeint(reactor_model, initial_cond, V_sim)

plt.figure(figsize=(10,7), dpi=500)
plt.plot(V_sim, solution6, label=['CA','CB','CC'])
plt.legend(loc='best')
plt.xlabel('Volume, L')
plt.ylabel('Concentration, (mol/L)')
plt.show()

#%%

# parameters
Ka = 0.05
Kb = 0.15
Pao = 12
eps = 1
A = 7.6
R = 0.082
T = 400 + 273.15
rho = 80
kprime = 0.0014
# D = 1.5
Uo = 2.5
Kc = 0.1

def sttr_reactor_model(x, z):
  
  # State variables
  X = x[0]
  
  # Algebraic equations
  U = Uo * (1 + eps*X)
  Pa = Pao * (1 - X) / (1 + eps*X)
  Pb = Pao * X / (1 + eps*X)
  # vo = Uo * math.pi * D * D / 4
  Cao = Pao/R/T
  # KCa = Ka * R * T
  Pc = Pb
  a = 1 / (1 + A * (z/U)**0.5)
  raprime = a * (-kprime * Pa / (1+ Ka * Pa + Kb * Pb + Kc * Pc))
  ra = rho * raprime
  
  
  # differential equations
  dXdz = -ra / U / Cao
  
  return np.array([dXdz]) 
  
sttr_initial_cond = [0.]
sttr_z_sim = np.linspace(0, 10., 10000)
solution7 = odeint(sttr_reactor_model, sttr_initial_cond, sttr_z_sim)

# Create a dataframe now that you know your X values
df = pd.DataFrame(sttr_z_sim, columns=['distance'])
df['conversion_X'] = solution7
df['U'] = Uo * (1 + eps*df['conversion_X'])
df['Pa'] = Pao * (1 - df['conversion_X']) / (1 + eps * df['conversion_X'])
df['Pb'] = Pao * df['conversion_X'] / (1 + eps * df['conversion_X'])
df['a'] = 1 / (1 + A * (df['distance']/df['U'])**0.5)
df['Pc'] = df['Pb']
df['raprime'] = df['a'] * (-kprime * df['Pa'] / (1+ Ka * df['Pa'] + Kb * df['Pb'] + Kc * df['Pc'])) 
df['ra'] = rho * df['raprime']

plt.figure(figsize=(14,11), dpi=500)
#plt.subplots_adjust(hspace=0.35)
plt.subplot(2,1,1)
plt.plot(df['distance'], df['conversion_X'], label='Conversion')
plt.plot(df['distance'], df['a'], label='catalyst deactivation')
plt.legend(loc='best')
plt.xlabel('distance, z (m)')
plt.ylabel('fraction (-)')

plt.subplot(2,1,2)
plt.plot(df['distance'], df['Pa'], label='Pressure A')
plt.plot(df['distance'], df['Pb'], label='Pressure B')
plt.legend(loc='best')
plt.xlabel('distance, z (m)')
plt.ylabel('Pressure, atm')
plt.show()
