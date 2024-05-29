#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 18:56:28 2024

@author: david
"""

import numpy as np
import scipy
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy import optimize
import pylab as plb
from statsmodels.tsa.stattools import acf

plb.rcParams['font.size'] = 40
plt.rcParams["figure.figsize"] = (18,12)
#%%
def r_new_constantf(x_0, k_x, gamma, D, delta_t):
    '''
    diffusion/random walk with initial position x_old and time step delta_t
    here, we add a constant force 
    '''

    w = np.random.normal()

    force_custom = 1/gamma*200*10**(-15)*delta_t

    
    x_1 = x_0 - 1/gamma*k_x*x_0*delta_t + np.sqrt(2*D*delta_t)*w + force_custom
    
    return(x_1)

def r_new_oscill(x_0, k_x, gamma, D, delta_t, t, a):
    '''
    diffusion/random walk with initial position x_old and time step delta_t
    here, we add time dependent force
    '''

    w = np.random.normal()
    
    phi = gamma/(1*10**(-6))
    
    f = (1/phi)*10**(6)

    force_custom = -1/gamma*k_x*delta_t*(x_0 - a*np.sin(2*np.pi*f*t))

    
    x_1 = x_0 + np.sqrt(2*D*delta_t)*w + force_custom
    
    return(x_1)

def r_new_stochresonance(x_0, k_x, gamma, D, delta_t, t, c):
    '''
    diffusion/random walk with initial position x_old and time step delta_t
    here, we add time dependent force
    '''

    w = np.random.normal()
    a = 1.0*10**7 #N/m³
    b = 1*10**(-6) #N/m
    
    phi = gamma/(1*10**(-6))
    
    f = (1/phi)
    
    force_custom = 1/gamma*(b*x_0*delta_t - a*x_0**3*delta_t) + c*np.sin(2*np.pi*f*t)

    x_1 = x_0 + np.sqrt(2*D*delta_t)*w + force_custom
    
    return(x_1)

def r_new_doublewell(x_0, gamma, D, delta_t):
    '''
    diffusion/random walk with initial position x_old and time step delta_t
    here, we add a constant force 
    '''

    w = np.random.normal()
    a = 1.0*10**7 #N/m³
    b = 1*10**(-6) #N/m
    
    force_custom = 1/gamma*(b*x_0*delta_t - a*x_0**3*delta_t)

    x_1 = x_0 + np.sqrt(2*D*delta_t)*w + force_custom
    
    return(x_1)

def r_new_rotatingf(r_0, k, gamma, D, delta_t):
    '''
    free diffusion/random walk with initial position x_old and time step delta_t
    here, we add a constant rotating force
    '''
    
    w = np.array([np.random.normal() for i in range(2)])
    
    x_0 = r_0[0]
    y_0 = r_0[1]
    
    omega = 300
    
    force_xpart = 1/gamma*delta_t*(k*x_0 + gamma*omega*y_0)
    force_ypart = 1/gamma*delta_t*(k*y_0 - gamma*omega*x_0)

    
    x_1 = x_0 + np.sqrt(2*D*delta_t)*w[0] + force_xpart
    y_1 = y_0 + np.sqrt(2*D*delta_t)*w[1] + force_ypart
    
    
    return([x_1, y_1])

#%%
def manual_acf(data, delta_t, nlags):
    
    x = data
    
    n = len(x)
    mean = np.mean(x)
    var = np.var(x)
    acf_values = np.zeros(nlags + 1)
    
    for lag in range(nlags + 1):
        cov = np.sum((x[:n-lag] - mean) * (x[lag:] - mean))
        acf_values[lag] = cov / (var * (n - lag))
    
    return acf_values

def manual_ccf(datax, datay, delta_t, nlags):
    '''
    cross-correlation function between datax and datay
    '''
    
    x = datax
    y = datay
    
    n = len(x)
    
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    var_x = np.var(x)
    var_y = np.var(y)
    
    ccf_values = np.zeros(nlags + 1)
    
    for lag in range(nlags + 1):
        cov = np.sum((x[:n-lag] - mean_x) * (y[lag:] - mean_y))
        ccf_values[lag] = cov / (np.sqrt(var_x * var_y) * (n - lag))
    
    return ccf_values

def manual_msd(data, nlags):
        
    
    n = len(data)
            
    msd_values = [np.mean((data[:n-lag] - data[lag:]) ** 2) for lag in range(nlags)]
    
    return msd_values

tau = 0.6*10**(-6) #seconds

T = 300 #kelvin
R = 1*10**(-6) #meters
eta = 0.001 #Ns/m²
k_B = 1.380649*10**(-23) #J/K
gamma = 6*np.pi*eta*R
D = k_B*T/gamma

#%%
x_0 = 0
y_0 = 0
k = 1 * 10**(-6) #N/m
delta_t = 10**(-4) #seconds

steps = 10000

n_tot = 10
nlags = 4000

ccf_tot = np.zeros(nlags)
acf_tot = np.zeros(nlags)

for ind in range(n_tot):
    if ind % 1 == 0:
        print(ind)
        
    r_0 = [x_0, y_0]

    traj_x = [r_0[0]]
    traj_y = [r_0[1]]
    for j in range(1, steps):
        
        tmp = r_new_rotatingf(r_0, k, gamma, D, delta_t)
        traj_x.append(tmp[0])
        traj_y.append(tmp[1])
    
        
        r_0 = [traj_x[j], traj_y[j]]
    
    acf_tot += manual_acf(np.array(traj_x)*10**(9), delta_t, nlags-1 )
    ccf_tot += manual_ccf(np.array(traj_x)*10**(9), np.array(traj_y)*10**(9),  delta_t, nlags-1)

traj_x = np.array(traj_x)*10**9
traj_y = np.array(traj_y)*10**9

plt.plot(np.linspace(0, nlags*delta_t, nlags)*10**(3), ccf_tot/n_tot, label = r'$\mathrm{C_{xy}(t)}$',  linewidth=10)
plt.plot(np.linspace(0, nlags*delta_t, nlags)*10**(3), acf_tot/n_tot, label = r'$\mathrm{C_{x}(t)}$',  linewdith=10)
plt.xlabel('t [ms]')
plt.ylabel('Intensity')
plt.legend()
plt.savefig('fig6_b1.pdf', dpi=500, bbox_inches='tight')


fig, ax = plt.subplots()
ax.plot(traj_x[:500], traj_y[:500], 'r', markersize=5.5)
ax.set_aspect('equal', adjustable='box')
ax.set_ylabel('y [nm]')
ax.set_xlabel('x [nm]')
fig.savefig('fig6_b2.pdf', dpi=500, bbox_inches='tight')


#%%
x_0 = 0
k_x = 1.0 * 10**(-6) #N/m

steps = int(1e+5)

traj_x = [x_0]

for j in range(1, steps):
    
    tmp = r_new_constantf(x_0, k_x, gamma, D, delta_t)
    traj_x.append(tmp)

    
    x_0 = traj_x[j]
    
traj_x = np.array(traj_x)*10**9

#%%

fig, ax = plt.subplots()

ny, binsy, patchesy = ax.hist((traj_x_st), bins = 100, histtype='bar',density=True, stacked=True, alpha = 0.8, label = 't = 0')
nx, binsx, patchesx = ax.hist((traj_x_st_f), bins = 100, histtype='bar',density=True, stacked=True, alpha = 0.8, label = 't > 0')
ax.legend()
ax.set_xlabel('x [nm]')
ax.set_ylabel('density')

ticks = [0, 0.00617]
ax.set_yticks(ticks)

dic = {0: "0", 
       0.00617 : "1.0"}
labels = [ticks[i] if t not in dic.keys() else dic[t] for i,t in enumerate(ticks)]
ax.set_yticklabels(labels)

fig.savefig('fig6_a.pdf', dpi=500, bbox_inches='tight')

#%%
x_0 = 0

delta_t = 10**(-3) #seconds

steps = 300000
traj_x = [x_0]

for j in range(1, steps):
    
    tmp = r_new_doublewell(x_0, gamma, D, delta_t)
    traj_x.append(tmp)

    
    x_0 = traj_x[j]
    
traj_x = np.array(traj_x)*10**9
space = np.linspace(0, steps*delta_t, steps)

plt.plot(space, traj_x)
plt.xlabel('t [s]')
plt.ylabel('x [nm]')
plt.xlim([0, steps*delta_t])
plt.savefig('fig6_c.pdf', dpi=500, bbox_inches='tight')

#%%


delta_t = 10**(-5) #seconds
k = 7 * 10**(-6) #N/m

steps = 100000

var_s = []
kspace = np.linspace(3.5,20,30)*10**(-6) #N/m
for i in range(len(kspace)):
    
    k = kspace[i]
    x_0 = 0
    traj_x = [x_0]

    
    for j in range(1, steps):

    
        tmp = r_new_oscill(x_0, k, gamma, D, delta_t, j*delta_t, 100*10**(-9))
        traj_x.append(tmp)
    
        
        x_0 = traj_x[j]
        
    traj_x = np.array(traj_x)*10**9
    var_s.append(np.var(traj_x))
    
plt.plot(kspace*10**(6), var_s, linewidth=3)
plt.plot(kspace*10**(6), var_s, 'ro')
plt.xlim([0,20])
plt.xlabel('k [fN/nm]')
plt.ylabel(r'$\mathrm{\sigma^2\: [nm^2]}$')
plt.savefig('fig6_e.pdf', dpi=500, bbox_inches='tight')



