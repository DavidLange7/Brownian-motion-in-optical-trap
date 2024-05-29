#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 13:31:33 2024

@author: david
"""

import numpy as np
import scipy
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy import optimize
import pylab as plb
from statsmodels.tsa.stattools import acf

plb.rcParams['font.size'] = 25
plt.rcParams["figure.figsize"] = (18,12)

#%%
def gaussian(x, mu, sigma):
    '''
    returns a gaussian distribution with mean mu and variance sigma**2
    '''
    
    return(1/(sigma*np.sqrt(2*np.pi))*np.exp(-1/2*((x-mu)/sigma)**2))

def x_new_inertia(x_0, x_1, w, delta_t, m, R, eta, k_B, T):
    '''
    free diffusion/random walk with initial position x_old and time step delta_t
    '''
    gamma = 6*np.pi*eta*R
    
    res_1 = ((2 + delta_t*(gamma/m))/(1 + delta_t*(gamma/m)))*x_1
    res_2 = (1/(1 + delta_t*(gamma/m)))*x_0
    res_3 = (np.sqrt(2*k_B*T*gamma)/(m+delta_t*gamma))*(delta_t**(3/2))*w
    
    return(res_1 - res_2 + res_3)

def x_new_noinertia(x_old, w, delta_t, D):
    '''
    free diffusion/random walk with initial position x_old and time step delta_t
    '''
    return(x_old + np.sqrt(2*D*delta_t)*w)

def custom_gaussian_random_numbers(size, variance, mean):
    '''
    returns gaussianly distributed random numbers with mean and variance and size
    '''
    rand_n = np.random.randn(size)
    
    W = rand_n * np.sqrt(variance) + mean
    
    return W

def velocity_correlation(data, delta_t):
    '''
    computes the autocorrelation (time average) of two pairs of positions
    '''
    
    v = [(data[i+1] - data[i])/delta_t for i in range(len(data) - 1)]
    
    autocorr = acf(v, nlags=None, fft = False)
    #result = np.correlate(v, v, mode='same')
    
    result = np.correlate(v, v, mode='full')
    return autocorr

def manual_acf(data, delta_t):
    
    x = np.array([(data[i+1] - data[i])/delta_t for i in range(len(data) - 1)])
    
    n = len(x)
    mean = np.mean(x)
    var = np.var(x)
    nlags = 347
    acf_values = np.zeros(nlags + 1)
    
    for lag in range(nlags + 1):
        cov = np.sum((x[:n-lag] - mean) * (x[lag:] - mean))
        acf_values[lag] = cov / (var * (n - lag))
    
    return acf_values

def manual_msd(data):
        
    nlags = 600
    
    n = len(data)
            
    msd_values = [np.mean((data[:n-lag] - data[lag:]) ** 2) for lag in range(nlags)] 
    
    return msd_values

#fig, axs = plt.subplots(2, 3, figsize=(18, 12))
#axs = axs.flatten()


tau = 0.6*10**(-6) #seconds
delta_t = 10*10**(-9) #seconds
#m = 11*10**(-12) #grams
T = 300 #kelvin
R = 1*10**(-6) #meters
m = 4/3*np.pi*R**3*2.6*10**3

eta = 0.001 #Ns/m²
k_B = 1.380649*10**(-23) #J/K
gamma = 6*np.pi*eta*R
D = k_B*T/gamma

tau = m/gamma

t_stop = 100000

upper_b = int((100*tau)/delta_t) + 1

#upper_b = t_stop

#%%
traj_inertia = np.zeros([upper_b])
traj_noinertia = np.zeros([upper_b])
vel_ni_tot = np.zeros(348)
vel_i_tot = np.zeros(348)

msd_ni_tot = np.zeros(600)
msd_i_tot = np.zeros(600)

n_tot = 4000

for k in range(n_tot):
    traj_inertia = np.zeros([upper_b])
    traj_noinertia = np.zeros([upper_b])
    for i in range(2, upper_b):
        w = np.random.normal()
        
        tmp_1 = x_new_inertia(traj_inertia[i-2], traj_inertia[i-1], w, delta_t, m, R, eta, k_B, T)
        tmp_2 = x_new_noinertia(traj_noinertia[i-1], w, delta_t, D)
        
        traj_inertia[i] = tmp_1
        traj_noinertia[i] = tmp_2
        
    vel_ni = manual_acf(traj_noinertia, delta_t)
    vel_i =  manual_acf(traj_inertia, delta_t)
    
    msd_ni = manual_msd(traj_noinertia)
    msd_i = manual_msd(traj_inertia)

    
    msd_ni_tot += msd_ni
    msd_i_tot += msd_i
    vel_ni_tot += vel_ni
    vel_i_tot += vel_i
#%%

fig, axs = plt.subplots(2, 2, figsize=(18, 12))
axs = axs.flatten()

space = np.linspace(0, 100, upper_b)
#indx = int(np.argwhere(np.isclose(space, 100, atol = 1e-2) == True))
indx = -1

axs[1].plot(space[:indx], traj_noinertia[:indx]*10**(9), label = 'no inertia', color = 'black')
axs[1].plot(space[:indx], traj_inertia[:indx]*10**(9), label = 'inertia', color = 'red', linewidth = 3)
axs[1].set_ylabel('x [nm]')
axs[1].set_xlabel(r'$\mathrm{t/\tau}$')
axs[1].text(2, 6.6, '(b)', fontsize=25)
axs[1].legend(loc = 'lower right', prop={'size': 20})
axs[1].set_ylim([-8, 8])
axs[1].set_yticks([-8,-4,0,4, 8])
axs[1].set_xlim([0,100])




indx = int(np.argwhere(np.isclose(space, 1, atol = 1e-2) == True)) + 10
axs[0].plot(space[:indx], traj_noinertia[:indx]*10**(9), label = 'no inertia', color = 'black')
axs[0].plot(space[:indx], traj_inertia[:indx]*10**(9), label = 'inertia', color = 'red', linewidth = 2.5)
axs[0].set_ylabel('x [nm]')
axs[0].set_xlabel(r'$\mathrm{t/\tau}$')
axs[0].text(0.02, 0.83, '(a)', fontsize=25)
axs[0].legend(loc = 'lower right', prop={'size': 20})
axs[0].set_ylim([-1, 1])
axs[0].set_yticks([-1,-0.5,0,0.5,1])
axs[0].set_xlim([0,1])


vel_ni_tot[vel_ni_tot<0] = 0
vel_i_tot[vel_i_tot<0] = 0

axs[2].set_xlabel(r'$\mathrm{t/\tau}$')
axs[2].set_ylabel(r'$\mathrm{C_v(t)}$')
axs[2].text(5.4, 0.96, '(c)', fontsize=25)
axs[2].plot(space[:348], (vel_ni_tot/n_tot), label = 'no inertia', color = 'black')
axs[2].plot(space[:348], (vel_i_tot/n_tot), label = 'inertia', color = 'red', linewidth = 2.5)
axs[2].set_xlim([-0.1,6])


axs[3].set_xlabel(r'$\mathrm{t/\tau}$')
axs[3].set_ylabel(r'$\mathrm{\langle x(t)^2\rangle \: [nm^2]}$')
axs[3].text(0.015, 1.65, '(d)', fontsize=25)
axs[3].loglog(space[:600], msd_ni_tot/n_tot*(10**9)**2, label = 'no inertia', color = 'black')
axs[3].loglog(space[:600], msd_i_tot/n_tot*(10**9)**2, label = 'inertia', color = 'red', linewidth = 2.5)
axs[3].set_xlim([0,10])


fig.tight_layout()
fig.savefig('fig2_browndiff.pdf', dpi=100, bbox_inches='tight')
