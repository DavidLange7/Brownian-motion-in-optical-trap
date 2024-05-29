#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 26 10:22:38 2024

@author: david
"""

import numpy as np
import scipy
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy import optimize
import pylab as plb

plb.rcParams['font.size'] = 30
plt.rcParams["figure.figsize"] = (18,12)
#%%
def gaussian(x, mu, sigma):
    '''
    returns a gaussian distribution with mean mu and variance sigma**2
    '''
    
    return(1/(sigma*np.sqrt(2*np.pi))*np.exp(-1/2*((x-mu)/sigma)**2))

def x_new(x_old, delta_t):
    '''
    free diffusion/random walk with initial position x_old and time step delta_t
    '''
    w = np.random.normal()
    return([w/np.sqrt(delta_t), x_old + np.sqrt(delta_t)*w])

def custom_gaussian_random_numbers(size, variance, mean):
    '''
    returns gaussianly distributed random numbers with mean and variance and size
    '''
    rand_n = np.random.randn(size)
    
    W = rand_n * np.sqrt(variance) + mean
    
    return W

fig, axs = plt.subplots(2, 3, figsize=(18, 12))
axs = axs.flatten()

delta_t = 1

upper_b = int(30/delta_t) + 1

sol_arr = np.zeros([10000, upper_b])
w_arr = np.zeros([10000, upper_b])

stds = []

for j in range(10000):
    for i in range(1, upper_b):
        tmp = x_new(sol_arr[j][i-1], delta_t)
        sol_arr[j, i] = tmp[1]
        w_arr[j,i] = tmp[0]
    stds.append(np.std(sol_arr[j]))

space = np.linspace(0, 30, upper_b)
region = []
sol_arr_t = np.transpose(sol_arr)
for k in range(upper_b):
    std = np.std(sol_arr_t[k])
    region.append(std)

tot = np.ndarray.flatten(np.stack((np.array(region[::-1]), -np.array(region))))
tot_sp = np.ndarray.flatten(np.stack((np.array(space[::-1]), np.array(space))))

axs[3].fill_between(tot_sp, tot, alpha=1, color='silver', label=r'$\mu \pm \sigma$')
axs[3].plot(space, sol_arr[1000], linewidth=2, color = 'blue')
axs[3].set_xlim([0,30])
axs[3].set_ylim([-8.2,8.2])
axs[3].set_yticks([-8, -4, 0, 4, 8])
axs[3].set_xticks([0, 10, 20, 30])
axs[3].set_xlabel('t')
axs[3].set_ylabel(r'x$_i$')
axs[3].text(0.5, -7.7, r'$\Delta$t = 1', fontsize=20)
axs[3].text(0.5, 6.5, '(d)', fontsize=30)

axs[0].set_xlim([0,30])
axs[0].set_ylim([-8.2,8.2])
axs[0].set_yticks([-8, -4, 0, 4, 8])
axs[0].set_ylabel(r'W$_i$')
axs[0].plot(space, w_arr[1000], 'b.')
axs[0].text(0.5, -7.7, r'$\Delta$t = 1', fontsize=20)
axs[0].text(0.5, 6.5, '(a)', fontsize=30)

delta_t = 0.5

upper_b = int(30/delta_t) + 1

sol_arr = np.zeros([10000, upper_b])
w_arr = np.zeros([10000, upper_b])

stds = []

for j in range(10000):
    for i in range(1, upper_b):
        tmp = x_new(sol_arr[j][i-1], delta_t)
        sol_arr[j, i] = tmp[1]
        w_arr[j,i] = tmp[0]
    stds.append(np.std(sol_arr[j]))

space = np.linspace(0, 30, upper_b)
region = []
sol_arr_t = np.transpose(sol_arr)
for k in range(upper_b):
    std = np.std(sol_arr_t[k])
    region.append(std)

tot = np.ndarray.flatten(np.stack((np.array(region[::-1]), -np.array(region))))
tot_sp = np.ndarray.flatten(np.stack((np.array(space[::-1]), np.array(space))))

axs[4].fill_between(tot_sp, tot, alpha=1, color='silver', label=r'$\mu \pm \sigma$')
axs[4].plot(space, sol_arr[1000], linewidth=2, color = 'blue')
axs[4].set_xlim([0,30])
axs[4].set_ylim([-8.2,8.2])
axs[4].set_yticks([])
axs[3].set_xticks([0, 10, 20, 30])
axs[4].set_xlabel('t')
axs[4].text(0.5, -7.7, r'$\Delta$t = 0.5', fontsize=20)
axs[4].text(0.5, 6.5, '(e)', fontsize=30)

axs[1].set_xlim([0,30])
axs[1].set_ylim([-8.2,8.2])
axs[1].set_yticks([])
axs[1].set_xticks([])
axs[1].plot(space, w_arr[1000], 'b.')
axs[1].text(0.5, -7.7, r'$\Delta$t = 0.5', fontsize=20)
axs[1].text(0.5, 6.5, '(b)', fontsize=30)

delta_t = 0.1

upper_b = int(30/delta_t) + 1

sol_arr = np.zeros([10000, upper_b])
w_arr = np.zeros([10000, upper_b])

stds = []

for j in range(10000):
    for i in range(1, upper_b):
        tmp = x_new(sol_arr[j][i-1], delta_t)
        sol_arr[j, i] = tmp[1]
        w_arr[j,i] = tmp[0]
    stds.append(np.std(sol_arr[j]))

space = np.linspace(0, 30, upper_b)
region = []
sol_arr_t = np.transpose(sol_arr)
for k in range(upper_b):
    std = np.std(sol_arr_t[k])
    region.append(std)

tot = np.ndarray.flatten(np.stack((np.array(region[::-1]), -np.array(region))))
tot_sp = np.ndarray.flatten(np.stack((np.array(space[::-1]), np.array(space))))

axs[5].fill_between(tot_sp, tot, alpha=1, color='silver', label=r'$\mu \pm \sigma$')
axs[5].plot(space, sol_arr[1000], linewidth=2, color = 'blue')
axs[5].set_xlim([0,30])
axs[5].set_ylim([-8.2,8.2])
axs[5].set_yticks([])
axs[5].set_xlabel('t')
axs[3].set_xticks([0, 10, 20, 30])
axs[5].text(0.5, -7.7, r'$\Delta$t = 0.1', fontsize=20)
axs[5].text(0.5, 6.5, '(f)', fontsize=30)

axs[2].set_xlim([0,30])
axs[2].set_ylim([-8.2,8.2])
axs[2].set_yticks([])
axs[2].set_xticks([])
axs[2].plot(space, w_arr[1000], 'b.')
axs[2].text(0.5, -7.7, r'$\Delta$t = 0.1', fontsize=20)
axs[2].text(0.5, 6.5, '(c)', fontsize=30)
#fig.savefig('fig1_randwalk.pdf', dpi=100, bbox_inches='tight')