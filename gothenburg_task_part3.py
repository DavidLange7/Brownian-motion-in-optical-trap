#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 13:18:21 2024

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
def r_new(r_0, k, gamma, D, delta_t):
    '''
    free diffusion/random walk with initial position x_old and time step delta_t
    '''
    
    w = np.array([np.random.normal() for i in range(3)])
    
    x_0 = r_0[0]
    y_0 = r_0[1]
    z_0 = r_0[2]
    
    x_1 = x_0 - 1/gamma*k[0]*x_0*delta_t + np.sqrt(2*D*delta_t)*w[0]
    y_1 = y_0 - 1/gamma*k[1]*y_0*delta_t + np.sqrt(2*D*delta_t)*w[1]
    z_1 = z_0 - 1/gamma*k[2]*z_0*delta_t + np.sqrt(2*D*delta_t)*w[2]
    
    
    return([x_1, y_1, z_1])


tau = 0.6*10**(-6) #seconds
delta_t = 10*10**(-3) #seconds

T = 300 #kelvin
R = 1*10**(-6) #meters
eta = 0.001 #Ns/m²
k_B = 1.380649*10**(-23) #J/K
gamma = 6*np.pi*eta*R
D = k_B*T/gamma

k_x = 1.0 * 10**(-6) #fN/nm
k_y = k_x
k_z = 0.2 * 10**(-6) #fN/nm

k = [k_x, k_y, k_z]
r_0 = [0, 0, 0]

steps = int(1e+4)

traj_x = [r_0[0]]
traj_y = [r_0[1]]
traj_z = [r_0[2]]

for j in range(1, steps):
    
    tmp = r_new(r_0, k, gamma, D, delta_t)
    traj_x.append(tmp[0])
    traj_y.append(tmp[1])
    traj_z.append(tmp[2])
    
    r_0 = [traj_x[j], traj_y[j], traj_z[j]]
    
traj_x = np.array(traj_x)*10**9
traj_y = np.array(traj_y)*10**9
traj_z = np.array(traj_z)*10**9

#%%
import plotly.express as px
import pandas as pd

data = {
    'x': traj_x,
    'y': traj_y,
    'z': traj_z
}
df = pd.DataFrame(data)

fig_heat = px.density_heatmap(df, x="x", y="y", marginal_x="histogram", marginal_y="histogram")

fig_heat.update_layout(
    title="",
    xaxis_title="x [nm]",
    yaxis_title="y [nm]",
    coloraxis_colorbar=dict(title="Density"),
    coloraxis_showscale=True
)

fig_heat.update_layout(
    font=dict(
        size=20),
    yaxis_range = [-200,200],
    xaxis_range = [-200,200], 
    yaxis_scaleanchor="x")

fig_heat.show()
#fig_heat.write_image("fig_2d_xy.pdf")

#%%

def gaussian(x, norm, mu, sigma):
    '''
    returns a gaussian distribution with mean mu and variance sigma**2
    '''
    
    return(norm*np.exp(-1/2*((x-mu)/sigma)**2))

def gaussian_2d(x, y, mux, varx, muy, vary):
    '''
    returns a 2d gaussian distribution with mean mu and variance sigma**2
    '''
    
    coeff = 1 / (2 * np.pi * np.sqrt(varx * vary))
    exponent = -0.5 * ((x - mux)**2 / varx + (y - muy)**2 / vary)
    
    return coeff * np.exp(exponent)
    
nx, binsx, patchesx = plt.hist((traj_x), bins = 100, histtype='bar',density=True, stacked=True, alpha = 0.8, label = 'x data')
ny, binsy, patchesy = plt.hist((traj_y), bins = 70, histtype='bar',density=True, stacked=True, alpha = 0.8, label = 'y data')
nz, binsz, patchesz = plt.hist((traj_z), bins = 70, histtype='bar',density=True, stacked=True, alpha = 0.8, label = 'z data')


meanx = np.mean(traj_x)
sigmax = np.sqrt(np.var(traj_x))
poptx, pcovx = optimize.curve_fit(gaussian, binsx[1:], nx, p0 = [1, meanx, sigmax])
space_gauss = np.linspace(-200, 200, 1000)
func_gaussx = gaussian(space_gauss, poptx[0], poptx[1], poptx[2])

meany = np.mean(traj_y)
sigmay = np.sqrt(np.var(traj_y))
popty, pcovy = optimize.curve_fit(gaussian, binsy[1:], ny, p0 = [1, meany, sigmay])
func_gaussy = gaussian(space_gauss, popty[0], popty[1], popty[2])

meanz = np.mean(traj_z)
sigmaz = np.sqrt(np.var(traj_z))
poptz, pcovz = optimize.curve_fit(gaussian, binsz[1:], nz, p0 = [1, meanz, sigmaz])
space_gaussz = np.linspace(-600, 600, 1000)
func_gaussz = gaussian(space_gaussz, poptz[0], poptz[1], poptz[2])

x_grid, y_grid = np.meshgrid(space_gauss, space_gauss)

weights = gaussian_2d(x_grid, y_grid, meanx, sigmax**2, meany, sigmay**2)

#%%
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm

fig = plt.figure()

gs = GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[3,1])

ax1 = fig.add_subplot(gs[0, 1])
#ax1.plot(traj_x, traj_z, 'r.', markersize=0.5)
weights = gaussian_2d(x_grid, y_grid, meanx, sigmax**2, meanz, sigmaz**2)
ax1.imshow(weights, extent=[space_gauss.min(), space_gauss.max(), space_gaussz.min(), space_gaussz.max()], origin='lower', cmap='Reds', aspect='auto')
ax1.set_xlim([-200,200])
ax1.set_ylim([-600, 600])
ax1.set_aspect('equal', adjustable='box')
ax1.set_xlabel('x [nm]')
ax1.set_ylabel('z [nm]')


ax2 = fig.add_subplot(gs[1, 1])
#ax2.plot(traj_x, traj_y, 'r.', markersize=0.5)
weights = gaussian_2d(x_grid, y_grid, meanx, sigmax**2, meany, sigmay**2)
ax2.imshow(weights, extent=[space_gauss.min(), space_gauss.max(), space_gauss.min(), space_gauss.max()], origin='lower', cmap='Reds', aspect='auto')
ax2.set_xlim([-200,200])
ax2.set_ylim([-200, 200])
ax2.set_aspect(1, adjustable='box')
ax2.set_xlabel('x [nm]')
ax2.set_ylabel('y [nm]')


traj_x_red = traj_x[:1500]
traj_y_red = traj_y[:1500]
traj_z_red = traj_z[:1500]


cmap = plt.get_cmap('Greys')
norm = plt.Normalize(vmin=200, vmax=len(traj_x_red))

#cmap = LinearSegmentedColormap.from_list('custom_grey', ['lightgrey', 'black'])
cmap = LinearSegmentedColormap.from_list('custom_blue', ['lightblue', 'darkblue'])

norm = plt.Normalize(vmin=0, vmax=len(traj_x_red)+1000)


ax3 = fig.add_subplot(gs[:, 0], projection='3d')

for i in range(len(traj_x) - 1):
    ax3.plot(traj_x_red[i:i+2], traj_y_red[i:i+2], traj_z_red[i:i+2], color=cmap(norm(i)))

#ax3.scatter(traj_x, traj_y, traj_z, s = 1, c = 'b', marker = '.')
ax3.set_aspect('equal', adjustable='box')
ax3.set_xticks([-200,0,200])
ax3.set_yticks([-200,0,200])

ax3.set_zlim([-450,450])


ax3.set_xlabel('x [nm]')
ax3.set_ylabel('y [nm]')
ax3.set_zlabel('z [nm]')


cmap_reversed = cmap.reversed()
sm = cm.ScalarMappable(cmap=cmap_reversed, norm=norm)
cbar = fig.colorbar(sm, ax=ax3, ticks=[])
cbar.set_label('Time [a.u.]')

x_sd = np.sqrt(np.var(traj_x))
y_sd = np.sqrt(np.var(traj_y)) 
z_sd = np.sqrt(np.var(traj_z)) 


phi = np.linspace(0,2*np.pi, 256).reshape(256, 1)
theta = np.linspace(0, np.pi, 256).reshape(-1, 256)

x = 3*x_sd*np.sin(theta)*np.cos(phi)
y = 3*y_sd*np.sin(theta)*np.sin(phi)
z = 3*z_sd*np.cos(theta)


ax3.plot_surface(x, y, z, color='grey', alpha = 0.3)

plt.tight_layout()
plt.show()
fig.savefig('fig3_opticaltrap.pdf', dpi=500, bbox_inches='tight')

#%%
ntop = 15

k_x_1 = np.linspace(0.1, 1, ntop-10) * 10**(-6)
k_x_2 = np.linspace(1.5, 10, 10)* 10**(-6)
k_x = np.array([*k_x_1, *k_x_2])
k_y = k_x
k_z = np.array([0.2 * 10**(-6) for i in range(ntop)])

k = np.transpose([k_x, k_y, k_z])

traj_xs = []
traj_ys = []

vars_x = []
vars_y = []

steps = 200000


for j in range(ntop):
    r_0 = [0,0,0]
    traj_x = [r_0[0]]
    traj_y = [r_0[1]]
    traj_z = [r_0[1]]
    
    delta_t = 5e-4
    
    
    for i in range(1, steps):
                
        tmp = r_new(r_0, [*k[j]], gamma, D, delta_t)
        
        traj_x.append(tmp[0])
        traj_y.append(tmp[1])
        traj_z.append(tmp[2])

        r_0 = [traj_x[i], traj_y[i], traj_z[i]]
    
    traj_x = np.array(traj_x)*10**9
    traj_y = np.array(traj_y)*10**9
    traj_z = np.array(traj_z)*10**9
        
    traj_xs.append(traj_x)
    traj_ys.append(traj_y)
    vars_x.append(np.var(traj_x))
    vars_y.append(np.var(traj_y))
    
    
def var_func_prop(k, A):
    
    return(A*1/(k))


plt.figure('variances')
plt.plot(k_y*(10**6), np.array(vars_y)/(10**(4)), 'o', color = 'black', label = 'simulation', markersize = 12)

popt, pcov = optimize.curve_fit(var_func_prop, k_y*(10**6), np.array(vars_y)/(10**(4)))
plt.xlabel(r'$\mathrm{k_y \: [fN/nm]}$')
plt.ylabel(r'$\mathrm{\sigma ^2_y \: [x10^4 \: nm^2]}$')
k_space = np.linspace(0.01, 20, 1000)
plt.xlim([0, 10])
plt.ylim([0, 1.55])
plt.plot(k_space, var_func_prop(k_space, popt[0]), label = 'reciprocal function fit')
plt.legend()

plt.savefig('fig4_variancefit.pdf', dpi=500, bbox_inches='tight')

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

def manual_msd(data, nlags):
        
    
    n = len(data)
            
    msd_values = [np.mean((data[:n-lag] - data[lag:]) ** 2) for lag in range(nlags)]
    
    return msd_values

steps = int(10000)

k_x = 5 * 10**(-6) #N/m
k_y = k_x
k_z = 0.2 * 10**(-6) #N/m

k = [k_x, k_y, k_z]

nlags = 500

delta_t = 0.001

acf_tot = np.zeros(nlags)
msd_tot = np.zeros(nlags)



n_tot = 1000
for ind in range(n_tot):
    
    r_0 = [0,0,0]
    traj_x = [r_0[0]]
    traj_y = [r_0[1]]
    traj_z = [r_0[1]]
    
    if ind % 100 == 0:
        print(ind)
    for i in range(1, steps):
                
        tmp = r_new(r_0, k, gamma, D, delta_t)
        
        traj_x.append(tmp[0])
        traj_y.append(tmp[1])
        traj_z.append(tmp[2])
    
        r_0 = [traj_x[i], traj_y[i], traj_z[i]]
        
    acf_tot += manual_acf(np.array(traj_x)*10**(9), delta_t, nlags-1)
    msd_tot += manual_msd(np.array(traj_x)*10**(9), nlags)

plt.plot(np.linspace(0, nlags*delta_t, nlags)*10**(3), acf_tot/n_tot)
plt.loglog(np.linspace(0, nlags*delta_t, nlags)*10**(3), msd_tot/n_tot)

'''
#%%

plt.figure('fig_Cx')
plt.plot(np.linspace(0, nlags*delta_t, nlags)*10**(3), acf_02, label ='k = 0.2 fN/nm', linewidth = 4)
plt.plot(np.linspace(0, nlags*delta_t, nlags)*10**(3), acf_1, label ='k = 1.0 fN/nm', linewidth = 4)
plt.plot(np.linspace(0, nlags*delta_t, nlags)*10**(3), acf_5, label ='k = 5.0 fN/nm', linewidth = 4)
plt.legend()
plt.xlabel('t [ms]')
plt.ylabel(r'$\mathrm{C_x(t)}$')

plt.savefig('fig5_autocorr.pdf', dpi=500, bbox_inches='tight')

plt.figure('fig_msd')
plt.loglog(np.linspace(0, nlags*delta_t, nlags)*10**(3), msd_02, label ='k = 0.2 fN/nm', linewidth = 4)
plt.loglog(np.linspace(0, nlags*delta_t, nlags)*10**(3), msd_1, label ='k = 1.0 fN/nm', linewidth = 4)
plt.loglog(np.linspace(0, nlags*delta_t, nlags)*10**(3), msd_5, label ='k = 5.0 fN/nm', linewidth = 4)
plt.legend()
plt.xlabel('t [ms]')
plt.ylabel(r'$\mathrm{\langle x(t)^2\rangle \: [nm^2]}$')
plt.xlim([10**(-2), 5*10**2])
plt.ylim([1, 10**5])

plt.savefig('fig5_msd.pdf', dpi=500, bbox_inches='tight')
'''