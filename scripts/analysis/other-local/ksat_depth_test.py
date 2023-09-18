
#%%
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

from landlab import RasterModelGrid
from landlab.components import GroundwaterDupuitPercolator
from DupuitLEM.grid_functions import bind_avg_recip_ksat, bind_avg_exp_ksat

#%%

df_params = {}
df_params['ksat_type'] = 'exp'
# df_params['ksat'] = 1
df_params['ksurface'] = 1
df_params['kdecay'] = 1
df_params['kdepth'] = 1

try:
    ksat_type = df_params['ksat_type']

    if ksat_type == 'recip':
        try:
            ks = df_params['ksurface']
            d = df_params['kdecay']

            ksat = bind_avg_recip_ksat(ks, d)
        except KeyError:
            print('could not find parameters ksurface and/or kdecay for ksat_type %s'%ksat_type)

    elif ksat_type == 'exp':
        try:
            ks = df_params['ksurface']
            k0  = df_params['kdepth']
            dk = df_params['kdecay']

            ksat = bind_avg_exp_ksat(ks, k0, dk)
        except KeyError:
            print('could not find parameters ksurface, kdepth, and/or kdecay for ksat_type %s'%ksat_type)
    else:
        print('Could not find ksat_type %s'%ksat_type)
        raise KeyError
except KeyError:
    print('Using constant ksat')
    ksat = df_params['ksat']

#%%

def k_func(h, b, dk, ks):
    return ks * np.exp(-(b-h)/dk)

def dk_func(dk, n, b):
    return (n*dk)/b * (1-np.exp(-b/dk))

def ks_func(n, keff):
    return n*keff

n = 2
b = 5
keff = 2
dk = fsolve(dk_func, 6, args=(n,b))[0]
ks = ks_func(n,keff)

#%%


def k_func_1(h, b, d, ks, n):
    return ks/(((b-h)/d+1))**n

b = 1
d = 1
ks = 0.001
h = np.linspace(0,1,50)
plt.figure()
for n in range(1,5):
    plt.plot(k_func_1(h,b,d,ks,n), h, label=f'n={n}')
plt.xlabel('ksat')
plt.ylabel('h')
plt.legend()

#%%
b = 1
d = 1
ks = 0.001
n = 1
h = np.linspace(0,1,50)
plt.figure()
for d in np.geomspace(0.1,5,5):
    plt.plot(k_func_1(h,b,d,ks,n), h, label=f'd={d}')
plt.xlabel('ksat')
plt.ylabel('h')
plt.legend()

#%%
# reciprocal ksat functions

def k_func_1(h, b, d, ks):
    # calculate ksat at aquifer thickness h with reciprocal function
    return ks/((b-h)/d+1)

def ks_func_1(T, b, d):
    # calculate surface ksat for reciprocal function
    return T/(d*np.log((b+d)/d))

def keff_func_1(h, b, d, ks):
    # calculate effective hydraulic conductivity for aquifer thickness h
    return (ks*d)/h * np.log((b + d)/(b + d - h))

#%%
# plot ksat with depth for different b

ks = 0.001
T = 5
plt.figure()
for b in np.geomspace(0.1,10,5):
    d = 0.2 * b
    h = np.linspace(0,b,50)
    ks = ks_func_1(T, b, d)
    plt.plot(k_func_1(h,b,d,ks), h, label=f'b={b}')
plt.xlabel('ksat')
plt.ylabel('h')
plt.legend()

#%%
# plot ksat with depth for different decay rates d

ks = 0.001
T = 5
b = 5
plt.figure()
for d in np.linspace(0.1,0.9,9):
    h = np.linspace(0,b,50)
    ks = ks_func_1(T, b, d)
    plt.plot(k_func_1(h,b,d,ks), h, label=f'd={d}')
plt.xlabel('ksat')
plt.ylabel('h')
# plt.xscale('log')
plt.legend()


#%%
# develop tests for DupuitLEM

# initialize model grid
mg = RasterModelGrid((4, 4), xy_spacing=1.0)
elev = mg.add_zeros("node", "topographic__elevation")
elev[:] = 1
base = mg.add_zeros("node", "aquifer_base__elevation")
base[:] = 0
wt = mg.add_zeros("node", "water_table__elevation")
wt[:] = 0.5

# initialize model without giving k_func
gdp = GroundwaterDupuitPercolator(mg)

# run model and assert that K hasn't changed from the default value
gdp.run_one_step(0)
assert np.equal(0.001, gdp.K).all()

gdp.run_with_adaptive_time_step_solver(0)
assert np.equal(0.001, gdp.K).all()

# create a simple k_func, where hydraulic conductivity varies linearly
# with depth, from Ks at surface to 0 at aquifer base
# def k_func_test(grid, Ks=0.01):
#     h = grid.at_node["aquifer__thickness"]
#     b = (
#         grid.at_node["topographic__elevation"]
#         - grid.at_node["aquifer_base__elevation"]
#     )
#     blink = map_mean_of_link_nodes_to_link(grid, b)
#     hlink = map_mean_of_link_nodes_to_link(grid, h)

#     return (hlink / blink) * Ks



# # initialize model with given k_func
# gdp1 = GroundwaterDupuitPercolator(mg, hydraulic_conductivity=k_func_test)

# # run model and assert that K has been updated correctly
# gdp1.run_one_step(0)
# assert np.equal(0.005, gdp1.K).all()

# gdp1.run_with_adaptive_time_step_solver(0)
# assert np.equal(0.005, gdp1.K).all()

