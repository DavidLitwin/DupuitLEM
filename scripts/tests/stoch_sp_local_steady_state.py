"""
Stochastic recharge + constant thickness + StreamPowerModel

Same as stoch_sp_1, but tests to see how to measure steady state topogrpahy

Date: 2 Jul 2020
"""
import os
import numpy as np
from itertools import product
import pandas as pd
import pickle

from landlab import RasterModelGrid
from landlab.components import (
    GroundwaterDupuitPercolator,
    LinearDiffuser,
    FastscapeEroder,
    PrecipitationDistribution,
    )
from landlab.io.netcdf import read_netcdf
from DupuitLEM import StreamPowerModel
from DupuitLEM.auxiliary_models import (
    HydrologyEventStreamPower, 
    RegolithConstantThicknessPerturbed,
    RegolithConstantThickness
    )
from DupuitLEM.grid_functions.grid_funcs import (
    bind_avg_hydraulic_conductivity
    )

# task_id = os.environ['SLURM_ARRAY_TASK_ID']
# ID = int(task_id)
ID = 33

#dim equations
def K_sp_fun(beq, n, Pe, gam, lam, pi, om):
    return (Pe*gam*lam*pi)/(beq**2*n*om**2)

def D_fun(beq, p, n, gam, lam, pi, om):
    return (beq*p*lam*pi*om)/(n*gam**2)

def ksat_fun(p, gam, phi):
    return (p*phi)/gam

def U_fun(p, n, pi):
    return (p*pi)/n

def l_fun(beq, gam, om):
    return (om*beq)/gam

#generate dimensioned parameters
def generate_parameters(p, beq, n, gam, pe, lam, pi, phi, om):

    l = l_fun(beq, gam, om)
    ksat = ksat_fun(p, gam, phi)
    U = U_fun(p, n, pi)
    D = D_fun(beq, p, n, gam, lam, pi, om)
    K = K_sp_fun(beq, n, pe, gam, lam, pi, om)

    return K, D, U, ksat, p, beq, l, n

#parameters
MSF = 500 # morphologic scaling factor [-]
T_h = 30*24*3600 # total hydrological time
T_m = 2.5e6*(365*24*3600) # total simulation time [s]

pe_all = np.geomspace(10,10000,6)
phi_all = np.geomspace(1,1000,6)
pe1 = np.array(list(product(pe_all,phi_all)))[:,0]
phi1 = np.array(list(product(pe_all,phi_all)))[:,1]
lam1 = 0.1
pi1 = 5e-6
om1 = 20
p1 = 1/(365*24*3600) # recharge rate [m/s]
n1 = 0.1 # drainable porosity []
gam1 = 0.2
beq1 = 1 #equilibrium depth [m]

storm_dt = 2*3600 # storm duration [s]
interstorm_dt = 48*3600 # interstorm duration [s]
p_d = p1*(storm_dt+interstorm_dt) # storm depth [m]

params = np.zeros((len(pe1),8))
for i in range(len(pe1)):

    params[i,:] = generate_parameters(p1, beq1, n1, gam1, pe1[i], lam1, pi1, phi1[i], om1)

df_params = pd.DataFrame(params,columns=['K', 'D', 'U', 'ksat', 'p', 'beq', 'l', 'n'])
df_params['storm_dt'] = storm_dt
df_params['interstorm_dt'] = interstorm_dt
df_params['depth'] = p_d

pickle.dump(df_params, open('parameters.p','wb'))

Ksp = df_params['K'][ID] #streampower coefficient
D = df_params['D'][ID] #hillslope diffusivity
U = df_params['U'][ID] #uplift Rate
Ks = df_params['ksat'][ID]
K0 = Ks*0.01
p = df_params['p'][ID]
beq = df_params['beq'][ID]
n = df_params['n'][ID]

#initialize grid_functions
ksat_fun = bind_avg_hydraulic_conductivity(Ks,K0,beq) # hydraulic conductivity [m/s]

#initialize grid
mg = read_netcdf('C:/Users/dgbli/Documents/MARCC_output/DupuitLEMResults/post_proc/stoch_sp_1_2/grid_%d.nc'%ID)
z = mg.at_node['topographic__elevation']
zb = mg.at_node['aquifer_base__elevation']

np.random.seed(1234)
grid = RasterModelGrid((100, 100), xy_spacing=10.0)
grid.set_status_at_node_on_edges(right=grid.BC_NODE_IS_CLOSED, top=grid.BC_NODE_IS_CLOSED, \
                              left=grid.BC_NODE_IS_FIXED_VALUE, bottom=grid.BC_NODE_IS_CLOSED)
elev = grid.add_zeros('node', 'topographic__elevation')
elev[:] = z.copy()
base = grid.add_zeros('node', 'aquifer_base__elevation')
base[:] = zb.copy()
wt = grid.add_zeros('node', 'water_table__elevation')
wt[:] = elev.copy()

#initialize landlab components
gdp = GroundwaterDupuitPercolator(grid, porosity=n, hydraulic_conductivity=ksat_fun, \
                                  regularization_f=0.01, recharge_rate=0.0, \
                                  courant_coefficient=0.9, vn_coefficient = 0.9)
pd = PrecipitationDistribution(grid, mean_storm_duration=storm_dt,
    mean_interstorm_duration=interstorm_dt, mean_storm_depth=p_d,
    total_t=T_h)
pd.seed_generator(seedval=1235)
ld = LinearDiffuser(grid, linear_diffusivity = D)

#initialize other models
hm = HydrologyEventStreamPower(
        grid,
        precip_generator=pd,
        groundwater_model=gdp,
)
#use surface_water__discharge for steady case
sp = FastscapeEroder(grid, K_sp = Ksp, m_sp = 1, n_sp=1, discharge_field='surface_water__discharge')
# rm = RegolithConstantThicknessPerturbed(grid, equilibrium_depth=beq, uplift_rate=U, std=1e-4, seed=1236)

rm = RegolithConstantThickness(grid, equilibrium_depth=beq, uplift_rate=U)

mdl = StreamPowerModel(grid,
        hydrology_model = hm,
        diffusion_model = ld,
        streampower_model = sp,
        regolith_model = rm,
        morphologic_scaling_factor = MSF,
        total_morphological_time = T_m,
        verbose=True,
        output_dict =None,
)

#%% run whole model one step at a time

N = 100

max_diff = np.zeros(N)
perc90_diff = np.zeros(N)
perc50_diff = np.zeros(N)
mean_diff = np.zeros(N)
mean_mean_diff = np.zeros(N)

for i in range(N):

    elev0 = elev.copy()
    mdl.run_step(mdl.dt_m)
    print('Completed model loop %d' % i)

    elev_diff = abs(elev-elev0)/(gam1*df_params['l'][ID])

    max_diff[i] = np.max(elev_diff)
    perc90_diff[i] = np.percentile(elev_diff,90)
    perc50_diff[i] = np.percentile(elev_diff,50)
    mean_diff[i] = np.mean(elev_diff)

    mean_mean_diff[i] = abs(np.mean(elev)-np.mean(elev0))/(gam1*df_params['l'][ID])
    
    
#%%
import matplotlib.pyplot as plt

plt.figure()
plt.plot(max_diff, alpha=0.5, label='max')
plt.plot(perc90_diff, alpha=0.5, label='90perc')
plt.plot(perc50_diff, alpha=0.5, label='50perc')
plt.plot(mean_diff, alpha=0.5, label='mean')
plt.yscale('log')
plt.legend()

plt.figure()
plt.plot(mean_mean_diff, alpha=0.5, label='mean')
plt.legend()