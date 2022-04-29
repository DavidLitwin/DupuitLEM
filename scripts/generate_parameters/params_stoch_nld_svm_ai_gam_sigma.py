"""
Generate parameters for StreamPowerModel with
-- HydrologyEventVadoseStreamPower or HydrologyEventVadoseThresholdStreamPower
-- FastscapeEroder
-- LinearDiffuser or TaylorNonLinearDiffuser
-- RegolithConstantThickness

Vary ai and sigma.

alpha = hg / lg
gamma = (b ksat hg) / (p lg^2)
Hi = (ksat hg^2) / (p lg^2)
sigma = (b n) / (p (tr + tb))
rho = tr / (tr + tb)
ai = (pet tb) / (p (tr + tb))
theta = E0 tg / hg
phi = na / ne

6 Dec 2021
"""

import os
import numpy as np
from itertools import product
import pandas

task_id = os.environ['SLURM_ARRAY_TASK_ID']
ID = int(task_id)

#dim equations
def K_fun(v0, lg, tg):
    return np.sqrt(lg)/(np.sqrt(v0)*tg)

def D_fun(lg, tg):
    return lg**2/tg

def U_fun(hg, tg):
    return hg/tg

def b_fun(hg, gam, hi):
    return (hg*gam)/hi

def ksat_fun(p, hg, lg, hi):
    return (lg**2*p*hi)/hg**2

def E0_fun(theta, hg, tg):
    return theta*(hg/tg)

def ds_fun(hg, ne, gam, sigma, hi):
    return (hg*ne*gam)/(hi*sigma)

def tr_fun(hg, p, ne, gam, sigma, hi, rho):
    return (hg*ne*gam*rho)/(p*sigma*hi)

def tb_fun(hg, p, ne, gam, sigma, hi, rho):
    return (hg*ne*gam)*(1-rho)/(p*sigma*hi)

def pet_fun(p, rho, ai):
    return (ai*p)/(1-rho)

def generate_parameters(p, ne, v0, hg, lg, tg, gam, hi, sigma, rho, ai, theta, phi):

    alpha = hg/lg
    K = K_fun(v0, lg, tg)
    D = D_fun(lg, tg)
    U = U_fun(hg, tg)
    b = b_fun(hg, gam, hi)
    ksat = ksat_fun(p, hg, lg, hi)
    E0 = E0_fun(theta, hg, tg)
    ds = ds_fun(hg, ne, gam, sigma, hi)
    tr = tr_fun(hg, p, ne, gam, sigma, hi, rho)
    tb = tb_fun(hg, p, ne, gam, sigma, hi, rho)
    pet = pet_fun(p, rho, ai)
    na = phi*ne

    return K, D, U, ksat, p, pet, b, ne, na, v0, hg, lg, tg, E0, ds, tr, tb, alpha, gam, hi, sigma, rho, ai, theta, phi


# params for both hyd1d recharge estimation and lem
sigma_all = [8,128]
gam_all = [1,16]
ai_all = [0.25, 0.75]
hi_all = [0.5, 5.0]

sc = 0.5
theta = 0.0
rho = 0.03
phi = 1.5
hg = 2.25
lg = 15 # geomorphic length scale [m]
tg = 22500*(365*24*3600) # geomorphic timescale [s]
v0 = 2.0*lg # contour width (also grid spacing) [m]
ne = 0.1 # drainable porosity [-]
p = 1.0/(365*24*3600) # average precip rate

Tg_nd = 2000 # total duration in units of tg [-]
dtg_max_nd = 2e-3 # maximum geomorphic timestep in units of tg [-]
ksf_base = 500 # morphologic scaling factor
Th_nd = 25 # hydrologic time in units of (tr+tb) [-]

bin_capacity_nd = 0.01 # bin capacity as a proportion of mean storm depth
Nx = 125 # number of grid cells width and height

params = []
for hi, ai, gam, sigma in product(hi_all, ai_all, gam_all, sigma_all):
    params.append(generate_parameters(p, ne, v0, hg, lg, tg, gam, hi, sigma, rho, ai, theta, phi))

df_params = pandas.DataFrame(np.array(params),columns=['K', 'D', 'U', 'ksat', 'p', 'pet', 'b', 'ne', 'na', 'v0', 'hg', 'lg', 'tg', 'E0', 'ds', 'tr', 'tb', 'alpha', 'gam', 'hi', 'sigma', 'rho', 'ai', 'theta', 'phi'])
df_params['Sc'] = sc
df_params['Nz'] = round((df_params['b']*df_params['na'])/(bin_capacity_nd*df_params['ds']))
df_params['Nx'] = Nx
df_params['td'] = (df_params['lg']*df_params['ne'])/(df_params['ksat']*df_params['hg']/df_params['lg']) # characteristic aquifer drainage time [s]
df_params['beta'] = (df_params['tr']+df_params['tb'])/df_params['td']
df_params['ha'] = (df_params['p']*df_params['lg'])/(df_params['ksat']*df_params['hg']/df_params['lg']) # characteristic aquifer thickness [m]
df_params['Tg'] = Tg_nd*df_params['tg'] # Total geomorphic simulation time [s]
df_params['ksf'] = ksf_base/df_params['beta'] # morphologic scaling factor
df_params['Th'] = Th_nd*(df_params['tr']+df_params['tb']) # hydrologic simulation time [s]
df_params['dtg'] = df_params['ksf']*df_params['Th'] # geomorphic timestep [s]
df_params['dtg_max'] = dtg_max_nd*df_params['tg'] # the maximum duration of a geomorphic substep [s]
df_params['output_interval'] = (10/(df_params['dtg']/df_params['tg'])).round().astype(int)
# df_params['extraction_tol'] = 0.001

df_params.loc[ID].to_csv('parameters.csv', index=True)
