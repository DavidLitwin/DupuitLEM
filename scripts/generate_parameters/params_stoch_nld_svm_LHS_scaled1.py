"""
Generate parameters for StreamPowerModel with
-- HydrologyEventVadoseStreamPower
-- FastscapeEroder
-- LinearDiffuser or TaylorNonLinearDiffuser
-- RegolithConstantThickness

Vary parameters using latin hypercube sampling

alpha = hg / lg
gamma = (b ksat hg) / (p lg^2)
beta (formerly Hi) = (ksat hg^2) / (p lg^2)
sigma = (b n) / (p (tr + tb))
rho = tr / (tr + tb)
ai = (pet tb) / (p (tr + tb))
phi = na / ne

6 Jun 2023
"""

#%%
import os
import numpy as np
from scipy.stats import qmc
import pandas

#%%


def K_fun(v0, lg, tg):
    return np.sqrt(lg)/(np.sqrt(v0)*tg)

def D_fun(lg, tg):
    return lg**2/tg

def U_fun(hg, tg):
    return hg/tg

def hg_fun(K, D, U, v0):
    return ((D * U**3)/(v0**2 * K**4))**(1/3)

def lg_fun(K, D, v0):
    return ((D**2)/(v0 * K**2))**(1/3)

def tg_fun(K, D, v0):
    return ((D)/(v0**2 * K**4))**(1/3)

def b_fun(hg, gam, beta):
    return (hg*gam)/beta

def ksat_fun(p, hg, lg, beta):
    return (lg**2*p*beta)/hg**2

def E0_fun(theta, hg, tg):
    return theta*(hg/tg)

def ds_fun(hg, ne, gam, sigma, beta):
    return (hg*ne*gam)/(beta*sigma)

def tr_fun(hg, p, ne, gam, sigma, beta, rho):
    return (hg*ne*gam*rho)/(p*sigma*beta)

def tb_fun(hg, p, ne, gam, sigma, beta, rho):
    return (hg*ne*gam)*(1-rho)/(p*sigma*beta)

def pet_fun(p, rho, ai):
    return (ai*p)/(1-rho)

def alpha_fun(hg, lg):
    return hg / lg

def beta_fun(hg, lg, ksat, p):
    return (ksat * hg**2) / (p * lg**2)

def gamma_fun(hg, lg, ksat, p, b):
    return (ksat * b * hg / lg) / (p * lg)

def sigma_fun(p, b, ne, tr, tb):
    return (b * ne) / (p * (tr + tb))

def ai_fun(p, tr, tb, pet):
    return (pet * tb)/(p * (tr + tb))

def rho_fun(tr, tb):
    return tr / (tr + tb)

def phi_fun(na, ne):
    return na / ne

def theta_fun(E0, hg, tg):
    return E0 / (hg / tg)


def generate_parameters(p, ne, v0, lg, tg, alpha, gam, beta, sigma, rho, ai, theta, phi):

    hg = alpha * lg
    K = K_fun(v0, lg, tg)
    D = D_fun(lg, tg)
    U = U_fun(hg, tg)
    b = b_fun(hg, gam, beta)
    ksat = ksat_fun(p, hg, lg, beta)
    E0 = E0_fun(theta, hg, tg)
    ds = ds_fun(hg, ne, gam, sigma, beta)
    tr = tr_fun(hg, p, ne, gam, sigma, beta, rho)
    tb = tb_fun(hg, p, ne, gam, sigma, beta, rho)
    pet = pet_fun(p, rho, ai)
    na = phi*ne

    return K, D, U, ksat, p, pet, b, ne, na, v0, hg, lg, tg, E0, ds, tr, tb, alpha, gam, beta, sigma, rho, ai, theta, phi

bnd_gam = [1.0, 10.0]
bnd_sigma = [10.0, 100.0]
bnd_beta = [0.5, 5.0]
bnd_alpha = [0.05, 0.5]
bnd_ai = [0.2, 0.8]
bnd_rho = [0.05, 0.6]

bnds = list(zip(bnd_gam, bnd_sigma, bnd_beta, bnd_alpha, bnd_ai, bnd_rho))
sampler = qmc.LatinHypercube(d=len(bnds[0]), seed=2023) # gam, sigma, beta, alpha, Ai, rho
sample = sampler.random(n=12)

scaled_sample = qmc.scale(sample, bnds[0], bnds[1])

sc = 1.25
theta = 0.0 # don't use threshold model
phi = 1.5
lg = 15 # geomorphic length scale [m]
tg = 10000*(365*24*3600) # geomorphic timescale [s]
v0 = 2.0*lg # contour width (also grid spacing) [m]
ne = 0.1 # drainable porosity [-]
p = 1.0/(365*24*3600) # average precip rate

Tg_nd = 2000 # total duration in units of tg [-]
dtg_max_nd = 5e-3 # maximum geomorphic timestep in units of tg [-]
Th_nd = 25 # hydrologic time in units of (tr+tb) [-]

bin_capacity_nd = 0.01 # bin capacity as a proportion of mean storm depth
Nx = 200 # number of grid cells width and height

params = []
for i in range(scaled_sample.shape[0]):
    gam = scaled_sample[i,0]
    sigma = scaled_sample[i,1]
    beta = scaled_sample[i,2]
    alpha = scaled_sample[i,3]
    ai = scaled_sample[i,4]
    rho = scaled_sample[i,5]
    params.append(generate_parameters(p, ne, v0, lg, tg, alpha, gam, beta, sigma, rho, ai, theta, phi))

df_all = pandas.DataFrame(np.array(params),columns=['K', 'D', 'U', 'ksat', 'p', 'pet', 'b', 'ne', 'na', 'v0', 'hg', 'lg', 'tg', 'E0', 'ds', 'tr', 'tb', 'alpha', 'gam', 'beta', 'sigma', 'rho', 'ai', 'theta', 'phi'])

# only the base (dimensional) parameters are carried forward; scales and dimensionless parameters are recomputed below
df_core = df_all[['K', 'D', 'U', 'ksat', 'p', 'pet', 'b', 'ne', 'na', 'v0', 'E0', 'ds', 'tr', 'tb']].copy()
scales = [0.75, 1.0, 1.25]
b_modes = ['fixed', 'inv', 'prop']
variants = []
for s_ in scales:
    for m_ in b_modes:
        if s_ == 1.0 and m_ != 'fixed':
            continue # b is identical to 'fixed' when s=1
        variants.append((s_, m_))


# create new rows for each variant
new_rows = []
for scale, mode in variants:
    for idx, row in df_core.iterrows():
        new_row = row.copy()
        new_row['source_run'] = idx
        new_row['scale'] = scale
        new_row['b_mode'] = mode
        new_row['b0'] = row['b']
        new_row['b'] = {'fixed': new_row['b0'], 'prop': new_row['b0']*scale, 'inv': new_row['b0']/scale}[mode]
        new_row['U0'] = row['U']
        new_row['U'] = row['U'] * scale
        new_rows.append(new_row)

#datafrome with all variants
df_params = pandas.DataFrame(new_rows, columns=['K', 'D', 'U', 'ksat', 'p', 'pet', 'b', 'ne', 'na', 'v0', 'E0', 'ds', 'tr', 'tb', 'source_run', 'scale', 'b_mode', 'b0', 'U0']).reset_index(drop=True) # unique index = array task ID

# recalculate scales and dimensionless parameters based on new b and U
df_params['hg'] = hg_fun(df_params['K'], df_params['D'], df_params['U'], df_params['v0'])
df_params['lg'] = lg_fun(df_params['K'], df_params['D'], df_params['v0'])
df_params['tg'] = tg_fun(df_params['K'], df_params['D'], df_params['v0'])
df_params['alpha'] = alpha_fun(df_params['hg'], df_params['lg'])
df_params['beta'] = beta_fun(df_params['hg'], df_params['lg'], df_params['ksat'], df_params['p'])
df_params['gam'] = gamma_fun(df_params['hg'], df_params['lg'], df_params['ksat'], df_params['p'], df_params['b'])
df_params['sigma'] = sigma_fun(df_params['p'], df_params['b'], df_params['ne'], df_params['tr'], df_params['tb'])
df_params['rho'] = rho_fun(df_params['tr'], df_params['tb'])
df_params['ai'] = ai_fun(df_params['p'], df_params['tr'], df_params['tb'], df_params['pet'])
df_params['phi'] = phi_fun(df_params['na'], df_params['ne'])
df_params['theta'] = theta_fun(df_params['E0'], df_params['hg'], df_params['tg'])

df_params['Sc'] = sc
df_params['Nz'] = round((df_params['b']*df_params['na'])/(bin_capacity_nd*df_params['ds']))
df_params['Nx'] = Nx
df_params['td'] = (df_params['lg']*df_params['ne'])/(df_params['ksat']*df_params['hg']/df_params['lg']) # characteristic aquifer drainage time [s]
df_params['ha'] = (df_params['p']*df_params['lg'])/(df_params['ksat']*df_params['hg']/df_params['lg']) # characteristic aquifer thickness [m]
df_params['Tg'] = Tg_nd*df_params['tg'] # Total geomorphic simulation time [s]
df_params['Th'] = Th_nd*(df_params['tr']+df_params['tb']) # hydrologic simulation time [s]
df_params['dtg'] = dtg_max_nd*df_params['tg']#df_params['ksf']*df_params['Th'] # geomorphic timestep [s]
df_params['dtg_max'] = 200*365*24*3600 #dtg_max_nd*df_params['tg'] # the maximum duration of a geomorphic substep [s]
df_params['ksf'] = df_params['dtg']/df_params['Th'] # morphologic scaling factor
df_params['output_interval'] = (20/(df_params['dtg']/df_params['tg'])).round().astype(int)

#%%

def convert_sec_to_yr(seconds):
    return seconds / (365*24*3600)

def convert_inverse_sec_to_per_yr(inverse_seconds):
    return inverse_seconds * (365*24*3600)

cols_with_sec = ['tg', 'tr', 'tb', 'td', 'Tg', 'Th', 'dtg', 'dtg_max']
cols_inverse_sec = ['p', 'pet', 'ksat', 'K', 'D', 'U', 'E0']

df_params_yr = df_params.copy()
df_params_yr[cols_with_sec] = df_params_yr[cols_with_sec].map(convert_sec_to_yr)
df_params_yr[cols_inverse_sec] = df_params_yr[cols_inverse_sec].map(convert_inverse_sec_to_per_yr)

#%%
task_id = os.environ['SLURM_ARRAY_TASK_ID']
ID = int(task_id)
df_params.iloc[ID].to_csv('parameters.csv', index=True)

