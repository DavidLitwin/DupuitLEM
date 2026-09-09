"""
Generate parameters for StreamPowerModel with from conditioned latin hypercube sampling on
U (erosion rate), P and PET from global watersheds. Other parameters are chosen by pure latin
hypercube sampling based on known parameter ranges, varying a mix of dimensioned and dimensionless
parameters to capture realistic parameter values: alpha, rho, lg, b, transmissivity, ds.

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

Adapted from params_stoch_nld_svm_cLHS.py, 6 Sept. 2026
"""

#%%
import os
import numpy as np
from scipy.stats import qmc
import pandas


#dim equations
def K_fun(v0, lg, tg):
    return np.sqrt(lg)/(np.sqrt(v0)*tg)

def D_fun(lg, tg):
    return lg**2/tg

def U_fun(hg, tg):
    return hg/tg

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


def hg_fun(K, D, U, v0):
    return ((D * U**3)/(v0**2 * K**4))**(1/3)

def lg_fun(K, D, v0):
    return ((D**2)/(v0 * K**2))**(1/3)

def tg_fun(K, D, v0):
    return ((D)/(v0**2 * K**4))**(1/3)

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

# import the pre-built conditioned LHS samples
dfc = pandas.read_csv('P_PET_E_params_sample_clhs2.csv')

# shuffle to remove any latent row structure
dfc = dfc.sample(frac=1, random_state=2026, ignore_index=True)
dfc

#%%

dfc.describe()

#%%

bnd_logalpha = [np.log10(0.05), np.log10(1)] # (-) dissection ratio hg/lg
bnd_loglg = [np.log10(15), np.log10(50)] # m # geomorphic length scale
bnd_logb = [-0.3, 1.5] # m
bnd_logT = [-5, -4] # m2/s
bnd_logds = [-3, -2] # m
bnd_logrho = [-1.5, -0.3] # (-)

bnds = list(zip(bnd_logalpha, bnd_loglg, bnd_logb, bnd_logT, bnd_logds, bnd_logrho))

sampler = qmc.LatinHypercube(d=len(bnds[0]), seed=2023) # alpha, lg, b, T, ds, rho
sample = sampler.random(n=len(dfc))
scaled_sample = qmc.scale(sample, bnds[0], bnds[1])

v0 = 30 # contour width (also grid spacing) [m]
sc = 1.0
theta = 0.0 # don't use threshold model
ne = 0.1 # drainable porosity [-]
na = 0.15
phi = na/ne

Tg_nd = 500 # total duration in units of tg [-]
dtg_nd = 1e-2 # target outer (morphologic-scaling) geomorphic timestep, in units of tg [-]
              # equivalent to an uplift-per-step target of dtg_nd*hg
dtg_max_nd = 2e-3 # maximum geomorphic substep, in units of tg [-]; subdivides dtg for
                   # numerical accuracy (~dtg_nd/dtg_max_nd substeps per outer step)
Th_nd = 25 # hydrologic time in units of (tr+tb) [-]

bin_capacity_nd = 0.01 # bin capacity as a proportion of mean storm depth
Nx = 200 # number of grid cells width and height

params = []
for i in range(scaled_sample.shape[0]):

    alpha = 10**scaled_sample[i,0]
    lg = 10**scaled_sample[i,1]
    U = 10**dfc['log10E'].iloc[i] / (3600*24*365) # m/s

    D = lg * U / alpha
    K = U / (alpha * np.sqrt(v0 * lg))

    hg = hg_fun(K, D, U, v0)
    tg = tg_fun(K, D, v0)

    b = 10**scaled_sample[i,2]
    T = 10**scaled_sample[i,3]
    ds = 10**scaled_sample[i,4]
    rho = 10**scaled_sample[i,5]
    ksat = T/b


    p = dfc['P'].iloc[i] / (3600*24*365) # m/s
    PET = dfc['PET'].iloc[i] / (3600*24*365) # m/s
    pet = PET/(1-rho)
    ai = pet/p
    tr = rho * (ds/p)
    tb = (1-rho) * (ds/p)

    beta = beta_fun(hg, lg, ksat, p)
    gamma = gamma_fun(hg, lg, ksat, p, b)
    sigma = sigma_fun(p, b, ne, tr, tb)

    E0 = E0_fun(theta, hg, tg)

    params.append([K, D, U, ksat, p, pet, b, ne, na, v0, hg, lg, tg, E0, ds, tr, tb, alpha, gamma, beta, sigma, rho, ai, theta, phi])

df_params = pandas.DataFrame(np.array(params),columns=['K', 'D', 'U', 'ksat', 'p', 'pet', 'b', 'ne', 'na', 'v0', 'hg', 'lg', 'tg', 'E0', 'ds', 'tr', 'tb', 'alpha', 'gam', 'beta', 'sigma', 'rho', 'ai', 'theta', 'phi'])
df_params['Sc'] = sc
df_params['Nz'] = round((df_params['b']*df_params['na'])/(bin_capacity_nd*df_params['ds']))
df_params['Nx'] = Nx
df_params['td'] = (df_params['lg']*df_params['ne'])/(df_params['ksat']*df_params['hg']/df_params['lg']) # characteristic aquifer drainage time [s]
df_params['ha'] = (df_params['p']*df_params['lg'])/(df_params['ksat']*df_params['hg']/df_params['lg']) # characteristic aquifer thickness [m]
df_params['Tg'] = Tg_nd*df_params['tg'] # Total geomorphic simulation time [s]
df_params['Th'] = Th_nd*(df_params['tr']+df_params['tb']) # hydrologic simulation time [s]
df_params['dtg'] = dtg_nd*df_params['tg'] # target outer geomorphic timestep [s] (= ksf*Th)
df_params['ksf'] = df_params['dtg']/df_params['Th'] # morphologic scaling factor, solved so ksf*Th = dtg
df_params['dtg_max'] = dtg_max_nd*df_params['tg'] # the maximum duration of a geomorphic substep [s]
df_params['output_interval'] = (10/(df_params['dtg']/df_params['tg'])).round().astype(int)
# with dtg, dtg_max, Tg all fixed fractions of tg, the number of outer steps
# (Tg/dtg = Tg_nd/dtg_nd) and substeps per step (dtg_nd/dtg_max_nd) are constant
# across every parameter combination, regardless of U, K, or D.

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

try:
    task_id = os.environ['SLURM_ARRAY_TASK_ID']
    ID = int(task_id)
    df_params.loc[ID].to_csv('parameters.csv', index=True)
except KeyError:
    print('In testing mode. Save first row of parameters to parameters.csv')
    df_params.loc[0].to_csv('../run_models/parameters.csv', index=True)

#%%

# df_params_yr['ai_corrected'] = df_params_yr['ai'] * (1-df_params_yr['rho'])
# %%
