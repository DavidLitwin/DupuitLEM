"""
Comparison variant of params_stoch_nld_svm_cLHS_scaled.py: identical in every
respect except that b (regolith thickness) is sampled as an independent LHS
parameter, over the same approximate range that b = 2*sqrt(hg) produces in the
base script, rather than being coupled to hg. This isolates the effect of the
b-hg coupling by holding everything else (U, K, D, P, PET, T, ds, rho sampling
and ranges) the same.

Generate parameters for StreamPowerModel from conditioned latin hypercube sampling on
U (erosion rate), P and PET from global watersheds.

Pipeline (kept as separate stages so dimensionless numbers, characteristic scales, and
dimensioned parameters don't get tangled together):

1. Forcing (external, from data): U, P, PET.
2. Secondary material parameters (sampled via LHS, independent of U): K, D, b, T
   (transmissivity), ds, rho -- the actual model inputs, in narrow physically-based
   ranges, not tied to the tectonic/climatic forcing. K and D are the "primary" free
   parameters here, symmetric with U: all three go directly into the standard
   hg_fun/lg_fun/tg_fun relations, nothing is solved for or derived from a
   dimensionless target first. Unlike the base script, b is sampled independently
   here rather than coupled to hg.
3. Characteristic scales (derived from 1+2 via hg_fun/lg_fun/tg_fun): hg, lg, tg.
   With n_sp=1, the detachment-limited steady-state scaling gives relief ~ U/K, and
   hg_fun = (D*U^3/(v0^2*K^4))^(1/3) is dominated by its U^1 factor when K, D are
   sampled from narrow ranges -- so hg (and therefore relief) is *strongly*
   dependent on U, with K and D acting as secondary modulators, matching the
   empirical U-relief relationship.
4. Dimensionless numbers (derived, diagnostic only): alpha, beta, gamma, sigma, ai.
   These are never sampled directly -- they're reported/plotted to sanity-check the
   resulting landscapes, not used to control the sampling.
5. Assemble the full dimensioned parameter table, plus timestep controls (Tg, ksf,
   dtg, dtg_max) as fixed fractions of tg so run length scales appropriately with U
   (see estimate_simulation_duration.py for checking this against pilot runs).

Note on alpha (dissection ratio hg/lg) and lg: since lg does not depend on U, alpha
is mathematically tied to hg's U-dependence (alpha = hg/lg). Some of that correlation
is realistic -- more tectonically active landscapes tend to be more dissected -- but
left unconstrained it produces implausible landscapes at the extremes of the U range,
and lg itself can drift outside a physically sensible range. Rather than impose an
artificial K~U (or D~U) correlation to suppress this, K and D are sampled
independently from their LHS ranges and only rejected/redrawn (per row) when the
resulting alpha or lg falls outside a plausible band. Any K-U or D-U correlation that
emerges is then an honest side effect of "don't allow absurd landscapes," not an
assumption baked into the sampling.

-- HydrologyEventVadoseStreamPower
-- FastscapeEroder
-- LinearDiffuser or TaylorNonLinearDiffuser
-- RegolithConstantThickness

Adapted from params_stoch_nld_svm_cLHS_v2.py, 9 Sept. 2026
"""

#%%
import os
import numpy as np
from scipy.stats import qmc
import pandas


# ------------------------------------------------------------------
# Dimensional relations (pure functions; used both to derive scales and,
# where noted, purely as diagnostics/consistency checks)
# ------------------------------------------------------------------

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


# ------------------------------------------------------------------
# Stage 2 helper: sample K, D per row, rejecting/redrawing combinations that
# would produce an implausible landscape (alpha or lg out of bounds) for that
# row's U.
# ------------------------------------------------------------------

def sample_K_D_constrained(
    U, v0, bnd_logK, bnd_logD, alpha_bounds, lg_bounds, base_sample,
    seed=0, max_tries=500,
):
    """
    Sample K and D (independent, secondary material parameters -- the actual
    model inputs) for each row, using `base_sample` (an (N,2) LHS sample in
    [0,1]^2, for coverage) as the first attempt, then falling back to
    independent uniform-in-log redraws (bounded by max_tries) for any row
    whose resulting alpha = hg_fun/lg_fun or lg = lg_fun falls outside the
    given bounds.

    Parameters
    ----------
    U : array-like, shape (N,)
        Forcing (uplift/erosion rate), m/s.
    v0 : float
        Contour width / grid spacing, m.
    bnd_logK : (low, high)
        log10 bounds on K, 1/s.
    bnd_logD : (low, high)
        log10 bounds on D, m2/s.
    alpha_bounds : (low, high)
        Acceptable range for alpha = hg/lg (dissection ratio).
    lg_bounds : (low, high)
        Acceptable range for lg, m (geomorphic length scale).
    base_sample : ndarray, shape (N, 2)
        LHS sample in [0,1]^2 for the initial (K, D) draw per row.
    seed : int
        Seed for the fallback-redraw RNG.
    max_tries : int
        Maximum redraws per row before accepting whatever the last draw gives
        (a warning is printed if any row hits this limit).

    Returns
    -------
    K, D, alpha, lg : ndarrays, shape (N,)
    n_redrawn : int
        Number of rows that needed at least one redraw.
    """
    rng = np.random.default_rng(seed)
    N = len(U)
    K = np.empty(N)
    D = np.empty(N)
    alpha = np.empty(N)
    lg = np.empty(N)
    n_redrawn = 0

    for i in range(N):
        logK = bnd_logK[0] + base_sample[i, 0] * (bnd_logK[1] - bnd_logK[0])
        logD = bnd_logD[0] + base_sample[i, 1] * (bnd_logD[1] - bnd_logD[0])

        for attempt in range(max_tries):
            K_i = 10**logK
            D_i = 10**logD
            hg_i = hg_fun(K_i, D_i, U[i], v0)
            lg_i = lg_fun(K_i, D_i, v0)
            alpha_i = hg_i / lg_i

            if (alpha_bounds[0] <= alpha_i <= alpha_bounds[1]
                    and lg_bounds[0] <= lg_i <= lg_bounds[1]):
                break

            logK = rng.uniform(*bnd_logK)
            logD = rng.uniform(*bnd_logD)
        else:
            print(f"warning: row {i} did not converge to alpha in "
                  f"{alpha_bounds} and lg in {lg_bounds} after {max_tries} "
                  f"tries (alpha={alpha_i:.3g}, lg={lg_i:.3g})")

        if attempt > 0:
            n_redrawn += 1

        K[i], D[i], alpha[i], lg[i] = K_i, D_i, alpha_i, lg_i

    return K, D, alpha, lg, n_redrawn


#%% P, PET, E (U) from conditioned latin hypercube on real data

dfc = pandas.read_csv('P_PET_E_params_sample_clhs2.csv')
dfc = dfc.sample(frac=1, random_state=2026, ignore_index=True)  # shuffle out any row structure

#%%

dfc.describe()

#%%

bnd_logK = [-13.5, -12.5]      # 1/s -- rock erodibility, secondary/independent of U
bnd_logD = [-10.5, -9.5]       # m2/s -- hillslope diffusivity, secondary/independent of U
bnd_logb = [np.log10(2), np.log10(30)]  # m -- regolith thickness, independent of hg
                                # (approximately the range that b = 2*sqrt(hg) spans
                                # in the base script, for a like-for-like comparison)
bnd_logT = [-5, -4]            # m2/s -- transmissivity
bnd_logds = [-3, -2]           # m -- mean storm depth
bnd_logrho = [-1.5, -0.3]      # (-) -- storm/interstorm duration ratio

alpha_bounds = (0.05, 5.0)     # acceptable dissection-ratio (hg/lg) band
lg_bounds = (15, 50)           # acceptable geomorphic length scale [m]
# rows outside either band get K, D redrawn -- see sample_K_D_constrained

bnds = list(zip(bnd_logK, bnd_logD, bnd_logb, bnd_logT, bnd_logds, bnd_logrho))

sampler = qmc.LatinHypercube(d=len(bnds[0]), seed=2023)  # K, D, b, T, ds, rho
sample = sampler.random(n=len(dfc))
scaled_sample = qmc.scale(sample, bnds[0], bnds[1])

v0 = 30  # contour width (also grid spacing) [m]
sc = 1.0
theta = 0.0  # don't use threshold model
ne = 0.1  # drainable porosity [-]
na = 0.15
phi = na/ne

U = 10**dfc['log10E'].values / (3600*24*365)  # m/s

K, D, alpha, lg, n_redrawn = sample_K_D_constrained(
    U, v0, bnd_logK, bnd_logD, alpha_bounds, lg_bounds, base_sample=sample[:, :2],
)
print(f'{n_redrawn}/{len(U)} rows needed a K,D redraw to satisfy '
      f'alpha in {alpha_bounds} and lg in {lg_bounds}')

#%%

hg = hg_fun(K, D, U, v0)
tg = tg_fun(K, D, v0)

b = 10**scaled_sample[:, 2]  # regolith thickness, independent of hg (comparison variant)

T = 10**scaled_sample[:, 3]
ds = 10**scaled_sample[:, 4]
rho = 10**scaled_sample[:, 5]
ksat = T/b

p = dfc['P'].values / (3600*24*365)    # m/s
PET = dfc['PET'].values / (3600*24*365)  # m/s
pet = PET/(1-rho)
ai = pet/p
tr = rho * (ds/p)
tb = (1-rho) * (ds/p)

#%%

beta = beta_fun(hg, lg, ksat, p)
gamma = gamma_fun(hg, lg, ksat, p, b)
sigma = sigma_fun(p, b, ne, tr, tb)
E0 = E0_fun(theta, hg, tg)

#%%

Tg_nd = 500        # total duration in units of tg [-]
dtg_nd = 1e-2       # target outer (morphologic-scaling) geomorphic timestep, in units of tg [-]
                    # equivalent to an uplift-per-step target of dtg_nd*hg
dtg_max_nd = 2e-3   # maximum geomorphic substep, in units of tg [-]; subdivides dtg for
                    # numerical accuracy (~dtg_nd/dtg_max_nd substeps per outer step)
Th_nd = 25          # hydrologic time in units of (tr+tb) [-]

bin_capacity_nd = 0.01  # bin capacity as a proportion of mean storm depth
Nx = 200  # number of grid cells width and height

df_params = pandas.DataFrame({
    'K': K, 'D': D, 'U': U, 'ksat': ksat, 'p': p, 'pet': pet, 'b': b, 'ne': ne,
    'na': na, 'v0': v0, 'hg': hg, 'lg': lg, 'tg': tg, 'E0': E0, 'ds': ds, 'tr': tr,
    'tb': tb, 'alpha': alpha, 'gam': gamma, 'beta': beta, 'sigma': sigma, 'rho': rho,
    'ai': ai, 'theta': theta, 'phi': phi,
})
df_params['Sc'] = sc
df_params['Nz'] = round((df_params['b']*df_params['na'])/(bin_capacity_nd*df_params['ds']))
df_params['Nx'] = Nx
df_params['td'] = (df_params['lg']*df_params['ne'])/(df_params['ksat']*df_params['hg']/df_params['lg'])  # characteristic aquifer drainage time [s]
df_params['ha'] = (df_params['p']*df_params['lg'])/(df_params['ksat']*df_params['hg']/df_params['lg'])  # characteristic aquifer thickness [m]
df_params['Tg'] = Tg_nd*df_params['tg']  # Total geomorphic simulation time [s]
df_params['Th'] = Th_nd*(df_params['tr']+df_params['tb'])  # hydrologic simulation time [s]
df_params['dtg'] = dtg_nd*df_params['tg']  # target outer geomorphic timestep [s] (= ksf*Th)
df_params['ksf'] = df_params['dtg']/df_params['Th']  # morphologic scaling factor, solved so ksf*Th = dtg
df_params['dtg_max'] = dtg_max_nd*df_params['tg']  # the maximum duration of a geomorphic substep [s]
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

import matplotlib.pyplot as plt

logU = np.log10(df_params['U'])
print('corr(logU, logK):     ', np.corrcoef(logU, np.log10(df_params['K']))[0, 1])
print('corr(logU, logD):     ', np.corrcoef(logU, np.log10(df_params['D']))[0, 1])
print('corr(logU, log hg):   ', np.corrcoef(logU, np.log10(df_params['hg']))[0, 1])
print('corr(logU, log alpha):', np.corrcoef(logU, np.log10(df_params['alpha']))[0, 1])

plt.figure()
plt.scatter(df_params['U'], df_params['K'])
plt.xscale('log'); plt.yscale('log')
plt.xlabel('U'); plt.ylabel('K')
plt.show()

plt.figure()
plt.scatter(df_params['U'], df_params['D'])
plt.xscale('log'); plt.yscale('log')
plt.xlabel('U'); plt.ylabel('D')
plt.show()

plt.figure()
plt.scatter(df_params['U'], df_params['hg'])
plt.xscale('log'); plt.yscale('log')
plt.xlabel('U'); plt.ylabel('hg')
plt.show()

plt.figure()
plt.scatter(df_params['U'], df_params['alpha'])
plt.xscale('log'); plt.yscale('log')
plt.xlabel('U'); plt.ylabel('alpha')
plt.show()

plt.figure()
plt.scatter(df_params['ksat'], df_params['K'])
plt.xscale('log'); plt.yscale('log')
plt.xlabel('ksat'); plt.ylabel('K')
plt.show()

plt.figure()
plt.scatter(df_params['pet'], df_params['U'])
plt.xscale('log'); plt.yscale('log')
plt.xlabel('pet'); plt.ylabel('U')
plt.show()

#%%
plt.figure()
plt.scatter(df_params['ksat'], df_params['U'])
plt.xscale('log'); plt.yscale('log')
plt.xlabel('ksat'); plt.ylabel('U')
plt.show()

plt.figure()
plt.scatter(df_params['b'], df_params['U'])
plt.xscale('log'); plt.yscale('log')
plt.xlabel('b'); plt.ylabel('U')
plt.show()

plt.figure()
plt.scatter(df_params['b']*df_params['ksat'], df_params['U'])
plt.xscale('log'); plt.yscale('log')
plt.xlabel('T'); plt.ylabel('U')
plt.show()

# %%


