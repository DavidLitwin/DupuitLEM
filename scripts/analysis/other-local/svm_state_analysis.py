# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 14:09:31 2022

@author: dgbli

Schenk Vadose - Trying to understand partitioning
"""
import numpy as np
import pandas
from itertools import product
import matplotlib.pyplot as plt

from DupuitLEM.auxiliary_models import SchenkVadoseModel


#dim equations
def b_fun(hg, gam, hi):
    return (hg*gam)/hi

def ksat_fun(p, hg, lg, hi):
    return (lg**2*p*hi)/hg**2

def ds_fun(hg, n, gam, sigma, hi):
    return (hg*n*gam)/(hi*sigma)

def tr_fun(hg, p, n, gam, sigma, hi, rho):
    return (hg*n*gam*rho)/(p*sigma*hi)

def tb_fun(hg, p, n, gam, sigma, hi, rho):
    return (hg*n*gam)*(1-rho)/(p*sigma*hi)

def generate_parameters(p, n, hg, lg, gam, hi, sigma, rho, ai):

    alpha = hg/lg
    b = b_fun(hg, gam, hi)
    ksat = ksat_fun(p, hg, lg, hi)
    ds = ds_fun(hg, n, gam, sigma, hi)
    tr = tr_fun(hg, p, n, gam, sigma, hi, rho)
    tb = tb_fun(hg, p, n, gam, sigma, hi, rho)
    pet = ai*p

    return ksat, p, pet, b, n, hg, lg, ds, tr, tb, alpha, gam, hi, sigma, rho, ai


directory = 'C:/Users/dgbli/Documents/Research Data/HPC output/DupuitLEMResults/post_proc'
base_output_path =  'stoch_ai_sigma_1'

#%%
# sig = 32
gam = 4
hi = 5.0
hg = 2.25
lg = 15
rho = 0.03
Srange = 0.2
b = 5
n = 0.1
Nz = 500
Nt = 1000
p = 3e-3

# ai = 1.5 #np.linspace(0.2,2,10)[0]
# pet = p*ai

sigma_all = np.geomspace(8.0, 128.0, 5)
ai_all = np.geomspace(0.25, 2.0, 7)

params = []
for sigma, ai in product(sigma_all, ai_all):
    params.append(generate_parameters(p, n, hg, lg, gam, hi, sigma, rho, ai))

df_params = pandas.DataFrame(np.array(params),columns=['ksat', 'p', 'pet', 'b', 'n', 'hg', 'lg', 'ds', 'tr', 'tb', 'alpha', 'gam', 'hi', 'sigma', 'rho', 'ai'])

max_recharge = np.zeros(len(df_params))
min_recharge = np.zeros(len(df_params))
max_extraction = np.zeros(len(df_params))
min_extraction = np.zeros(len(df_params))
precip = np.zeros(len(df_params))
for i in df_params.index:
    print(i)

    svm = SchenkVadoseModel(potential_evapotranspiration_rate=df_params.pet[i],
                            available_relative_saturation=Srange,
                            profile_depth=df_params.b[i],
                            porosity=df_params.n[i],
                            num_bins=int(Nz),
    )
    
    svm.run_model(num_timesteps=Nt,
                  mean_storm_depth=df_params.ds[i],
                  mean_storm_duration=df_params.tr[i],
                  mean_interstorm_duration=df_params.tb[i],
                  random_seed=14032022
                  )
    
    max_recharge[i] = svm.cum_recharge[0]
    min_recharge[i] = svm.cum_recharge[-1]
    max_extraction[i] = svm.cum_extraction[0]
    min_extraction[i] = svm.cum_extraction[-1]
    precip[i] = svm.cum_precip

#%%
plt.figure()
plt.scatter(df_params.ai, (precip-max_recharge)/precip, c=df_params.sigma, marker='s', label='Max WT' )
plt.scatter(df_params.ai, (precip-min_recharge)/precip, c=df_params.sigma, marker='o', label='Min WT')
plt.plot([0,1], [0,1], 'k--')
plt.plot([1,2], [1,1], 'k--')
plt.legend(frameon=False)
plt.colorbar(label='Sigma')
plt.xlabel('Ai')
plt.ylabel('(P-R)/P')
plt.title('Budyko Without Extraction')
plt.savefig('%s/%s/budyko_no_extraction.png'%(directory, base_output_path), dpi=300)



plt.figure()
plt.scatter(df_params.ai, (precip+np.minimum(-max_recharge-max_extraction,0))/precip, c=df_params.sigma, marker='s', label='Max WT' )
plt.scatter(df_params.ai, (precip+np.minimum(-max_recharge-max_extraction,0))/precip, c=df_params.sigma, marker='o', label='Min WT')
plt.plot([0,1], [0,1], 'k--')
plt.plot([1,2], [1,1], 'k--')
plt.legend(frameon=False)
plt.colorbar(label='Sigma')
plt.xlabel('Ai')
plt.ylabel('(P-R+E)/P')
plt.title('Budyko With Extraction')
plt.savefig('%s/%s/budyko_extraction.png'%(directory, base_output_path), dpi=300)

plt.figure()
plt.scatter(df_params.ai, (precip+np.minimum(-max_recharge-min_extraction,0))/precip, c=df_params.sigma, marker='s', label='Max WT storm, Min WT interstorm' )
plt.scatter(df_params.ai, (precip+np.minimum(-min_recharge-max_extraction,0))/precip, c=df_params.sigma, marker='o', label='Min WT storm, Max WT interstorm')
plt.plot([0,1], [0,1], 'k--')
plt.plot([1,2], [1,1], 'k--')
plt.legend(frameon=False)
plt.colorbar(label='Sigma')
plt.xlabel('Ai')
plt.ylabel('(P-R+E)/P')
plt.ylim(-0.05,1.3)
plt.title('Budyko With Extraction')
plt.savefig('%s/%s/budyko_extraction_min_max.png'%(directory, base_output_path), dpi=300)

#%%

plt.figure()
plt.plot(np.cumsum(svm.precip_all), label='precip')
plt.plot(np.cumsum(svm.unsat_recharge_all), label='recharge')
plt.plot(np.cumsum(svm.runoff_all), label='runoff')
plt.plot(np.cumsum(svm.et_all), label='ET')
plt.legend()




