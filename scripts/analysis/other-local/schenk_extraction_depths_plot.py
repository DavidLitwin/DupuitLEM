# -*- coding: utf-8 -*-
"""
Created on Tue May 10 09:29:45 2022

@author: dgbli
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import cm

from DupuitLEM.auxiliary_models import SchenkVadoseModel
from DupuitLEM.auxiliary_models.schenk_analytical_solutions import extraction_pdf, extraction_cdf

#%% One example

profile_depth = np.linspace(0.0, 5, 500)
Sawc = 1.0
tr = 15000
tb = 430000
ds = 0.015
pet = 1.5e-7

cdf = extraction_cdf(profile_depth, ds, tb, pet, Sawc)
pdf = extraction_pdf(profile_depth, ds, tb, pet, Sawc)

#%% numerical and analytical solutions with varying Ai and Sigma

directory = 'C:/Users/dgbli/Documents/Research Data/HPC output/DupuitLEMResults/post_proc'
base_output_path = 'stoch_ai_sigma_4' #'stoch_gam_sigma_9' #
model_runs = np.arange(35)

dfs = []
for ID in model_runs:
    df = pd.read_csv('%s/%s/params_ID_%d.csv'%(directory,base_output_path, ID), index_col=0)
    dfs.append(df)
df_params = pd.concat(dfs, axis=1, ignore_index=True).T
df_params.to_csv('%s/%s/params.csv'%(directory,base_output_path), index=True, float_format='%.3e')


Nz = 10000 
extraction_profiles = np.zeros((Nz, 35))
recharge_profiles = np.zeros((Nz, 35))

extraction_pdfs = np.zeros((Nz, 35))

rooting_pdfs = np.zeros((Nz, 35))
rooting_cdfs = np.zeros((Nz, 35))

extraction_cdfs = np.zeros((Nz, 35))
extraction_cdfs_analytical = np.zeros((Nz, 35))
for ID in range(35):
    
    b = df_params['b'][ID] #characteristic depth  [m]
    pet = df_params['pet'][ID]
    na = df_params['na'][ID] #plant available volumetric water content
    tr = df_params['tr'][ID] #mean storm duration [s]
    tb = df_params['tb'][ID] #mean interstorm duration [s]
    ds = df_params['ds'][ID] #mean storm depth [m]
    
    svm = SchenkVadoseModel(
                    potential_evapotranspiration_rate=pet,
                     available_water_content=na,
                     profile_depth=b,
                     num_bins=Nz,
                     )
    svm.generate_state_from_analytical(ds, tb, random_seed=20220408)
    svm.run_model(num_timesteps=1000,
                  mean_storm_depth=ds,
                  mean_storm_duration=tr,
                  mean_interstorm_duration=tb,
                  random_seed=101
                  )
    
    rooting_pdfs[:,ID] = svm.plant_rooting_pdf
    extraction_pdfs[:,ID] = extraction_pdf(svm.depths, ds, tb, pet, na)
    # extraction_profiles[:,ID] = svm.cum_extraction/svm.cum_interstorm_dt
    # recharge_profiles[:,ID] = svm.cum_recharge/svm.cum_storm_dt
    
    
    extraction_cdfs[:,ID] = np.cumsum(extraction_pdfs[:,ID]*np.diff(svm.depths)[0])
    extraction_cdfs_analytical[:,ID] = extraction_cdf(svm.depths, ds, tb, pet, na)
    rooting_cdfs[:,ID] = np.cumsum(rooting_pdfs[:,ID]*np.diff(svm.depths)[0])



#%% PDF of rooting and analytical solution

viridis = cm.get_cmap('viridis', len(np.unique(df_params.sigma)))
colors = viridis.colors


ai_all = np.unique(df_params.ai)
fig, ax = plt.subplots()
j = 0
for i in range(35):
    
    if df_params['ai'][i] == ai_all[-1]:
        ax.plot(extraction_pdfs[:,i], svm.depths, linestyle='--', color=colors[j], label='analytical')
        ax.plot(rooting_pdfs[:,i], svm.depths, linestyle='-', color=colors[j], label='numerical')
        j+=1
        
ax.set_xlabel('Root water uptake density')           
ax.set_ylabel('Depth')
ax.set_ylim((1.8,0))
fig.legend(frameon=False)

#%% CDF of rooting and analytical solution

ai_all = np.unique(df_params.ai)
fig, ax = plt.subplots()
j = 0
for i in range(35):
    
    if df_params['ai'][i] == ai_all[0]:
        if j==0:
            ax.plot(extraction_cdfs_analytical[:,i], svm.depths, linestyle='--', color=colors[j], label='analytical')
            ax.plot(rooting_cdfs[:,i], svm.depths, linestyle='-', color=colors[j], label=r'$\sigma$=%.2f'%df_params['sigma'][i])
        else:
            ax.plot(extraction_cdfs_analytical[:,i], svm.depths, linestyle='--', color=colors[j])
            ax.plot(rooting_cdfs[:,i], svm.depths, linestyle='-', color=colors[j], label=r'$\sigma$=%.2f'%df_params['sigma'][i])
        j+=1
         
ax.set_xlabel('Root water uptake cdf')           
ax.set_ylabel('Depth')
ax.set_ylim((1.8,0))
ax.legend(frameon=False)


    
#%% recharge and extraction

ai_all = np.unique(df_params.ai)


fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
for i in range(35):
    
    if df_params['ai'][i] == ai_all[2]:
        axs[0].plot(extraction_profiles[:,i], svm.depths, linestyle='--')
        axs[1].plot(recharge_profiles[:,i], svm.depths, linestyle='--')
        
    elif df_params['ai'][i] == ai_all[-1]:
        axs[0].plot(extraction_profiles[:,i], svm.depths, linestyle='-')
        axs[1].plot(recharge_profiles[:,i], svm.depths, linestyle='-')
        
axs[0].set_xlabel('Average root water uptake (m/s)')     
axs[1].set_xlabel('Average Recharge (m/s)')        
axs[0].set_ylabel('Depth')
axs[0].set_ylim((1.8,0))
axs[1].set_ylim((1.8,0))


#%% try setting extraction depth mask

ID = 0

b = df_params['b'][ID] #characteristic depth  [m]
pet = df_params['pet'][ID]
na = df_params['na'][ID] #plant available volumetric water content
tr = df_params['tr'][ID] #mean storm duration [s]
tb = df_params['tb'][ID] #mean interstorm duration [s]
ds = df_params['ds'][ID] #mean storm depth [m]

svm = SchenkVadoseModel(
                potential_evapotranspiration_rate=pet,
                 available_water_content=na,
                 profile_depth=b,
                 num_bins=Nz,
                 )
svm.generate_state_from_analytical(ds, tb, random_seed=20220408)
cdf = svm.set_max_extraction_depth(ds,tb,threshold=0.05)
mask = svm.extraction_depth_mask
svm.run_model(num_timesteps=5000,
              mean_storm_depth=ds,
              mean_storm_duration=tr,
              mean_interstorm_duration=tb,
              random_seed=101
              )
rooting_pdf = svm.plant_rooting_pdf
extraction = svm.cum_extraction