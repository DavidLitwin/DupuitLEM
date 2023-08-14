# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 09:23:00 2022

@author: dgbli
"""

import pandas as pd


directory = 'C:/Users/dgbli/Documents/Research Data/HPC output/DupuitLEMResults/post_proc'

dfs_list = []
bops = ['stoch_ai_sigma_10', 'stoch_ai_sigma_11', 'stoch_gam_sigma_14', 'stoch_gam_sigma_15', 'stoch_gam_sigma_16']


for base_output_path in bops:
    df_params = pd.read_csv('%s/%s/params.csv'%(directory,base_output_path))

    df_params['hg (m)'] = df_params['hg']
    df_params['lg (m)'] = df_params['lg']
    df_params['ha (m)'] = df_params['ha']
    df_params['td (s)'] = df_params['td']
    df_params['v0 (m)'] = df_params['v0']
    df_params['ksat (m/s)'] = df_params['ksat']
    df_params['p (m/s)'] = df_params['p']
    df_params['b (m)'] = df_params['b']
    df_params['tr (s)'] = df_params['tr']
    df_params['tb (s)'] = df_params['tb']
    df_params['ds (m)'] = df_params['ds']
    df_params['Th (s)'] = df_params['Th']
    
    df_params['tg (yr)'] = df_params['tg']/(3600*24*365)
    df_params['K (1/yr)'] = df_params['K']*(3600*24*365)
    df_params['D (m2/yr)'] = df_params['D']*(3600*24*365)
    df_params['U (m/yr)'] = df_params['U']*(3600*24*365)
    df_params['Tg (yr)'] = df_params['Tg']/(3600*24*365)
    df_params['dtg (yr)'] = df_params['dtg']/(3600*24*365)
    df_params['beta'] = df_params['hi']
    df_params['gamma'] = df_params['gam']
    df_params['lambda'] = (df_params['Nx']*df_params['v0']/df_params['lg'])
    df_params['delta'] = df_params['td']/df_params['tg']
    df_params['v0/lg'] = df_params['v0']/df_params['lg']
    
    df_params['code'] = ['%s-%d'%(base_output_path,i) for i in df_params.index]
    
    df_new = df_params[[
        'code',
        'alpha',
        'beta',
        'gamma',
        'delta',
        'lambda',
        'sigma',
        'rho',
        'ai',
        'phi',
        'hg (m)',
        'lg (m)',
        'tg (yr)',
        'ha (m)',
        'td (s)',
        'K (1/yr)',
        'D (m2/yr)',
        'U (m/yr)',
        'v0 (m)',
        'Sc',
        'ksat (m/s)',
        'p (m/s)',
        'b (m)',
        'tr (s)',
        'tb (s)',
        'ds (m)',
        'ne',
        'na',
        'ksf',
        'Th (s)',
        'Tg (yr)',
        'dtg (yr)',
        'Nx',
        'Nz',
        'v0/lg',
        ]]

    dfs_list.append(df_new)

df_out = pd.concat(dfs_list)
out_directory = 'C:/Users/dgbli/Documents/Papers/Ch2_stochastic_gw_landscape_evolution'
df_out.to_csv('%s/parameter_table.csv'%(out_directory), index=False)

