"""
Generate parameters for simple streampower model, varying K, D, U, and v0.
Includes generalised characteristic scales (any m, n) from nondimensionalisation.

New Nov. 2023.

"""
#%%
import os
import numpy as np
import pandas as pd
from itertools import product
import matplotlib.pyplot as plt
import matplotlib

# generalised characteristic scales from Litwin et al (2022)
def calc_tg(K, D, U, m, n, v0):
    return ((D**(n-m) * U**(2-2*n))/(v0**(2*m) * K**2))**(1/(m+n))
def calc_hg(K, D, U, m, n, v0):
    return ((D**(n-m) * U**(2+m-n))/(v0**(2*m) * K**2))**(1/(m+n))
def calc_lg(K, D, U, m, n, v0):
    return ((D**n * U**(1-n))/(v0**m * K))**(1/(m+n))

# generalised characteristic scales from T+K (2018)
def calc_tc(K, D, U, m, n):
    return (K**(-2) * D**(n-2*m) * U**(2-2*n))**(1/(n+2*m))
def calc_hc(K, D, U, m, n):
    return (K**(-2) * D**(n-2*m) * U**(2-n+2*m))**(1/(n+2*m))
def calc_lc(K, D, U, m, n):
    return (K**(-1) * D**n * U**(1-n))**(1/(n+2*m))

task_id = os.environ['SLURM_ARRAY_TASK_ID']
ID = int(task_id)


ksn_all = [5, 10, 20]
v0_all = [10, 20, 50]
D_all = np.linspace(2e-3, 2e-2, 5)
m = 0.5
n = 1.0
U = 5e-4

Nx = 400
Ny = 200
dt_nd = 0.05
T_nd = 250

routing_method = 'D8'
r_condition = 0.0 #1e-8


prod = np.array(list(product(ksn_all, D_all, v0_all)))
df_params = pd.DataFrame(prod, columns=['ksn_pred', 'D', 'v0'])

df_params['U'] = U
df_params['m'] = m
df_params['n'] = n
df_params['Sc'] = 0.5

df_params['K'] = df_params['U']/df_params['ksn_pred']**df_params['n']

# generalised characteristic scales
df_params['tg'] = calc_tg(df_params.K, df_params.D, df_params.U, df_params.m, df_params.n, df_params.v0)
df_params['hg'] = calc_hg(df_params.K, df_params.D, df_params.U, df_params.m, df_params.n, df_params.v0)
df_params['lg'] = calc_lg(df_params.K, df_params.D, df_params.U, df_params.m, df_params.n, df_params.v0)

# T+K generalised scales
df_params['lc'] = calc_lc(df_params.K, df_params.D, df_params.U, df_params.m, df_params.n)
df_params['hc'] = calc_hc(df_params.K, df_params.D, df_params.U, df_params.m, df_params.n)
df_params['tc'] = calc_tc(df_params.K, df_params.D, df_params.U, df_params.m, df_params.n)

df_params['lc/v0'] = df_params['lc']/df_params['v0']
df_params['lg/v0'] = df_params['lg']/df_params['v0']
# print('lg/v0:', min(df_params['lg/v0']),max(df_params['lg/v0']) )

# df_params['ksn_pred'] = (df_params['U']/df_params['K'])**(1/df_params['n'])
df_params['ksn_pred_spld'] = df_params['ksn_pred'] * (df_params['lg/v0']+1)**(1/df_params.n)
df_params['CI'] = (df_params.K * df_params.v0**df_params.m * (Nx * df_params.v0)**(df_params.m+df_params.n))/(df_params.D * df_params.U**(1-df_params.n))

# print('ksn:', min(df_params['ksn_pred']),max(df_params['ksn_pred']) )

df_params['Nx'] = Nx
df_params['Ny'] = Ny
df_params['T'] = T_nd * df_params['tg']
df_params['dt'] = dt_nd * df_params['tg']
df_params['routing_method'] = routing_method
df_params['r_condition'] = r_condition
df_params['output_interval'] = 500
df_params['BCs'] = 4141

#%%
df_params.loc[ID].to_csv('parameters.csv', index=True)

# %% plots to test how the results should look

# dfg = df_params.groupby(['m'])

# fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(2,6))
# i = 0
# for m, g in dfg:
    
#     axs[i].scatter(g['lg/v0'], g['lc/v0'], s=3)
#     axs[i].set_title(f'm={m[0]}')
#     axs[i].set_xscale('log')
#     axs[i].set_yscale('log')
#     i+=1
# plt.tight_layout()
# # %%

# dfg = df_params.groupby(['m', 'ksn_pred'])

# cmap = matplotlib.colormaps['Reds']
# colors1 = [cmap(i) for i in [0.3, 0.5, 0.7]]

# fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(9,7))
# ax = axs.flatten()
# i = 0
# for (m, ksn), g in dfg:
#     print(m,ksn)
#     j = 0
#     for v0, g1 in g.groupby('v0'):
#         ax[i].plot(g1['D'], g1['ksn_pred_spld'], linestyle='-', color=colors1[j])
#         ax[i].scatter(g1['D'], g1['ksn_pred_spld'], s=10, label=f'$v_0$={v0:.2f}', color=colors1[j])
#         j+=1
#     ax[i].axhline(ksn, color='k', linestyle='--', label='$k_{sn,pred}$')
#     ax[i].set_title(f'm={m:.1f}, ksn={ksn:.0f}')
#     ax[i].set_yscale('log')
#     ax[i].set_xscale('log')
#     ax[i].set_ylim((4,200))

#     i += 1
# ax[0].legend(frameon=False)
# plt.tight_layout()


# %%
