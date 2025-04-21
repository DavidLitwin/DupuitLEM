
#%%
import numpy as np
import pandas as pd
import statsmodels.api as sm 

from matplotlib import cm, colors, ticker
import matplotlib.pyplot as plt
from landlab.io.netcdf import from_netcdf
# plt.rc('text', usetex=True)

from generate_colormap import get_continuous_cmap

mac_path = '/Users/dlitwin'
pc_path = 'C:/Users/dgbli'

base_path = mac_path

directory = f'{base_path}/Documents/Research Data/HPC output/DupuitLEMResults/post_proc'

def calc_mean_relief(grid):
    elev = grid.at_node['topographic__elevation']
    x = np.unique(grid.x_of_node)
    varline = np.zeros(len(x))
    
    for i, xi in enumerate(x):
        row = np.where(grid.x_of_node == xi)[0][1:-1]
        zrow = elev[row]
        varline[i] = np.var(zrow)
    
    return np.sqrt(np.nanmean(varline))

def calc_horiz_elev_change(grid):
    elev = grid.at_node['topographic__elevation']
    x = np.unique(grid.x_of_node)
    
    rowf = np.where(grid.x_of_node == x[1])[0][1:-1]
    rowl = np.where(grid.x_of_node == x[-2])[0][1:-1]
    return np.nanmin(elev[rowl]) - np.nanmin(elev[rowf])

#%% load stats from DR and BR

df_qstats = pd.read_csv(f'{base_path}/Documents/Papers/Ch3_oregon_ridge_soldiers_delight/df_qstats.csv', index_col=0).T
df_rstats = pd.read_csv(f'{base_path}/Documents/Papers/Ch3_oregon_ridge_soldiers_delight/df_Relief_stats.csv', index_col=0)

df_sat_DR = pd.read_csv(f'{base_path}/Documents/Papers/Ch3_oregon_ridge_soldiers_delight/df_sat_DR.csv', index_col=0)
df_sat_BR = pd.read_csv(f'{base_path}/Documents/Papers/Ch3_oregon_ridge_soldiers_delight/df_sat_BR.csv', index_col=0)

df_params_DR = pd.read_csv(f'{base_path}/Documents/Research Data/HPC output/DupuitLEMResults/CaseStudy/CaseStudy_cross_2-0/parameters.csv', index_col=0)['0']
df_params_BR = pd.read_csv(f'{base_path}/Documents/Research Data/HPC output/DupuitLEMResults/CaseStudy/CaseStudy_cross_2-1/parameters.csv', index_col=0)['1']

hg = np.array([df_params_DR['hg'],df_params_BR['hg']])
v0 = np.array([df_params_DR['v0'],df_params_BR['v0']])

df_qstats['Qf/Q'] = 1 - df_qstats['Qb']/df_qstats['Q']
df_qstats['R/hg'] = df_rstats['q50']/hg
df_qstats['sat_var'] = [df_sat_DR.loc['sat_variable'][0],df_sat_BR.loc['sat_variable'][0]]

#%% Dunne discussion -- get relief for all relevant model runs

base_output_path_2 = 'CaseStudy_cross_15'
names = ['stoch_gam_sigma_14', 'stoch_gam_sigma_15', 'stoch_gam_sigma_16', base_output_path_2]
num_runs = [25, 25, 25, 4]

results_dict = {}
params_dict = {}
for k, base_output_path in enumerate(names):

    plot_runs = np.arange(num_runs[k])

    # read existing
    df_results = pd.read_csv('%s/%s/results.csv'%(directory,base_output_path))
    df_params = pd.read_csv('%s/%s/params.csv'%(directory,base_output_path))

    # topography
    mean_r_nd = np.zeros(len(plot_runs))
    mean_r_trend = np.zeros(len(plot_runs))
    for i in plot_runs:
        try:
            grid = from_netcdf('%s/%s/grid_%d.nc'%(directory, base_output_path, i))
            elev = grid.at_node['topographic__elevation']
            r = elev - np.nanmin(elev)
            hmean = np.mean(grid.at_node['wtrel_mean_end_interstorm'][grid.core_nodes])*df_params['b'][i]
            
            ## case 0 (Relief/hg)
            # rnd = r/df_params['hg'][i]
            # mean_r_nd[i] = np.mean(rnd[grid.core_nodes])
            mean_r_nd[i] = calc_mean_relief(grid)/df_params['hg'][i]
            mean_r_trend[i] = calc_horiz_elev_change(grid)/df_params['hg'][i]

        except:
            mean_r_nd[i] = np.nan
    
    df_results['mean_r_nd'] = mean_r_nd
    df_results['mean_r_trend'] = mean_r_trend

    results_dict[base_output_path] = df_results
    params_dict[base_output_path] = df_params

#%% Dunne discussion -- get LSDTT relief for CaseStudy model runs


path_2 = f'{base_path}/Documents/Research Data/HPC output/DupuitLEMResults/post_proc/{base_output_path_2}/lsdtt/'

files_ht = ["%s-%d_HilltopData_TN.csv"%(base_output_path_2, i) for i in range(4)]
dfs_ht = [pd.read_csv(path_2 + file_ht) for file_ht in files_ht]
names_ht = ['DR-DR', 'BR-BR', 'DR-BR', 'BR-DR'] # in order

Lh = [df['Lh'].values for df in dfs_ht ]
R = [df['R'].values for df in dfs_ht ]

hg = params_dict[base_output_path_2]['hg']
lg = params_dict[base_output_path_2]['lg']

# df_lsdtt_case = pd.DataFrame(index=names_ht)
# df_lsdtt_case['Lh q1'] = np.array([np.percentile(lh, 25) for lh in Lh])/lg
# df_lsdtt_case['Lh q3'] = np.array([np.percentile(lh, 75) for lh in Lh])/lg
# df_lsdtt_case['Lh med'] = np.array([np.percentile(lh, 50) for lh in Lh])/lg
# df_lsdtt_case['R q1'] = np.array([np.percentile(r, 25) for r in R])/hg
# df_lsdtt_case['R q3'] = np.array([np.percentile(r, 75) for r in R])/hg
# df_lsdtt_case['R med'] = np.array([np.percentile(r, 50) for r in R])/hg
rnd_casestudy = np.array([np.percentile(r, 50) for r in R])/hg


#%% Dunne discussion - make plot with DR and BR as well

plt.rc('axes', labelsize=14) 
fig, ax = plt.subplots(figsize=(6,4.2))

# plot ch.2 results
main_names = ['stoch_gam_sigma_14', 'stoch_gam_sigma_15', 'stoch_gam_sigma_16']
for k, base_output_path in enumerate(main_names):

    dfr = results_dict[base_output_path]
    sc = ax.scatter(1-dfr['Qb/Q'],
                dfr['mean_r_nd'],
                c='0.75',
                s=25,
                # c=df_results['sat_variable'],
                # cmap='plasma',vmin=0.0, vmax=1.0,
                )

# plot case study modeled results
# df_results = results_dict['CaseStudy_cross_2']
# df_params = params_dict['CaseStudy_cross_2']
# sc = ax.scatter(1-df_results['Qb/Q'],
#             df_results['mean_r_nd'],
#             # c=df_results['sat_variable'],
#             # cmap='plasma',vmin=0.0,vmax=1.0,
#             label='Modeled: Var',
#             marker='P',
#             )

# plot case study modeled results - LSDTT elevation
dfr = results_dict[base_output_path_2]
sc = ax.scatter(1-dfr['Qb/Q'][0],
            rnd_casestudy[0],
            label='Modeled: DR',
            marker='^',
            color='firebrick'
            )
sc = ax.scatter(1-dfr['Qb/Q'][1],
            rnd_casestudy[1],
            label='Modeled: BR',
            color='royalblue',
            marker='^',
            )

# plot based on quantities derived from data at DR and BR
ax.scatter(df_qstats['Qf/Q']['DR'], 
            df_qstats['R/hg']['DR'], 
            s=50, marker='s', color='firebrick',
            label='Field: DR') 
ax.scatter(df_qstats['Qf/Q']['BR'], 
            df_qstats['R/hg']['BR'], 
            s=50, marker='s', color='royalblue',
            label='Field: BR')   
     
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'$ \langle Q_f \rangle / \langle Q \rangle$')
ax.set_ylabel(r'$\overline{Z} / h_g$')
ax.legend(frameon=False)
fig.tight_layout()
# plt.savefig('%s/%s/relief_qfi_hg_sat.pdf'%(directory, base_output_path), dpi=300) #_sat
# plt.savefig('/Users/dlitwin/Documents/Papers/Ch3_oregon_ridge_soldiers_delight/relief_qfi_hg_sat.pdf', dpi=300)


#%% get all hillslope numbers etc from existing model runs

names = ['stoch_gam_sigma_14', 'stoch_gam_sigma_15', 'stoch_gam_sigma_16']
plot_runs = np.arange(25)
dfs_r = []
dfs_p = []
fig, ax = plt.subplots(figsize=(6,4.2))
for k, base_output_path in enumerate(names):


    for ID in plot_runs:
        try:
            df_r = pd.read_csv('%s/%s/output_ID_%d.csv'%(directory,base_output_path, ID), index_col=0)
            df_p = pd.read_csv('%s/%s/params_ID_%d.csv'%(directory,base_output_path, ID), index_col=0)

            grid = from_netcdf('%s/%s/grid_%d.nc'%(directory, base_output_path, ID))
            elev = grid.at_node['topographic__elevation']
            r = elev - np.nanmin(elev)
            hmean = np.mean(grid.at_node['wtrel_mean_end_interstorm'][grid.core_nodes])*df_p.loc['b'][0]

            ## case 1 (relief/ha)
            df_r.loc['hillslope_num'] = calc_mean_relief(grid)/hmean
            dfs_r.append(df_r)
            dfs_p.append(df_p)

        except FileNotFoundError:
            pass
        
df_pall = pd.concat(dfs_p, axis=1, ignore_index=True).T
df_rall = pd.concat(dfs_r, axis=1, ignore_index=True).T

#%% model hillslope number with dimensionless parameters (hi -> beta)

df_pall_log = np.log(df_pall)
hsn_log = np.log(df_rall['hillslope_num'])

X = df_pall_log[['gam', 'hi', 'sigma']] 
y = hsn_log
## fit a OLS model with intercept on TV and Radio 
X = sm.add_constant(X) 
est = sm.OLS(y, X).fit() 
pred = est.predict(X)
est.summary()

#%% plot observed predicted using the regression model

# pred_alt = np.exp(est.params[0])*df_pall['gam']**(1/3)*df_pall['hi']**(4/3)*df_pall['sigma']**(1/2)
pred_alt = 4*df_pall['gam']**(1/3)*df_pall['hi']**(4/3)*df_pall['sigma']**(1/2)

fig, ax = plt.subplots(figsize=(5,4))
ax.scatter(np.exp(pred), df_rall['hillslope_num'], color='k', alpha=0.4, label='Model Fit')
ax.scatter(pred_alt, df_rall['hillslope_num'], color='b', alpha=0.5, label=r'$4\gamma^{1/3}\,\beta^{4/3}\,\sigma^{1/2}$')
ax.axline([0,0],[1,1], linestyle='--', color='k')
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_ylabel('Observed Hi', fontsize=14)
ax.set_xlabel('Predicted Hi', fontsize=14)
ax.legend(fontsize=12)
fig.tight_layout()
plt.savefig(f'{base_path}/Documents/Papers/Ch2_stochastic_gw_landscape_evolution/predict_hi.png', dpi=300)
plt.savefig(f'{base_path}/Documents/Papers/Ch2_stochastic_gw_landscape_evolution/predict_hi.pdf', dpi=300)


#%% plot observed predicted with exponents of 1

fig, ax = plt.subplots()
ax.scatter(df_pall['hi']*(df_pall['gam'])*df_pall['sigma'], df_rall['hillslope_num'], alpha=0.3)
ax.axline([0,0],[1,1], linestyle='--', color='k')
ax.set_yscale('log')
ax.set_xscale('log')

#%%

############ Same but with Latin Hypercube sampled runs


base_output_path = 'stoch_lhs_3'
plot_runs = np.arange(50)

plt.rc('axes', labelsize=14) 

# results
dfs = []
for ID in plot_runs:
    try:
        df = pd.read_csv('%s/%s/output_ID_%d.csv'%(directory,base_output_path, ID), index_col=0)
    except FileNotFoundError:
        df =  pd.DataFrame(columns=df.columns)
    dfs.append(df)
df_results = pd.concat(dfs, axis=1, ignore_index=True).T

# parameters
dfs = []
for ID in plot_runs:
    df = pd.read_csv('%s/%s/params_ID_%d.csv'%(directory,base_output_path, ID), index_col=0)
    dfs.append(df)
df_params = pd.concat(dfs, axis=1, ignore_index=True).T
df_params.to_csv('%s/%s/params.csv'%(directory,base_output_path), index=True, float_format='%.3e')

# get dtypes right
for ind in df_params.columns:
    try:
        df_params[ind] = df_params[ind].astype(float)
    except ValueError:
        df_params[ind] = df_params[ind].astype(str)

mean_h = np.zeros(len(plot_runs))
mean_r = np.zeros(len(plot_runs))
mean_r_nd = np.zeros(len(plot_runs))
mean_r_trend = np.zeros(len(plot_runs))
for i in plot_runs:
    try:
        grid = from_netcdf('%s/%s/grid_%d.nc'%(directory, base_output_path, i))
        elev = grid.at_node['topographic__elevation']
        r = elev - np.nanmin(elev)
        mean_h[i] = np.mean(grid.at_node['wtrel_mean_end_interstorm'][grid.core_nodes])*df_params['b'][i]
        
        mean_r[i] = calc_mean_relief(grid)
        mean_r_nd[i] = calc_mean_relief(grid)/df_params['hg'][i]
        mean_r_trend[i] = calc_horiz_elev_change(grid)/df_params['hg'][i]

    except:
        mean_r_nd[i] = np.nan
df_results['mean_r'] = mean_r
df_results['mean_h'] = mean_h

#%% Dunne discussion - gam sigma multi-run  

fig, ax = plt.subplots(figsize=(6,4.2))

sc = ax.scatter((1-df_results['Qb/Q']),#/(1- df_results['Q/P']),
            df_results['mean_r']/df_params['hg'],
            c=df_results['sat_variable'],
            # s= df_params['rho']**2*500, #df_params['alpha']**2*1000,
            vmin=0.0,
            vmax=1.0,
            cmap='plasma')
ax.set_xscale('log')
ax.set_yscale('log')
# ax.set_xlabel(r'$ \left(\langle Q_f \rangle / \langle Q \rangle\right) / \left(\langle AET\rangle / \langle P \rangle \right)$')
ax.set_xlabel(r'$\langle Q_f \rangle / \langle Q \rangle$')
ax.set_ylabel(r'$\overline{Z} / h_g$')
ax.set_title(r'Latin Hypercube: $\alpha,\, \beta,\, \gamma,\, \sigma,\,$Ai')
# ax.set_title(r'Latin Hypercube: $\alpha,\, \beta,\, \gamma,\, \sigma,\, \rho,\, \phi,\, $Ai')
fig.colorbar(sc, label='Sat Variable')
fig.tight_layout()
# plt.savefig('%s/%s/relief_qfi_hg_sat.pdf'%(directory, base_output_path), dpi=300) #_sat


#%% baseflow fraction 

fig, ax = plt.subplots(figsize=(6,4.2))

sc = ax.scatter((1- df_results['Qb/Q'])/(1- df_results['Q/P']),
            df_results['mean_r']/df_params['hg'],
            c=df_results['sat_variable'],
            # s= df_params['rho']**2*500, #df_params['alpha']**2*1000,
            vmin=0.0,
            vmax=1.0,
            cmap='plasma')
ax.set_xscale('log')
ax.set_yscale('log')
# ax.set_xlabel(r'$ \left(\langle Q_b \rangle / \langle Q \rangle\right) / \left(\langle AET\rangle / \langle P \rangle \right)$')
ax.set_xlabel(r'$\langle Q_b \rangle / \langle Q \rangle$')
ax.set_ylabel(r'$\overline{Z} / h_g$')
# ax.set_title(r'Latin Hypercube: $\alpha,\, \beta,\, \gamma,\, \sigma,\,$Ai')
ax.set_title(r'Latin Hypercube: $\alpha,\, \beta,\, \gamma,\, \sigma,\, \rho,\, \phi,\, $Ai')
fig.colorbar(sc, label='Sat Variable')
fig.tight_layout()
# plt.savefig('%s/%s/relief_qfi_hg_sat.pdf'%(directory, base_output_path), dpi=300) #_sat

# %%

fig, ax = plt.subplots(figsize=(6,4.2))

sc = ax.scatter(df_results['sat_variable'],
            # df_results['mean_r']/df_results['mean_h']/df_params['hi'],
            df_results['mean_r']/df_params['hg'],
            c=df_params['ai'],
            # s= df_params['rho']**2*500, #df_params['alpha']**2*1000,
            vmin=0.0,
            vmax=1.0,
            cmap='plasma')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Sat Variable')
# ax.set_ylabel(r'$\left(\overline{Z} / \langle h \rangle \right) / \beta$')
ax.set_ylabel(r'$\overline{Z} / h_g$')
ax.set_title(r'Latin Hypercube: $\alpha,\, \beta,\, \gamma,\, \sigma,\,$Ai')
fig.colorbar(sc, label='Aridity')
fig.tight_layout()
# plt.savefig('%s/%s/relief_sat_hg.pdf'%(directory, base_output_path), dpi=300) #_sat

# %%
