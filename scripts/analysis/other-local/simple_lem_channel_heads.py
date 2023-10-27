
#%%
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from matplotlib import colors

from landlab.io.netcdf import from_netcdf, to_netcdf
from landlab.components import FlowAccumulator

base_output_path = 'simple_lem_Nx_1'
directory = '/Users/dlitwin/Documents/Research Data/HPC output/DupuitLEMResults/post_proc/'
id_range = range(4)

df_params = pd.read_csv(directory+base_output_path+'/parameters.csv')

#%%

i = 4
hg = df_params['hg'][i]
lg = df_params['lg'][i]

grid = from_netcdf('%s/%s/grid_%d.nc'%(directory, base_output_path, i))
elev = grid.at_node['topographic__elevation']
elev_star = elev/hg
dx_star = grid.dx/lg
y = np.arange(grid.shape[0] + 1) * dx_star - dx_star * 0.5
x = np.arange(grid.shape[1] + 1) * dx_star - dx_star * 0.5

# slope area 
S = grid.at_node['slope_D8']
A = grid.at_node['drainage_area']
 
# make dimensionless versions
a_star = (A/grid.dx)/lg
S_star = S*lg/hg

# cheads = np.log(a_star/S_star*(1+0.5*np.sqrt(a_star)*S_star))
# cheads[np.isinf(cheads)] = np.nan

K = df_params['K'][i]
D = df_params['D'][i]
U = df_params['U'][i]
v0 = df_params['v0'][i]
a = A/grid.dx

cheads =  - 5/2 * K * np.sqrt(v0 * a) * S + D * S/a + U
# cheads = - D * S/a + U
cheads[np.isinf(cheads)] = np.nan

# plot single hillshade
ls = LightSource(azdeg=135, altdeg=45)
plt.figure()
plt.imshow(ls.hillshade(elev_star.reshape(grid.shape).T, vert_exag=2, dx=dx_star, dy=dx_star), 
            origin="lower", 
            extent=(x[0], x[-1], y[0], y[-1]), 
            cmap='gray')
md = plt.imshow(cheads.reshape(grid.shape).T, 
            origin="lower", 
            extent=(x[0], x[-1], y[0], y[-1]), 
            cmap='seismic', 
            norm=colors.CenteredNorm(0.0),
            alpha=0.8)
plt.colorbar(md, label=r'$-\frac{5}{2}K\sqrt{v_0a}S - D\frac{S}{a}+ U$')
plt.ylabel(r'$y/\ell_g$')
plt.xlabel(r'$x/\ell_g$')
# plt.savefig('%s/hillshade_%s.png'%(save_directory, base_output_path), dpi=300) 
# %%
