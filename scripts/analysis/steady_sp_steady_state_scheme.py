"""
Scheme to get to steady state topography:
- load completed steady sp results
- run HydrologySteadyStreamPower to find Q
- run a simple model until steady state without changing Q
- check by running the steady sp model again
"""

import os
import glob
import numpy as np
import pickle
import pandas as pd

from landlab.io.netcdf import read_netcdf, from_netcdf, to_netcdf

from landlab import RasterModelGrid, HexModelGrid
from landlab.components import (
    GroundwaterDupuitPercolator,
    LinearDiffuser,
    FastscapeEroder,
    )
from DupuitLEM.auxiliary_models import (
    HydrologySteadyStreamPower,
    RegolithConstantThickness,
    )
from DupuitLEM import StreamPowerModel

task_id = os.environ['SLURM_ARRAY_TASK_ID']
ID = int(task_id)
base_path = os.environ['BASE_OUTPUT_FOLDER']

try:
    os.mkdir('./data/steady_state')
except:
    print('steady_state dir already exists')
try:
    os.mkdir('./data/postrun')
except:
    print('steady_state dir already exists')

# Load parameters and last grid file from steady sp
df_params = pickle.load(open('./parameters.p','rb'))

grid_files = glob.glob('./data/*.nc')
files = sorted(grid_files, key=lambda x:int(x.split('_')[-1][:-3]))
iteration = int(files[-1].split('_')[-1][:-3])
try:
    mg = from_netcdf(files[-1])
except KeyError:
    mg = read_netcdf(files[-1])
mg.set_status_at_node_on_edges(right=mg.BC_NODE_IS_CLOSED, top=mg.BC_NODE_IS_CLOSED, \
                              left=mg.BC_NODE_IS_FIXED_VALUE, bottom=mg.BC_NODE_IS_CLOSED)

elev = mg.at_node['topographic__elevation']
base = mg.at_node['aquifer_base__elevation']
wt = mg.at_node['water_table__elevation']

Ks = df_params['ksat'][ID] # hydraulic conductivity [m/s]
p = df_params['p'][ID] # recharge rate [m/s]
n = df_params['n'][ID] # drainable porosity [-]
b = df_params['b'][ID] # characteristic depth  [m]
Th = df_params['Th'][ID] # hydrological timestep


############### Run hydrological model to determine Q

gdp = GroundwaterDupuitPercolator(mg,
          porosity=n,
          hydraulic_conductivity=Ks,
          regularization_f=0.01,
          recharge_rate=p,
          courant_coefficient=0.1,
          vn_coefficient = 0.1,
)

hm = HydrologySteadyStreamPower(
        mg,
        groundwater_model=gdp,
        hydrological_timestep=Th,
)

#run model
hm.run_step()

############ Run the simple model to steady state
# parameters
Q = mg.at_node["surface_water__discharge"]

K = df_params['K'][ID]
Ksp = K/p # if discharge field is (Q/sqrt(A)) streampower coeff is K/p
D = df_params['D'][ID]
U = df_params['U'][ID]
T = 100*(1/K) # total time in units of tg = 1/K
dt = 1e-3*(1/K) # timestep in units of tg = 1/K
N = int(T//dt)
output_interval = 1
stop_rate = 1e-3 # max(dzdt)/U rate when we say steady state reached

# initialize model components
dm = LinearDiffuser(mg, linear_diffusivity=D)
sp = FastscapeEroder(mg,
        K_sp=Ksp,
        m_sp=1,
        n_sp=1,
        discharge_field="surface_water_area_norm__discharge",
)
rm = RegolithConstantThickness(mg, equilibrium_depth=b, uplift_rate=U)
calc_rate_of_change = lambda elev, elev0, dtm, N: np.max(abs(elev - elev0)) / (N * dtm)

# run for some large time
for i in range(N):

    # find pits for flow accumulation
    hm.dfr._find_pits()
    if hm.dfr._number_of_pits > 0:
        hm.lmb.run_one_step()

    # run flow accumulation on average_surface_water__specific_discharge
    # use the already instantiated component, even though we're only using
    # the updated drainage_area.
    hm.fa.run_one_step()

    # discharge field with form for Q*: (K/p)(Q/sqrt(A)) = K Q* sqrt(A)
    hm.q_an[:] = Q / np.sqrt(hm.area)

    dm.run_one_step(dt)
    sp.run_one_step(dt)

    # uplift and regolith production
    rm.run_step(dt)


    if i%output_interval==0  or i == max(range(N)):
        print('finished iteration %d'%i)

        # save output at specified interval
        filename = './data/steady_state/%d_grid_%d.nc'%(ID,i)
        to_netcdf(mg, filename, include="at_node:topographic__elevation", format="NETCDF4")

        if i > 0:
            # open previous saved file, find max rate of change
            filename0 = './data/steady_state/%d_grid_%d.nc'%(ID,i-output_interval)
            grid0 = from_netcdf(filename0)
            elev0 = grid0.at_node["topographic__elevation"]
            dzdt = calc_rate_of_change(
                elev, elev0, dt, output_interval
            )

            # stop if rate is met
            if dzdt/U < stop_rate:
                print(
                    "Stopping rate condition met, dzdt = %.4e" % dzdt
                )
                break

if dzdt/U >= stop_rate:
    print("stopping rate not met, dzdt = %.4e" % dzdt)


########## Run full steady sp model again and check rate of change after 2k cycles
MSF = df_params['MSF'][ID]
N = 2002
dtm = MSF*hm.T_h
Tg = N*dtm

output = {}
output["output_interval"] = 500
output["output_fields"] = [
        "at_node:topographic__elevation",
        "at_node:aquifer_base__elevation",
        "at_node:water_table__elevation",
        ]
output["base_output_path"] = './data/postrun/%s'%base_path[0:12]
output["run_id"] = ID #make this task_id if multiple runs

postrun_ss_cond = {}
postrun_ss_cond['stop_at_rate'] = 1e-3*U
postrun_ss_cond['how'] = 'percentile'
postrun_ss_cond['percentile_value'] = 90

mdl = StreamPowerModel(mg,
        hydrology_model=hm,
        diffusion_model=dm,
        erosion_model=sp,
        regolith_model=rm,
        morphologic_scaling_factor=MSF,
        total_morphological_time=Tg,
        verbose=True,
        output_dict=output,
        steady_state_condition=postrun_ss_cond,
)
mdl.run_model()
