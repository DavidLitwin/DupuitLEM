"""
Author: David Litwin

26 May 2020
"""
import time
import numpy as np
import pandas as pd

from landlab.io.netcdf import to_netcdf, from_netcdf


class StreamPowerModel:
    """
    Landscape evolution model using the GroundwaterDupuitPercolator in which flow
    that results from overland flow is tracked and averaged to update topography
    using a streampower formulation and FastscapeEroder.
    """

    def __init__(self,
        grid,
        hydrology_model = None,
        diffusion_model = None,
        erosion_model = None,
        regolith_model = None,
        morphologic_scaling_factor = 500,
        total_morphological_time = 1e6*365*24*3600,
        output_dict = None,
        steady_state_condition = None,
        verbose=False,
        ):


        """
        Initialize StreamPowerModel.

        Parameters:
        --------
        hydrology_model: an instance of a DupuitLEM hydrology model, either
            HydrologyEventStreamPower or HydrologySteadyStreamPower.
            default: None
        diffusion_model: an instance of a landlab diffusion component.
            default: None
        erosion_model: an instance of the landlab FastscapeEroder component.
            default: None
        regolith_model: an instance of a DupuitLEM regolith model.
            default: None
        morphologic_scaling_factor: float. Multiplying factor on the hydrological
            timestep to calculate the morphologic timestep.
            default: 500
        total_morphological_time: float. Total model duration.
            default: 1e6*365*24*3600 (1Myr)
        output_dict: dict containing fields to specify output behavior.
            default: None
            dict contains the following fields:
                output_interval: int. The number of model iterations between saving output
                output_fields: list of string(s). Fields at node that will be saved
                    to netcdf file. Field should also include the specifier for
                    the grid element location of the specified field. For example,
                    "at_node:topographic__elevation".
                base_path: string. The path and folder base name where the output will
                    be saved.
                id: int. The identifying number of the particular run. Output files
                    are saved to (base_path)-(id)/grid_(id).nc
        steady_state_condition: dict containting information for stopping
            model if a condition on elevation change is met. Stopping conditions
            are currently implemented only for the rate of change between the
            present elevation state, and the state recorded in the most recently
            saved output. So, using steady_state_condition is conditional on
            saving output with output_dict.

            default: None
            dict contains the following fields:
                stop_at_rate: float. the critical rate of elevation change [m/yr]
                how: string. Either 'mean' (find the mean elevation change rate)
                    or 'percentile' (find the corresponding percentile of change)
                value: float. if 'how' is 'percentile', this is the chosen percentile.

        """

        self.verboseprint = print if verbose else lambda *a, **k: None

        self._grid = grid
        self._cores = self._grid.core_nodes
        self._elev = self._grid.at_node["topographic__elevation"]
        self._base = self._grid.at_node["aquifer_base__elevation"]
        self._wt = self._grid.at_node["water_table__elevation"]
        self._gw_flux = self._grid.add_zeros('node', 'groundwater__specific_discharge_node')

        self.hm = hydrology_model
        self.dm = diffusion_model
        self.rm = regolith_model
        self.sp = erosion_model

        self.MSF = morphologic_scaling_factor # morphologic scaling factor [-]
        self.T_m = total_morphological_time #total model time [s]
        if self.T_m and self.MSF:
            self.dt_m = self.hm.T_h*self.MSF
            self.N = int(self.T_m//self.dt_m)

        if output_dict:
            self.save_output = True
            self.output_interval = output_dict["output_interval"]
            self.output_fields = output_dict["output_fields"]
            self.base_path = output_dict["base_output_path"]
            self.id =  output_dict["run_id"]
        else:
            self.save_output = False

        if steady_state_condition:
            self.stop_cond = True
            self.stop_rate = steady_state_condition['stop_at_rate']

            if steady_state_condition['how'] == 'mean':
                self.calc_rate_of_change = lambda elev, elev0, dtm, N: np.mean(abs(elev-elev0))/(N*dtm)

            elif steady_state_condition['how'] == 'percentile':
                c = steady_state_condition['value']
                self.calc_rate_of_change = lambda elev, elev0, dtm, N: np.percentile(abs(elev-elev0),c)/(N*dtm)

            else:
                raise ValueError('stopping condition method %s is not supported'%stopping_cond['how'])
        else:
            self.stop_cond = False

        self.verboseprint('Model initialized')

    def run_step(self, dt_m):
        """
        Run geomorphic step:
        - update discharge field based on stochastic precipitation
        - diffusion and erosion based on discharge field
        - uplift and regolith change
        - check for boundary issues
        """

        #run gw model, calculate discharge fields
        self.hm.run_step()

        #run linear diffusion, erosion
        self.dm.run_one_step(dt_m)
        self.sp.run_one_step(dt_m)

        #uplift and regolith production
        self.rm.run_step(dt_m)

        #check for places where erosion below baselevel occurs, or water table falls below base or above elev
        if (self._elev<self._base).any(): self.verboseprint('Eroded to bedrock')
        self._base[self._elev<self._base] = self._elev[self._elev<self._base] - np.finfo(float).eps

        if (self._elev<0.0).any(): self.verboseprint('Eroded below baselevel')
        # self._elev[self._elev<0.0] = 0.0

        if (self._wt<self._base).any(): self.verboseprint('Water table below base')
        self._wt[self._wt<self._base] = self._base[self._wt<self._base] + np.finfo(float).eps
        self._grid.at_node['aquifer__thickness'][self._cores] = (self._wt - self._base)[self._cores]

        if (self._wt>self._elev).any(): self.verboseprint('Water table above surface')
        self._wt[self._wt>self._elev] = self._elev[self._wt>self._elev]
        self._grid.at_node['aquifer__thickness'][self._cores] = (self._wt - self._base)[self._cores]

    def run_model(self):

        N = self.N
        z_change = np.zeros((N, 5))

        # Run model forward
        for i in range(N):
            elev0 = self._elev.copy()
            self.run_step(self.dt_m)
            self.verboseprint('Completed model loop %d' % i)

            elev_diff = abs(self._elev-elev0)
            z_change[i,0] = np.max(elev_diff)
            z_change[i,1] = np.percentile(elev_diff,90)
            z_change[i,2] = np.percentile(elev_diff,50)
            z_change[i,3] = np.mean(elev_diff)
            z_change[i,4] = np.mean(self._elev-elev0)

            if self.save_output:

                if i % self.output_interval == 0 or i==max(range(N)):

                    # save the specified grid fields
                    self._gw_flux[:] = self.hm._gdp.calc_gw_flux_at_node()
                    filename = self.base_path + str(self.id) + '_grid_' + str(i) + '.nc'
                    to_netcdf(self._grid, filename, include=self.output_fields, format="NETCDF4")

                    # save elevation change information
                    df_ouput = pd.DataFrame(data=z_change, columns=['max_abs','90_abs', '50_abs', 'mean_abs', 'mean'])
                    filename = self.base_path + str(self.id) + '_elev_change.csv'
                    df_ouput.to_csv(filename, index=False, float_format='%.3e')

                    # check stopping condition
                    if self.stop_cond and i > 0:

                        filename0 = self.base_path + str(self.id) + '_grid_' + str(i-self.output_interval) + '.nc'
                        grid0 = from_netcdf(filename0)
                        elev0 = grid0.at_node['topographic__elevation']
                        dzdt = self.calc_rate_of_change(self._elev, elev0, self.dt_m, self.output_interval)

                        if dzdt < self.stop_rate:
                            verboseprint('Stopping rate condition met, dzdt = %.4e'%dzdt)
                            break
