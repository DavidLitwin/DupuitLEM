"""
Author: David Litwin

26 May 2020
"""
import numpy as np
import pandas as pd

from landlab.io.netcdf import to_netcdf, from_netcdf


class StreamPowerModel:
    """
    Landscape evolution model using the GroundwaterDupuitPercolator in which flow
    that results from overland flow is tracked and averaged to update topography
    using a streampower formulation and FastscapeEroder.
    """

    def __init__(
        self,
        grid,
        hydrology_model=None,
        diffusion_model=None,
        erosion_model=None,
        regolith_model=None,
        morphologic_scaling_factor=500,
        total_morphological_time=1e6 * 365 * 24 * 3600,
        maximum_morphological_dt=None,
        output_dict=None,
        steady_state_condition=None,
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
        maximum_morphological_dt: float. The maximum allowable morphologic timestep.
            default: None
        output_dict: dict containing fields to specify output behavior.
            default: None
            dict contains the following fields:
                output_interval: int. The number of model iterations between saving output
                output_fields: list of string(s). Fields at node that will be saved
                    to netcdf file.
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
                percentile_value: float. if 'how' is 'percentile', this is the chosen percentile.

        """

        self.verboseprint = print if verbose else lambda *a, **k: None

        self._grid = grid
        self._cores = self._grid.core_nodes
        self._elev = self._grid.at_node["topographic__elevation"]
        self._base = self._grid.at_node["aquifer_base__elevation"]
        self._wt = self._grid.at_node["water_table__elevation"]
        self._gw_flux = self._grid.add_zeros(
            "node", "groundwater__specific_discharge_node"
        )

        self.hm = hydrology_model
        self.dm = diffusion_model
        self.rm = regolith_model
        self.sp = erosion_model

        self.MSF = morphologic_scaling_factor  # morphologic scaling factor [-]
        self.T_m = total_morphological_time  # total model time [s]
        self.dt_m_max = maximum_morphological_dt  # max morphologic timestep [s]
        if self.T_m and self.MSF:
            self.dt_m = self.hm.T_h * self.MSF
            self.N = int(self.T_m // self.dt_m)

        if output_dict:
            self.save_output = True
            self.output_interval = output_dict["output_interval"]
            self.output_fields = output_dict["output_fields"]
            self.base_path = output_dict["base_output_path"]
            self.id = output_dict["run_id"]
        else:
            self.save_output = False

        if steady_state_condition:
            if not output_dict:
                raise ValueError(
                    "output_dict must be provided to run with steady_state_condition."
                )

            self.stop_cond = True
            self.stop_rate = steady_state_condition["stop_at_rate"]

            if steady_state_condition["how"] == "mean":
                self.calc_rate_of_change = lambda elev, elev0, dtm, N: np.mean(
                    abs(elev - elev0)
                ) / (N * dtm)

            elif steady_state_condition["how"] == "percentile":
                c = steady_state_condition["percentile_value"]
                self.calc_rate_of_change = lambda elev, elev0, dtm, N: np.percentile(
                    abs(elev - elev0), c
                ) / (N * dtm)

            else:
                raise ValueError(
                    "stopping condition method %s is not supported"
                    % steady_state_condition["how"]
                )
        else:
            self.stop_cond = False

        self.verboseprint("Model initialized")

    def run_step(self, dt_m, dt_m_max=None):
        """
        Run geomorphic step:
        - update discharge field based on stochastic precipitation
        - diffusion and erosion based on discharge field
        - uplift and regolith change
        - check for boundary issues

        Parameters:
        --------
        dt_m: morphoplogic timestep. float.
        dt_m_max: maximum morphoplogic timestep. float. If provided,
            geomorphic timestep dt_m is subdivided so that
            no substep exceeds dt_m_max.
            default: None
        """

        # run gw model, calculate discharge fields
        self.hm.run_step()

        if dt_m_max is not None:

            remaining_time = dt_m
            self.num_substeps = 0

            while remaining_time > 0.0:
                substep_dt = min(remaining_time, dt_m_max)
                # run linear diffusion, erosion
                self.dm.run_one_step(substep_dt)
                self.sp.run_one_step(substep_dt)

                # uplift and regolith production
                self.rm.run_step(substep_dt)

                remaining_time -= substep_dt
                self.num_substeps += 1

        else:
            # run linear diffusion, erosion
            self.dm.run_one_step(dt_m)
            self.sp.run_one_step(dt_m)

            # uplift and regolith production
            self.rm.run_step(dt_m)

        # check for places where erosion below baselevel occurs, or water table falls below base or above elev
        if (self._elev < self._base).any():
            self.verboseprint("Eroded to bedrock")
        self._base[self._elev < self._base] = (
            self._elev[self._elev < self._base] - np.finfo(float).eps
        )

        if (self._elev < 0.0).any():
            self.verboseprint("Eroded below baselevel")
        # self._elev[self._elev<0.0] = 0.0

        if (self._wt < self._base).any():
            self.verboseprint("Water table below base")
        self._wt[self._wt < self._base] = (
            self._base[self._wt < self._base] + np.finfo(float).eps
        )
        self._grid.at_node["aquifer__thickness"][self._cores] = (self._wt - self._base)[
            self._cores
        ]

        if (self._wt > self._elev).any():
            self.verboseprint("Water table above surface")
        self._wt[self._wt > self._elev] = self._elev[self._wt > self._elev]
        self._grid.at_node["aquifer__thickness"][self._cores] = (self._wt - self._base)[
            self._cores
        ]

    def run_model(self):
        """
        Run StreamPowerModel for full duration specified by total_morphological_time.
        Record elevation change quantiles [0, 10, 50, 90, 100] and mean.
        If output dictionary was provided, record output.
        If output dictionary and steady state condition were provided, record output and stop when steady state condition is met.
        """

        N = self.N
        self.z_change = np.zeros((N, 5))

        # Run model forward
        for i in range(N):

            self.run_step(self.dt_m, dt_m_max=self.dt_m_max)
            self.verboseprint("Completed model loop %d" % i)

            if self.save_output:

                if i % self.output_interval == 0 or i == max(range(N)):

                    # save the specified grid fields
                    filename = self.base_path + "%d_grid_%d.nc" % (self.id, i)
                    to_netcdf(
                        self._grid,
                        filename,
                        include=self.output_fields,
                        format="NETCDF4",
                    )

                if self.stop_cond and i % self.output_interval == 0 and i > 0:

                    # check stopping condition
                    filename0 = self.base_path + "%d_grid_%d.nc" % (
                        self.id,
                        i - self.output_interval,
                    )
                    grid0 = from_netcdf(filename0)
                    elev0 = grid0.at_node["topographic__elevation"]
                    dzdt = self.calc_rate_of_change(
                        self._elev, elev0, self.dt_m, self.output_interval
                    )

                    if dzdt < self.stop_rate:
                        self.verboseprint(
                            "Stopping rate condition met, dzdt = %.4e" % dzdt
                        )
                        break
