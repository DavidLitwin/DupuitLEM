"""
StreamPowerModel class of DupuitLEM. Recommended not to use this class
due to stability issues.

Author: David Litwin

19 May 2020
"""
import numpy as np

from landlab.io.netcdf import to_netcdf


class ShearStressModel:
    """
    Landscape evolution model using the GroundwaterDupuitPercolator. Here fluvial erosion
    is calculated by calculating excess shear stress, determining erosion rate
    and applying it in an explicit forward Euler scheme.
    """

    def __init__(
        self,
        grid,
        hydrology_model=None,
        diffusion_model=None,
        regolith_model=None,
        morphologic_scaling_factor=None,
        total_morphological_time=None,
        output_dict=None,
        verbose=False,
    ):

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

        self.MSF = morphologic_scaling_factor  # morphologic scaling factor [-]
        self.T_m = total_morphological_time  # total model time [s]
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
        self.verboseprint("Model initialized")

    def run_step(self, dt_m):

        # run gw model, calculate erosion rate
        self.hm.run_step()

        # run linear diffusion, erosion
        self.dm.run_one_step(dt_m)
        self._elev += self.hm.dzdt * dt_m

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
        self._elev[self._elev < 0.0] = 0.0
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

    def run_model(self):
        """ run the model for the full duration"""

        N = self.N
        max_rel_change = np.zeros(N)
        perc90_rel_change = np.zeros(N)

        # Run model forward
        for i in range(N):
            elev0 = self._elev.copy()

            self.run_step(self.dt_m)
            self.verboseprint("Completed model loop %d" % i)

            elev_diff = abs(self._elev - elev0) / elev0
            max_rel_change[i] = np.max(elev_diff)
            perc90_rel_change[i] = np.percentile(elev_diff, 90)

            if self.save_output:

                if i % self.output_interval == 0 or i == max(range(N)):
                    self._gw_flux[:] = self.hm.gdp.calc_gw_flux_at_node()

                    filename = self.base_path + str(self.id) + "_grid_" + str(i) + ".nc"
                    to_netcdf(
                        self._grid,
                        filename,
                        include=self.output_fields,
                        format="NETCDF4",
                    )
                    print("Completed loop %d" % i)

                    filename = (
                        self.base_path + str(self.id) + "_max_rel_change" + ".txt"
                    )
                    np.savetxt(filename, max_rel_change, fmt="%.4e")

                    filename = (
                        self.base_path + str(self.id) + "_90perc_rel_change" + ".txt"
                    )
                    np.savetxt(filename, perc90_rel_change, fmt="%.4e")
