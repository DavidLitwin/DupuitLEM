"""
Landscape evolution model using the GroundwaterDupuitPercolator in which
recharge occurs at a steady rate through the model duration. Here fluvial erosion
is handled by the FastscapeEroder.

Author: David Litwin

"""
import time
import numpy as np

from landlab import RasterModelGrid
from landlab.components import (
    GroundwaterDupuitPercolator,
    FlowAccumulator,
    FastscapeEroder,
    LinearDiffuser,
    LakeMapperBarnes,
    DepressionFinderAndRouter,
    )
from DupuitLEM.grid_functions.grid_funcs import calc_avg_hydraulic_conductivity


class SimpleSteadyRecharge:

    """
    Simple groundwater landscape evolution model with constant uplift/baselevel
    fall, linear hillslope diffusive transport, and detachment limited erosion
    generated by accumulated groundwater return flow and saturation excess overland flow
    from steady recharge to a shallow aquifer.

    """

    def __init__(self,params,save_output=True):

        self._grid = params.pop("grid"))
        self._cores = self._grid.core_nodes

        self.R = params.pop("recharge_rate")) #[m/s]
        self.Ks = params.pop("hydraulic_conductivity") #[m/s]
        self.K0 = params.pop("min_hydraulic_conductivity")  #[m/s]
        self.d_k = params.pop("characteristic_k_depth")
        self.n = params.pop("porosity")
        self.r = params.pop("regularization_factor")
        self.c = params.pop("courant_coefficient")
        self.vn = params.pop("vn_coefficient")

        self.w0 = params.pop("permeability_production_rate") #[m/s]
        self.d_s = params.pop("characteristic_w_depth")
        self.U = params.pop("uplift_rate") # uniform uplift [m/s]
        self.m = params.pop("m_sp") #Exponent on Q []
        self.n = params.pop("n_sp") #Exponent on S []
        self.K = params.pop("k_sp") #erosivity coefficient [m-1/2 s−1/2]
        self.D = params.pop("hillslope_diffusivity") # hillslope diffusivity [m2/s]

        self.dt_h = params.pop("hydrological_timestep") # hydrological timestep [s]
        self.T = params.pop("total_time") # total simulation time [s]
        self.MSF = params.pop("morphologic_scaling_factor") # morphologic scaling factor [-]
        self.dt_m = self.MSF*self.dt_h
        self.N = int(self.T//self.dt_m)

        self._elev = self._grid.at_node("topographic__elevation")
        self._base = self._grid.at_node("aquifer_base__elevation")
        self._wt = self._grid.at_node("water_table__elevation")
        self._gw_flux = self._grid.add_zeros('node', 'groundwater__specific_discharge_node')

        if save_output:
            self._output_interval = params.pop("output_interval")
            self._output_fields = params.pop("output_fields")
            self._track_times = params.pop("track_times")


        # initialize model components
        self.gdp = GroundwaterDupuitPercolator(self._grid, porosity=self.n, \
                                          recharge_rate=self.R, regularization_f=self.r, \
                                          courant_coefficient=self.c, vn_coefficient = self.vn)
        self.fa = FlowAccumulator(self._grid, surface='topographic__elevation', flow_director='D8',  \
                              runoff_rate='average_surface_water__specific_discharge')
        self.lmb = LakeMapperBarnes(self._grid, method='D8', fill_flat=False,
                                      surface='topographic__elevation',
                                      fill_surface='topographic__elevation',
                                      redirect_flow_steepest_descent=False,
                                      reaccumulate_flow=False,
                                      track_lakes=False,
                                      ignore_overfill=True)
        self.sp = FastscapeEroder(self._grid, K_sp = self.K, m_sp = self.m, n_sp=self.n, discharge_field='surface_water__discharge')
        self.ld = LinearDiffuser(self._grid, linear_diffusivity = self.D)
        self.dfr = DepressionFinderAndRouter(self._grid)


    def run_model(self):
        """ run the model for the full duration"""

        N = self.N
        num_substeps = np.zeros(N)
        max_rel_change = np.zeros(N)
        perc90_rel_change = np.zeros(N)
        times = np.zeros((N,5))
        num_pits = np.zeros(N)

        t0 = time.time()
        # Run model forward
        for i in range(N):
            elev0 = self._elev.copy()

            t1 = time.time()
            #set hydraulic conductivity based on depth
            self.gdp.K = calc_avg_hydraulic_conductivity(self._grid,grid.at_node['aquifer__thickness'],
                                             self._elev-self._base,
                                             self.K0,self.Ks,self.d_k,
                                             )
            #run gw model
            self.gdp.run_with_adaptive_time_step_solver(self.dt_h)
            num_substeps[i] = self.gdp.number_of_substeps

            t2 = time.time()
            #uplift and regolith production
            self._elev[self._cores] += self.U*self.dt_m
            self._base[self._cores] += self.U*self.dt_m - self.w0*np.exp(-(self._elev[self._cores]-self._base[self._cores])/self.d_s)*self.dt_m

            t3 = time.time()
            self.dfr._find_pits()
            if self.dfr._number_of_pits > 0:
                self.lmb.run_one_step()

            t4 = time.time()
            self.fa.run_one_step()

            t5 = time.time()
            self.ld.run_one_step(dt_m)
            self.sp.run_one_step(dt_m)
            self._elev[self._elev<self._base] = self._base[self._elev<self._base]

            t6 = time.time()
            times[i:] = [t2-t1, t3-t2, t4-t3, t5-t4, t6-t5]
            num_pits[i] = self.dfr._number_of_pits
            elev_diff = abs(self._elev-elev0)/elev0
            max_rel_change[i] = np.max(elev_diff)
            perc90_rel_change[i] = np.percentile(elev_diff,90)
