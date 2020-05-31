"""
Combined hydrological and fluvial erosion models for DupuitLEM. Take instantiated
landlab components and functions indicating how the shear stress and erosion
should be calculated from overland flow, and return an effective erosion rate
over the hydrological model run time.

19 May 2020
"""

import numpy as np

from landlab.components import (
    FlowDirectorD8,
    FlowAccumulator,
    LakeMapperBarnes,
    DepressionFinderAndRouter,
    )

class HydrologicalModel:
    """
    Base class for hydrological model.

    Parameters
    -----
    grid: a landlab grid with GroundwaterDupuitPercolator already instantiated

    """
    def __init__(self, grid):

        self._grid = grid

        self.fd = FlowDirectorD8(self._grid)
        self.fa = FlowAccumulator(self._grid, surface='topographic__elevation', flow_director=self.fd,  \
                              runoff_rate='average_surface_water__specific_discharge')
        self.lmb = LakeMapperBarnes(self._grid, method='D8', fill_flat=False,
                                      surface='topographic__elevation',
                                      fill_surface='topographic__elevation',
                                      redirect_flow_steepest_descent=False,
                                      reaccumulate_flow=False,
                                      track_lakes=False,
                                      ignore_overfill=True)
        self.dfr = DepressionFinderAndRouter(self._grid)

    def run_step(self):
        raise NotImplementedError


class HydrologyIntegrateShearStress(HydrologicalModel):

    """"
    Stochastic hydrological model running pairs of events and interevents over
    total time specifiied by instantiated PrecipitationDistribution.
    A trapezoidal integration method is used to find effective
    shear stress and erosion rate.

    Note: This method may overestimate shear stress and erosion rate during the recession period.

    Parameters
    -----
    grid: landlab grid
    precip_generator: instantiated PrecipitationDistribution
    groundwater_model: instantiated GroundwaterDupuitPercolator
    shear_stress_function: function that takes a grid with topography and discharge and returns shear stress at node
    erosion_rate_function: function that takes grid with topography and shear stress and returns erosion rate
    tauc: shear stress threshold for erosion

    """

    def __init__(
        self,
        grid,
        precip_generator=None,
        groundwater_model=None,
        shear_stress_function=None,
        erosion_rate_function=None,
        tauc = 0,
        ):
        super().__init__(grid)

        self._tau = self._grid.add_zeros("node","surface_water__shear_stress")
        self.pd = precip_generator
        self.gdp = groundwater_model
        self.calc_shear_stress = shear_stress_function
        self.calc_erosion_from_shear_stress = erosion_rate_function
        self._tauc = tauc
        self.T_h = self.pd._run_time

    @staticmethod
    def calc_storm_eff_shear_stress(tau0,tau1,tau2,tauc,tr,tb):
        """
        Calculate effective shear stress over the course of an event-interevent
        period using a trapezoidal approximation, which accounts for the linear
        interpolation of when the threshold shear stress is exceeded. Note that this
        method may overestimate shear stress during interevent if it quickly drops
        below the threshold value.
        """

        tauint1 = np.zeros_like(tau0)
        tauint2 = np.zeros_like(tau0)

        c1 = np.logical_and(tau0>tauc,tau1>tauc)
        c2 = np.logical_and(tau0<tauc,tau1>tauc)
        c3 = np.logical_and(tau0>tauc,tau1<tauc)
        #c4 = np.logical_and(tau0<tauc,tau1<tauc) #implied
        tauint1[c1] = 0.5*tr*(tau0[c1]+tau1[c1]-2*tauc)
        tauint1[c2] = 0.5*tr*(tau1[c2]-tauc)*((tau1[c2]-tauc)/(tau1[c2]-tau0[c2]))
        tauint1[c3] = 0.5*tr*(tau0[c3]-tauc)*((tau0[c3]-tauc)/(tau0[c3]-tau1[c3]))
        #tauint1[c4] = 0.0 #implied

        c1 = np.logical_and(tau1>tauc,tau2>tauc)
        c2 = np.logical_and(tau1<tauc,tau2>tauc)
        c3 = np.logical_and(tau1>tauc,tau2<tauc)
        #c4 = np.logical_and(tau1<tauc,tau2<tauc) #implied
        tauint2[c1] = 0.5*tb*(tau1[c1]+tau2[c1]-2*tauc)
        tauint2[c2] = 0.5*tb*(tau2[c2]-tauc)*((tau2[c2]-tauc)/(tau2[c2]-tau1[c2]))
        tauint2[c3] = 0.5*tb*(tau1[c3]-tauc)*((tau1[c3]-tauc)/(tau1[c3]-tau2[c3]))
        #tauint2[c4] = 0.0 #implied

        taueff = (tauint1+tauint2)/(tr+tb) + tauc
        return taueff

    def generate_exp_precip(self):
        """
        Generate series of storm_dts, interstorm_dts, and intensities from
        PrecipitationDistribution.
        """
        storm_dts = []
        interstorm_dts = []
        intensities = []

        for (storm_dt, interstorm_dt) in self.pd.yield_storms():
            storm_dts.append(storm_dt)
            interstorm_dts.append(interstorm_dt)
            intensities.append(float(self._grid.at_grid['rainfall__flux']))

        self.storm_dts = storm_dts
        self.interstorm_dts = interstorm_dts
        self.intensities = intensities

    def run_step(self):

        """
        Hydrological model for series of event-interevent pairs, calculate shear stresses
        at end of event and interevent, calculate erosion rate.
        """

        #generate new precip time series
        self.generate_exp_precip()

        #find and route flow if there are pits
        self.dfr._find_pits()
        if self.dfr._number_of_pits > 0:
            self.lmb.run_one_step()

        #update flow directions
        self.fd.run_one_step()

        self.dzdt_eff = np.zeros_like(self._tau)
        tau2 = self._tau.copy()
        for i in range(len(self.storm_dts)):
            tau0 = tau2.copy() #save prev end of interstorm shear stress

            #run event, accumulate flow, and calculate resulting shear stress
            self.gdp.recharge = self.intensities[i]
            self.gdp.run_with_adaptive_time_step_solver(self.storm_dts[i])
            _,_ = self.fa.accumulate_flow(update_flow_director=False)
            tau1 = self.calc_shear_stress(self._grid)

            #run interevent, accumulate flow, and calculate resulting shear stress
            self.gdp.recharge = 0.0
            self.gdp.run_with_adaptive_time_step_solver(self.interstorm_dts[i])
            _,_ = self.fa.accumulate_flow(update_flow_director=False)
            tau2 = self.calc_shear_stress(self._grid)

            #calculate effective shear stress across event-interevent pair
            self._tau[:] = calc_storm_eff_shear_stress(tau0,tau1,tau2,self._tauc,self.storm_dts[i],self.interstorm_dts[i])

            #calculate erosion rate, and then add time-weighted erosion rate to get effective erosion rate at the end of for loop
            dzdt = calc_erosion_from_shear_stress(self._grid)
            self.dzdt_eff += (self.storm_dts[i]+self.interstorm_dts[i])/self.T_h * dzdt

class HydrologyEventShearStress(HydrologicalModel):

    """"
    Run hydrological model for series of event-interevent pairs, calculate
    instantaneous shear stress and erosion rate at beginning and end of event.
    Calculate average erosion rate *for event only* and average this over the
    whole duration. This method assumes erosion is negligible during the
    interevent periods.

    Parameters
    -----
    grid: landlab grid
    precip_generator: instantiated PrecipitationDistribution
    groundwater_model: instantiated GroundwaterDupuitPercolator
    shear_stress_function: function that takes a grid with topography and discharge and returns shear stress at node
    erosion_rate_function: function that takes grid with topography and shear stress and returns erosion rate

    """

    def __init__(
        self,
        grid,
        precip_generator=None,
        groundwater_model=None,
        shear_stress_function=None,
        erosion_rate_function=None,
    ):

        super().__init__(grid)

        self._tau = self._grid.add_zeros("node","surface_water__shear_stress")
        self.pd = precip_generator
        self.gdp = groundwater_model
        self.calc_shear_stress = shear_stress_function
        self.calc_erosion_from_shear_stress = erosion_rate_function
        self.T_h = self.pd._run_time

    def generate_exp_precip(self):

        storm_dts = []
        interstorm_dts = []
        intensities = []

        for (storm_dt, interstorm_dt) in self.pd.yield_storms():
            storm_dts.append(storm_dt)
            interstorm_dts.append(interstorm_dt)
            intensities.append(float(self._grid.at_grid['rainfall__flux']))

        self.storm_dts = storm_dts
        self.interstorm_dts = interstorm_dts
        self.intensities = intensities

    def run_step(self):
        """"
        Run hydrological model for series of event-interevent pairs, calculate shear stresses
        and calculate effective erosion rate over the total_hydrological_time. Erosion rate
        is from event period only.
        """

        #generate new precip time series
        self.generate_exp_precip()

        #find and route flow if there are pits
        self.dfr._find_pits()
        if self.dfr._number_of_pits > 0:
            self.lmb.run_one_step()

        #update flow directions
        self.fd.run_one_step()

        self.max_substeps_storm = 0
        self.max_substeps_interstorm = 0
        self.dzdt_eff = np.zeros_like(self._tau)
        dzdt2 = np.zeros_like(self._tau)
        for i in range(len(self.storm_dts)):
            dzdt0 = dzdt2.copy() #save prev end of interstorm erosion rate

            #run event, accumulate flow, and calculate resulting shear stress
            self.gdp.recharge = self.intensities[i]
            self.gdp.run_with_adaptive_time_step_solver(self.storm_dts[i])
            _,_ = self.fa.accumulate_flow(update_flow_director=False)
            self._tau[:] = self.calc_shear_stress(self._grid)
            dzdt1 = self.calc_erosion_from_shear_stress(self._grid)
            self.max_substeps_storm = max(self.max_substeps_storm,self.gdp.number_of_substeps)

            #run interevent, accumulate flow, and calculate resulting shear stress
            self.gdp.recharge = 0.0
            self.gdp.run_with_adaptive_time_step_solver(self.interstorm_dts[i])
            _,_ = self.fa.accumulate_flow(update_flow_director=False)
            self._tau[:] = self.calc_shear_stress(self._grid)
            dzdt2 = self.calc_erosion_from_shear_stress(self._grid)
            self.max_substeps_interstorm = max(self.max_substeps_interstorm,self.gdp.number_of_substeps)

            #calculate erosion, and then add time-weighted erosion rate to get effective erosion rate at the end of for loop
            #note that this only accounts for erosion during the storm period
            deltaz = 0.5*(dzdt0+dzdt1)*self.storm_dts[i]
            self.dzdt_eff += deltaz / self.T_h


    def run_step_record_state(self):
        """"
        Run hydrological model for series of event-interevent pairs, calculate shear stresses
        and calculate effective erosion rate over the total_hydrological_time

        track the state of the model:
            time, intensity, tau_all (shear stress), Q_all (discharge),
            wtrel_all (relative water table position),
            qs_all (surface water specific discharge)

        """

        #fields to record:
        self.time = np.zeros(2*len(self.storm_dts)+1)
        self.intensity = np.zeros(2*len(self.storm_dts)+1)
        self.tau_all = np.zeros((2*len(self.storm_dts)+1,len(self._tau))) #all shear stress
        self.Q_all = np.zeros((2*len(self.storm_dts)+1,len(self._tau))) #all discharge
        self.wtrel_all = np.zeros((2*len(self.storm_dts)+1,len(self._tau))) #all relative water table elevation
        self.qs_all = np.zeros((2*len(self.storm_dts)+1,len(self._tau))) #all surface water specific discharge

        self.max_substeps_storm = 0
        self.max_substeps_interstorm = 0

        #find and route flow if there are pits
        self.dfr._find_pits()
        if self.dfr._number_of_pits > 0:
            self.lmb.run_one_step()

        #update flow directions
        self.fd.run_one_step()

        self.max_substeps_storm = 0
        self.max_substeps_interstorm = 0
        self.dzdt_eff = np.zeros_like(self._tau)
        dzdt2 = np.zeros_like(self._tau)
        for i in range(len(self.storm_dts)):
            dzdt0 = dzdt2.copy() #save prev end of interstorm erosion rate

            #run event, accumulate flow, and calculate resulting shear stress
            self.gdp.recharge = self.intensities[i]
            self.gdp.run_with_adaptive_time_step_solver(self.storm_dts[i])
            _,_ = self.fa.accumulate_flow(update_flow_director=False)
            self._tau[:] = self.calc_shear_stress(self._grid)
            dzdt1 = self.calc_erosion_from_shear_stress(self._grid)
            self.max_substeps_storm = max(self.max_substeps_storm,self.gdp.number_of_substeps)

            #record event
            self.max_substeps_storm = max(self.max_substeps_storm,self.gdp.number_of_substeps)
            self.time[i*2+1] = self.time[i*2]+self.storm_dts[i]
            self.intensity[i*2] = self.intensities[i]
            self.tau_all[i*2+1,:] = self._tau
            self.Q_all[i*2+1,:] = self._grid.at_node['surface_water__discharge']
            self.wtrel_all[i*2+1,:] = (self._wt-self._base)/(self._elev-self._base)
            self.qs_all[i*2+1,:] = self._grid.at_node['surface_water__specific_discharge']

            #run interevent, accumulate flow, and calculate resulting shear stress
            self.gdp.recharge = 0.0
            self.gdp.run_with_adaptive_time_step_solver(self.interstorm_dts[i])
            _,_ = self.fa.accumulate_flow(update_flow_director=False)
            self._tau[:] = self.calc_shear_stress(self._grid)
            dzdt2 = self.calc_erosion_from_shear_stress(self._grid)
            self.max_substeps_interstorm = max(self.max_substeps_interstorm,self.gdp.number_of_substeps)

            #record interevent
            self.max_substeps_interstorm = max(self.max_substeps_interstorm,self.gdp.number_of_substeps)
            self.time[i*2+2] = self.time[i*2+1]+self.interstorm_dts[i]
            self.tau_all[i*2+2,:] = self._tau
            self.Q_all[i*2+2,:] = self._grid.at_node['surface_water__discharge']
            self.wtrel_all[i*2+2,:] = (self._wt-self._base)/(self._elev-self._base)
            self.qs_all[i*2+2,:] = self._grid.at_node['surface_water__specific_discharge']

            #calculate erosion, and then add time-weighted erosion rate to get effective erosion rate at the end of for loop
            #note that this only accounts for erosion during the storm period
            deltaz = 0.5*(dzdt0+dzdt1)*self.storm_dts[i]
            self.dzdt_eff += deltaz / self.T_h

class HydrologySteadyShearStress(HydrologicalModel):
    """"
    Run hydrological model for steady recharge.
    Calculate shear stress and erosion rate at the end of the timestep

    Parameters
    -----
    grid: landlab grid
    groundwater_model: instantiated GroundwaterDupuitPercolator
    shear_stress_function: function that takes a grid with topography and discharge and returns shear stress at node
    erosion_rate_function: function that takes grid with topography and shear stress and returns erosion rate

    """

    def __init__(
        self,
        grid,
        groundwater_model=None,
        shear_stress_function=None,
        erosion_rate_function=None,
        ):
        super().__init__(grid)

        self._tau = self._grid.add_zeros("node","surface_water__shear_stress")
        self.gdp = groundwater_model
        self.calc_shear_stress = shear_stress_function
        self.calc_erosion_from_shear_stress = erosion_rate_function

    def run_step(self,dt_h):
        """
        Run steay model one step. Update groundwater state, route and accumulate flow,
        calculate shear stress and erosion rate.

        Parameters
        -----
        dt_h: float. Groundwater model timestep
        """

        #run gw model
        self.gdp.run_with_adaptive_time_step_solver(dt_h)
        self.number_substeps = self.gdp.number_of_substeps

        #find pits for flow accumulation
        self.dfr._find_pits()
        if self.dfr._number_of_pits > 0:
            self.lmb.run_one_step()

        #run flow accumulation
        self.fa.run_one_step()

        #calc shear stress and erosion
        self._tau[:] = self.calc_shear_stress(self._grid)
        self.dzdt = self.calc_erosion_from_shear_stress(self._grid)



class HydrologyEventStreamPower(HydrologicalModel):

    """"
    Run hydrological model for series of event-interevent pairs, calculate
    instantaneous flow rate at the beginning and end of event. This method
    assumes erosion is negligible during the interevent periods.
    HydrologyEventStreamPower is meant to be passed to
    StochasticRechargeStreamPower, where erosion rate is calcualted.

    Parameters
    -----
    grid: landlab grid
    precip_generator: instantiated PrecipitationDistribution
    groundwater_model: instantiated GroundwaterDupuitPercolator

    """

    def __init__(
        self,
        grid,
        precip_generator=None,
        groundwater_model=None,
    ):

        super().__init__(grid)

        self.q_eff = self._grid.add_zeros("node","surface_water_effective__discharge")
        self.pd = precip_generator
        self.gdp = groundwater_model
        self.T_h = self.pd._run_time

    def generate_exp_precip(self):

        storm_dts = []
        interstorm_dts = []
        intensities = []

        for (storm_dt, interstorm_dt) in self.pd.yield_storms():
            storm_dts.append(storm_dt)
            interstorm_dts.append(interstorm_dt)
            intensities.append(float(self._grid.at_grid['rainfall__flux']))

        self.storm_dts = storm_dts
        self.interstorm_dts = interstorm_dts
        self.intensities = intensities

    def run_step(self):
        """"
        Run hydrological model for series of event-interevent pairs, calculate shear stresses
        and calculate effective erosion rate over the total_hydrological_time. Erosion rate
        is from event period only.
        """

        #generate new precip time series
        self.generate_exp_precip()

        #find and route flow if there are pits
        self.dfr._find_pits()
        if self.dfr._number_of_pits > 0:
            self.lmb.run_one_step()

        #update flow directions
        self.fd.run_one_step()

        self.max_substeps_storm = 0
        self.max_substeps_interstorm = 0
        q_total_vol = np.zeros_like(self.q_eff)
        q2 = np.zeros_like(self.q_eff)
        for i in range(len(self.storm_dts)):
            q0 = q2.copy() #save prev end of interstorm flow rate

            #run event, accumulate flow
            self.gdp.recharge = self.intensities[i]
            self.gdp.run_with_adaptive_time_step_solver(self.storm_dts[i])
            _,q1 = self.fa.accumulate_flow(update_flow_director=False)
            self.max_substeps_storm = max(self.max_substeps_storm,self.gdp.number_of_substeps)

            #run interevent, accumulate flow
            self.gdp.recharge = 0.0
            self.gdp.run_with_adaptive_time_step_solver(self.interstorm_dts[i])
            _,q2 = self.fa.accumulate_flow(update_flow_director=False)
            self.max_substeps_interstorm = max(self.max_substeps_interstorm,self.gdp.number_of_substeps)

            #calculate erosion, and then add time-weighted erosion rate to get effective erosion rate at the end of for loop
            #note that this only accounts for erosion during the storm period
            q_event_vol = 0.5*(q0+q1)*self.storm_dts[i]
            q_total_vol += q_event_vol

        self.q_eff[:] = q_total_vol/self.T_h
