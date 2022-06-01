# -*- coding: utf-8 -*-
"""
Created on Wed May 12 08:37:58 2021

An implementation of the SWIEM vadose zone model described by Schenk (2008)
"The Shallowest Possible Water Extracton Profile: A Null Model for Global Root
Distributons". The model tracks discrete soil layers, and is "lazy" in the
sense that ET always removes water from the highest soil layers where water
is present. The saturation state in each layer is binary: either it is
saturated to some "field capacity" or has drained to some "wilting point".

Added to this model, we present a solution for determining the recharge to
the water table at depths. We calculate the water that fills soil layers
below each soil layer depth and call this the recharge to the water table.

@author: dgbli
"""

import numpy as np
from .schenk_analytical_solutions import extraction_cdf, saturation_state

class SchenkVadoseModel:
    def __init__(
        self,
        potential_evapotranspiration_rate=2e-7,
        available_water_content=0.15,
        profile_depth=5,
        num_bins=500,
    ):
        """
        Initialize SchenkVadoseModel.

        Parameters
        ----------
        potential_evapotranspiration_rate: float (L/T).
            Default=2x10^-7 m/s.
        available_water_content: float (-).
            Plant available water content, as a proportion of total volume.
            Range 0 to 1. Default=0.15.
        profile_depth: float (L).
            Depth of vadose zone to be considered. Defaul=5 m.
        num_bins: int (-).
            Number of saturation bins in which to divide profile. Default=500.
        """
        self.pet = potential_evapotranspiration_rate
        self.Sawc = available_water_content
        self.b = profile_depth
        self.Nz = int(num_bins)

        # bottom elev of each bin
        self.depths = np.linspace(self.b / self.Nz, self.b, self.Nz)
        # saturation state (binary) in each bin
        self.sat_profile = np.zeros_like(self.depths)
        # saturation state difference (binary) changed each timestep
        self.sat_diff = np.zeros_like(self.depths)
        # recharge depth when water table is at each profile depth
        self.recharge_at_depth = np.zeros_like(self.depths)
        # extraction depth when water table is at each profile depth
        self.extraction_at_depth = np.zeros_like(self.depths)
        # water depth capacity in each bin
        self.bin_capacity = (self.b / self.Nz) * self.Sawc
        # mask to set extraction below a certain depth to zero
        self.extraction_depth_mask = np.zeros_like(self.depths, dtype=bool)

    def generate_state_from_analytical(
        self, mean_storm_depth, mean_interstorm_duration, random_seed=None,
    ):

        """
        Set the saturation profile by generating random values from
        the analytical solution for saturation state.

        Parameters
        ----------
        mean_storm_depth: float (L). Mean storm depth.
        mean_interstorm_duration: float (T). Mean interstorm duration.
        """
        if random_seed:
            np.random.seed(random_seed)

        self.analytical_sat_prob = saturation_state(self.depths,
                                            mean_interstorm_duration,
                                            mean_storm_depth,
                                            self.pet,
                                            self.Sawc,
                                            )

        r = np.random.rand(len(self.depths))
        self.sat_profile = 1 * (r < self.analytical_sat_prob)

    def set_max_extraction_depth(
        self,
        mean_storm_depth,
        mean_interstorm_duration,
        threshold=0.01,
    ):

        """
        Set the field extraction_depth_mask, which, when used in
        conjunction with the `run_interevent` method, sets the root water uptake
        rate for water tables below a certain depth equal to zero. It does this
        using the analytical cdf of rooting depth to find the location where
        the proportion (1 - threshold) of uptake occurs above this depth. This
        threshold generally only affects root water uptake in arid cases
        (pet*tb > ds), where the equilibrium solution to the model suggests
        some root water uptake occurs at infinite depth.

        Parameters
        ----------
        mean_storm_depth: float (L). Mean storm depth.
        mean_interstorm_duration: float (T). Mean interstorm duration.
        threshold: threshold relative frequency of extraction. Default 0.001.
        """

        cdf = extraction_cdf(
            self.depths,
            mean_storm_depth,
            mean_interstorm_duration,
            self.pet,
            self.Sawc,
        )

        self.extraction_depth_mask = cdf > 1 - threshold
        return cdf

    def generate_storm(
        self,
        mean_storm_depth,
        mean_storm_duration,
        mean_interstorm_duration,
        random_seed=None,
    ):

        """Generate one storm depth, duration, and insterstorm duration from
        exponential distributions.

        Parameters
        ----------
        mean_storm_depth: float (L). Mean storm depth.
        mean_storm_duration: float (T). Mean storm duration.
        mean_interstorm_duration: float (T). Mean interstorm duration.
        """

        if random_seed:
            np.random.seed(random_seed)

        self.Dr = np.random.exponential(mean_storm_depth)
        self.Tr = np.random.exponential(mean_storm_duration)
        self.Tb = np.random.exponential(mean_interstorm_duration)

    def run_event(self, storm_depth):
        """Run storm event, updating saturation profile and recharge at depth.

        Parameters
        ----------
        storm_dt: float. Storm depth.
        """

        # clear sat diff
        self.sat_diff[:] = 0

        # number of bins the storm will fill
        n_to_fill = round(storm_depth / self.bin_capacity)

        # change bin status
        inds_to_fill = np.where(self.sat_profile == 0)[0][0:n_to_fill]
        self.sat_profile[inds_to_fill] = 1

        # calculate recharge
        self.sat_diff[inds_to_fill] = 1
        self.recharge_at_depth[:] = (
            n_to_fill - np.cumsum(self.sat_diff)
        ) * self.bin_capacity

    def calc_recharge_rate(self, wt_from_surface, storm_dt):
        """calculate the recharge rate during storm event given the depth of water
        table from surface. Returns the recharge rate for each water table
        depth provided. If supplied wt depth is greater than profile depth,
        recharge rate is depth at base of the profile.

        Parameters
        ----------
        wt_from_surface: array of floats. Positive values from 0 to
            maximum aquifer depth, which is usually this is also profile depth.
        storm_dt: float. Storm duration.
        """

        wt_from_surface[wt_from_surface > self.b] = self.b
        wt_digitized = np.digitize(wt_from_surface, self.depths, right=True)

        out = self.recharge_at_depth[wt_digitized] / storm_dt

        return out

    def run_interevent(self, interstorm_dt):
        """Run storm interevent, updating saturation profile.

        Parameters
        ----------
        interstorm_dt: float. Duration without precipitation."""

        # clear sat diff
        self.sat_diff[:] = 0

        # number of bins ET will drain
        n_to_drain = round(self.pet * interstorm_dt / self.bin_capacity)

        # change bin status
        inds_to_drain = np.where(self.sat_profile == 1)[0][0:n_to_drain]
        self.sat_profile[inds_to_drain] = 0

        # calculate extraction
        self.sat_diff[inds_to_drain] = -1
        self.extraction_at_depth[:] = (
            -n_to_drain - np.cumsum(self.sat_diff)
        ) * self.bin_capacity

        # set extraction to zero where masked
        self.extraction_at_depth[self.extraction_depth_mask] = 0.0

    def calc_extraction_rate(self, wt_from_surface, interstorm_dt):
        """calculate the extraction rate given the depth of water table from
        surface. Returns the (negative) extraction rate for each water
        table depth provided. If supplied depth is at or greater than profile depth,
        extraction rate is zero.

        Parameters
        ----------
        wt_from_surface: array of floats. Positive values from 0 to
            maximum aquifer depth, which is usually this is also profile depth.
        interstorm_dt: float. Interstorm duration.
        """

        wt_from_surface[wt_from_surface > self.b] = self.b
        wt_digitized = np.digitize(wt_from_surface, self.depths, right=True)

        out = self.extraction_at_depth[wt_digitized] / interstorm_dt
        out[wt_from_surface == self.b] = 0.0

        return out

    def run_one_step(
        self,
        mean_storm_depth,
        mean_storm_duration,
        mean_interstorm_duration,
        random_seed=None,
    ):

        """Run step: generate exponential storm depth duration, interstorm
        duration, run event, and run interevent.

        Parameters
        ----------
        mean_storm_depth: float (L). Mean storm depth.
        mean_storm_duration: float (T). Mean storm duration.
        mean_interstorm_duration: float (T). Mean interstorm duration.
        random_seed: int (-) numpy random seed for reproduceability.
        """

        if random_seed:
            np.random.seed(random_seed)

        self.generate_storm(
            mean_storm_depth, mean_storm_duration, mean_interstorm_duration,
        )
        self.run_event(self.Dr)
        self.run_interevent(self.Tb)

    def run_model(
        self,
        num_timesteps=100,
        mean_storm_depth=0.02,
        mean_storm_duration=1e3,
        mean_interstorm_duration=1e5,
        random_seed=None,
    ):

        """Run model: run step Nt times, calculate average recharge depth and
        recharge frequency at each depth in the profile.

        Parameters
        ----------
        num_timesteps: int (-) number of storm interstorm pairs to run.
        mean_storm_depth: float (L). Mean storm depth.
        mean_storm_duration: float (T). Mean storm duration.
        mean_interstorm_duration: float (T). Mean interstorm duration.
        random_seed: int (-) numpy random seed for reproduceability.

        """

        self.d = mean_storm_depth
        self.tr = mean_storm_duration
        self.tb = mean_interstorm_duration
        self.Nt = num_timesteps
        if random_seed:
            np.random.seed(random_seed)

        self.bool_extraction_at_depth = np.zeros_like(self.depths)
        self.cum_recharge = np.zeros_like(self.depths)
        self.cum_extraction = np.zeros_like(self.depths)
        self.bool_recharge = np.zeros_like(self.depths)
        self.bool_extraction = np.zeros_like(self.depths)
        self.cum_storm_dt = 0
        self.cum_interstorm_dt = 0
        self.cum_precip = 0

        for i in range(self.Nt):

            self.run_one_step(self.d, self.tr, self.tb)

            self.bool_extraction_at_depth += -self.sat_diff
            self.cum_recharge += self.recharge_at_depth
            self.cum_extraction += self.extraction_at_depth
            self.bool_recharge += self.recharge_at_depth > 0.0
            self.bool_extraction += -self.extraction_at_depth > 0.0
            self.cum_storm_dt += self.Tr
            self.cum_interstorm_dt += self.Tb
            self.cum_precip += self.d

        # define mean recharge depth as depth of events > 0 at that profile depth
        self.mean_recharge_depth = self.cum_recharge / self.bool_recharge
        # define frequency as number of times recharge occurs over total time
        self.recharge_frequency = self.bool_recharge / (
            self.cum_storm_dt + self.cum_interstorm_dt
        )
        # plant rooting depth pdf
        self.plant_rooting_pdf = self.bool_extraction_at_depth / (np.sum(self.bool_extraction_at_depth) * np.diff(self.depths)[0])
