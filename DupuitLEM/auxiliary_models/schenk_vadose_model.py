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


class SchenkVadoseModel:
    """If run independently, then use storm properties. If using with
    HydrologyEventVadoseStreamPower, generate precipitation with
    PrecipitationDistribution and then set Dr, Tr, and Tb. """

    def __init__(
        self,
        potential_evapotranspiration_rate=2e-7,
        upper_relative_saturation=0.5,
        lower_relative_saturation=0.1,
        profile_depth=5,
        porosity=0.1,
        num_bins=500,
        num_timesteps=100,
        mean_storm_depth=0.02,
        mean_storm_duration=1e3,
        mean_interstorm_duration=1e5,
        random_seed=None,
    ):

        self.d = mean_storm_depth  # mm
        self.tr = mean_storm_duration  # day
        self.tb = mean_interstorm_duration  # day
        self.pet = potential_evapotranspiration_rate  # mm/day
        self.Sfc = upper_relative_saturation
        self.Swp = lower_relative_saturation
        self.b = profile_depth  # mm
        self.n = porosity
        self.Nt = num_timesteps
        self.Nz = num_bins
        if random_seed:
            np.random.seed(random_seed)

        self.depths = np.linspace(
            self.b / self.Nz, self.b, self.Nz
        )  # bottom elev of each bin
        self.sat_profile = np.zeros_like(
            self.depths
        )  # saturation state (binary) in each bin
        self.sat_diff = np.zeros_like(self.depths)
        self.recharge_at_depth = np.zeros_like(self.depths)
        self.bin_capacity = (self.b / self.Nz) * self.n * (self.Sfc - self.Swp)

    def generate_all_storms(self):

        self.d_all = np.array([np.random.exponential(self.d) for i in range(self.Nt)])
        self.tr_all = np.array([np.random.exponential(self.tr) for i in range(self.Nt)])
        self.tb_all = np.array([np.random.exponential(self.tb) for i in range(self.Nt)])

    def generate_storm(self):

        self.Dr = np.random.exponential(self.d)
        self.Tr = np.random.exponential(self.tr)
        self.Tb = np.random.exponential(self.tb)

    def run_event(self, storm_depth):

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
        self.sat_diff[:] = 0

    def run_interevent(self, interstorm_dt):

        # number of bins ET will drain
        n_to_drain = round(self.pet * interstorm_dt / self.bin_capacity)

        # change bin status
        inds_to_drain = np.where(self.sat_profile == 1)[0][0:n_to_drain]
        self.sat_profile[inds_to_drain] = 0

    def run_one_step(self):

        self.generate_storm()
        self.run_event(self.Dr)
        self.run_interevent(self.Tb)

    def run_model(self):

        self.cum_recharge = np.zeros_like(self.depths)
        self.bool_recharge = np.zeros_like(self.depths)
        self.cum_storm_dt = 0
        self.cum_interstorm_dt = 0

        for i in range(self.Nt):

            self.run_one_step()

            self.cum_recharge += self.recharge_at_depth
            self.bool_recharge += self.recharge_at_depth > 0.0
            self.cum_storm_dt += self.Tr
            self.cum_interstorm_dt += self.Tb

        # define mean recharge depth as depth of events > 0 at that profile depth
        self.mean_recharge_depth = self.cum_recharge / self.bool_recharge
        # define requency as number of times recharge occurs over total time
        self.recharge_frequency = self.bool_recharge / (
            self.cum_storm_dt + self.cum_interstorm_dt
        )
