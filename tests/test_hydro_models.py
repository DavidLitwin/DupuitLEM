"""
Tests of the DupuitLEM steady and stochastic models, showing functionality
on raster and hexagonal grids, and ability to save output.

Date: 8 Oct 2020
"""

import numpy as np
from numpy.testing import assert_almost_equal, assert_equal

from landlab import HexModelGrid, RasterModelGrid
from landlab.components import (
    GroundwaterDupuitPercolator,
    PrecipitationDistribution,
)
from DupuitLEM.auxiliary_models import (
    SchenkVadoseModel,
    HydrologyEventStreamPower,
    HydrologySteadyStreamPower,
    HydrologyEventVadoseStreamPower,
    HydrologyEventThresholdStreamPower,
    HydrologyEventVadoseThresholdStreamPower,
)


# HydrologySteadyStreamPower
def test_steady_sp_raster():
    """
    Initialize HydrologySteadyStreamPower on a raster grid.
    After one timestep it returns all recharge as discharge.
    """

    mg = RasterModelGrid((3, 3), xy_spacing=10.0)
    mg.status_at_node[mg.status_at_node == 1] = 4
    mg.status_at_node[0] = 1
    mg.add_ones("node", "topographic__elevation")
    mg.add_zeros("node", "aquifer_base__elevation")
    mg.add_ones("node", "water_table__elevation")

    gdp = GroundwaterDupuitPercolator(mg, recharge_rate=1e-6)
    hm = HydrologySteadyStreamPower(mg, groundwater_model=gdp)

    hm.run_step()

    assert_almost_equal(hm.q[4], 1e-4)
    assert_almost_equal(hm.q_an[4], 1e-5)


def test_steady_sp_hex():
    """
    Initialize HydrologySteadyStreamPower on a hex grid.
    After one timestep it returns all recharge as discharge.
    """

    mg = HexModelGrid((3, 3), node_layout="rect", spacing=10.0)
    mg.status_at_node[mg.status_at_node == 1] = 4
    mg.status_at_node[0] = 1

    mg.add_ones("node", "topographic__elevation")
    mg.add_zeros("node", "aquifer_base__elevation")
    mg.add_ones("node", "water_table__elevation")

    gdp = GroundwaterDupuitPercolator(mg, recharge_rate=1e-6)
    hm = HydrologySteadyStreamPower(
        mg, groundwater_model=gdp, routing_method="Steepest"
    )

    hm.run_step()

    assert_almost_equal(hm.q[4], (np.sqrt(3) / 2) * 10 ** (-4))
    assert_almost_equal(hm.q_an[4], np.sqrt((np.sqrt(3) / 2)) * 10 ** (-5))


# HydrologyEventStreamPower
def test_stoch_sp_raster():
    """
    Initialize HydrologyEventStreamPower on a raster grid.
    Use single storm-interstorm pair and make sure it returns the
    quantity calculated. This is not an analytical solution, just
    the value that is returned when using gdp and adaptive
    timestep solver.
    """

    mg = RasterModelGrid((3, 3), xy_spacing=10.0)
    mg.set_status_at_node_on_edges(
        right=mg.BC_NODE_IS_CLOSED,
        top=mg.BC_NODE_IS_CLOSED,
        left=mg.BC_NODE_IS_CLOSED,
        bottom=mg.BC_NODE_IS_FIXED_VALUE,
    )
    mg.add_ones("node", "topographic__elevation")
    mg.add_zeros("node", "aquifer_base__elevation")
    mg.add_ones("node", "water_table__elevation")

    gdp = GroundwaterDupuitPercolator(mg, recharge_rate=1e-6)
    pd = PrecipitationDistribution(
        mg,
        mean_storm_duration=10,
        mean_interstorm_duration=100,
        mean_storm_depth=1e-3,
        total_t=100,
    )
    pd.seed_generator(seedval=1)
    hm = HydrologyEventStreamPower(mg, precip_generator=pd, groundwater_model=gdp)

    hm.run_step()

    assert_almost_equal(hm.q_eff[4], 0.000352283)
    assert_almost_equal(hm.q_an[4], 0.000352283 / 10.0)


def test_stoch_sp_hex():
    """
    Initialize HydrologyEventStreamPower on a hex grid.
    Use single storm-interstorm pair and make sure it returns the
    quantity calculated. This is not an analytical solution, just
    the value that is returned when using gdp and adaptive
    timestep solver. Confirms that hex grid returns the same value
    as raster grid, adjusted for cell area.
    """

    mg = HexModelGrid((3, 3), node_layout="rect", spacing=10.0)
    mg.status_at_node[mg.status_at_node == 1] = 4
    mg.status_at_node[0] = 1

    mg.add_ones("node", "topographic__elevation")
    mg.add_zeros("node", "aquifer_base__elevation")
    mg.add_ones("node", "water_table__elevation")

    gdp = GroundwaterDupuitPercolator(mg, recharge_rate=1e-6)
    pd = PrecipitationDistribution(
        mg,
        mean_storm_duration=10,
        mean_interstorm_duration=100,
        mean_storm_depth=1e-3,
        total_t=100,
    )
    pd.seed_generator(seedval=1)
    hm = HydrologyEventStreamPower(
        mg, precip_generator=pd, groundwater_model=gdp, routing_method="Steepest"
    )

    hm.run_step()

    assert_almost_equal(hm.q_eff[4], 0.000352283 * np.sqrt(3) / 2)
    assert_almost_equal(
        hm.q_an[4], 0.000352283 * np.sqrt(3) / 2 / np.sqrt(np.sqrt(3) / 2 * 100)
    )


def test_stoch_sp_raster_record_state():
    """
    Initialize HydrologyEventStreamPower on a raster grid.
    Use several storm-interstorm pairs and make sure state recorded
    as expected.
    """

    mg = RasterModelGrid((3, 3), xy_spacing=10.0)
    mg.set_status_at_node_on_edges(
        right=mg.BC_NODE_IS_CLOSED,
        top=mg.BC_NODE_IS_CLOSED,
        left=mg.BC_NODE_IS_CLOSED,
        bottom=mg.BC_NODE_IS_FIXED_VALUE,
    )
    mg.add_ones("node", "topographic__elevation")
    mg.add_zeros("node", "aquifer_base__elevation")
    wt = mg.add_ones("node", "water_table__elevation")

    gdp = GroundwaterDupuitPercolator(mg, recharge_rate=1e-6)
    pd = PrecipitationDistribution(
        mg,
        mean_storm_duration=10,
        mean_interstorm_duration=100,
        mean_storm_depth=1e-3,
        total_t=200,
    )
    pd.seed_generator(seedval=1)
    hm = HydrologyEventStreamPower(mg, precip_generator=pd, groundwater_model=gdp)

    wt0 = wt.copy()
    hm.run_step_record_state()

    times = np.array(
        [
            0.0,
            hm.storm_dts[0],
            hm.storm_dts[0] + hm.interstorm_dts[0],
            hm.storm_dts[0] + hm.interstorm_dts[0] + hm.storm_dts[1],
            hm.storm_dts[0]
            + hm.interstorm_dts[0]
            + hm.storm_dts[1]
            + hm.interstorm_dts[1],
        ]
    )
    intensities = np.zeros(5)
    intensities[0] = hm.intensities[0]
    intensities[2] = hm.intensities[1]

    assert_equal(hm.time, times)
    assert_equal(hm.intensity, intensities)

    assert_equal(hm.qs_all.shape, (5, 9))
    assert_equal(hm.Q_all.shape, (5, 9))
    assert_equal(hm.wt_all.shape, (5, 9))

    assert_equal(hm.qs_all[0, :], np.zeros(9))
    assert_equal(hm.Q_all[0, :], np.zeros(9))
    assert_equal(hm.wt_all[0, :], wt0)


def test_stoch_sp_methods_same():
    """
    Initialize HydrologyEventStreamPower on a raster grid.
    Use several storm-interstorm pairs and make sure run_step_record_state
    method gives the same answer as run_step method.
    """

    mg = RasterModelGrid((3, 3), xy_spacing=10.0)
    mg.set_status_at_node_on_edges(
        right=mg.BC_NODE_IS_CLOSED,
        top=mg.BC_NODE_IS_CLOSED,
        left=mg.BC_NODE_IS_CLOSED,
        bottom=mg.BC_NODE_IS_FIXED_VALUE,
    )
    mg.add_ones("node", "topographic__elevation")
    mg.add_zeros("node", "aquifer_base__elevation")
    mg.add_ones("node", "water_table__elevation")

    gdp = GroundwaterDupuitPercolator(mg, porosity=0.2)
    pd = PrecipitationDistribution(
        mg,
        mean_storm_duration=10,
        mean_interstorm_duration=100,
        mean_storm_depth=1e-3,
        total_t=1000,
    )
    pd.seed_generator(seedval=1)
    hm = HydrologyEventStreamPower(mg, precip_generator=pd, groundwater_model=gdp)
    hm.run_step_record_state()

    mg1 = RasterModelGrid((3, 3), xy_spacing=10.0)
    mg1.set_status_at_node_on_edges(
        right=mg1.BC_NODE_IS_CLOSED,
        top=mg1.BC_NODE_IS_CLOSED,
        left=mg1.BC_NODE_IS_CLOSED,
        bottom=mg1.BC_NODE_IS_FIXED_VALUE,
    )
    mg1.add_ones("node", "topographic__elevation")
    mg1.add_zeros("node", "aquifer_base__elevation")
    mg1.add_ones("node", "water_table__elevation")

    gdp1 = GroundwaterDupuitPercolator(mg1, porosity=0.2)
    pd1 = PrecipitationDistribution(
        mg1,
        mean_storm_duration=10,
        mean_interstorm_duration=100,
        mean_storm_depth=1e-3,
        total_t=1000,
    )
    pd1.seed_generator(seedval=1)
    hm1 = HydrologyEventStreamPower(mg1, precip_generator=pd1, groundwater_model=gdp1)
    hm1.run_step()

    assert_equal(hm.q_eff, hm1.q_eff)


# HydrologyEventThresholdStreamPower
def test_stoch_sp_threshold_raster_null():
    """
    Initialize HydrologyEventThresholdStreamPower on a raster grid.
    Use single storm-interstorm pair and make sure it returns the
    quantity calculated. This is not an analytical solution, just
    the value that is returned when using gdp and adaptive
    timestep solver. Confirms that when streampower threshold is
    zero (Default), returns the same values as HydrologyEventStreamPower.
    """

    mg = RasterModelGrid((3, 3), xy_spacing=10.0)
    mg.set_status_at_node_on_edges(
        right=mg.BC_NODE_IS_CLOSED,
        top=mg.BC_NODE_IS_CLOSED,
        left=mg.BC_NODE_IS_CLOSED,
        bottom=mg.BC_NODE_IS_FIXED_VALUE,
    )
    mg.add_ones("node", "topographic__elevation")
    mg.add_zeros("node", "aquifer_base__elevation")
    mg.add_ones("node", "water_table__elevation")

    gdp = GroundwaterDupuitPercolator(mg, recharge_rate=1e-6)
    pd = PrecipitationDistribution(
        mg,
        mean_storm_duration=10,
        mean_interstorm_duration=100,
        mean_storm_depth=1e-3,
        total_t=100,
    )
    pd.seed_generator(seedval=1)
    hm = HydrologyEventThresholdStreamPower(
        mg, precip_generator=pd, groundwater_model=gdp
    )

    hm.run_step()

    assert_almost_equal(hm.q_eff[4], 0.000352283)
    assert_almost_equal(hm.q_an[4], 0.000352283 / 10.0)


def test_stoch_sp_threshold_hex():
    """
    Initialize HydrologyEventStreamPower on a hex grid.
    Use single storm-interstorm pair and make sure it returns the
    quantity calculated. This is not an analytical solution, just
    the value that is returned when using gdp and adaptive
    timestep solver. Confirms that hex grid returns the same value
    as raster grid, adjusted for cell area. Confirms that when streampower
    threshold is zero (Default), returns the same values as
    HydrologyEventStreamPower.
    """

    mg = HexModelGrid((3, 3), node_layout="rect", spacing=10.0)
    mg.status_at_node[mg.status_at_node == 1] = 4
    mg.status_at_node[0] = 1

    mg.add_ones("node", "topographic__elevation")
    mg.add_zeros("node", "aquifer_base__elevation")
    mg.add_ones("node", "water_table__elevation")

    gdp = GroundwaterDupuitPercolator(mg, recharge_rate=1e-6)
    pd = PrecipitationDistribution(
        mg,
        mean_storm_duration=10,
        mean_interstorm_duration=100,
        mean_storm_depth=1e-3,
        total_t=100,
    )
    pd.seed_generator(seedval=1)
    hm = HydrologyEventThresholdStreamPower(
        mg, precip_generator=pd, groundwater_model=gdp, routing_method="Steepest"
    )

    hm.run_step()

    assert_almost_equal(hm.q_eff[4], 0.000352283 * np.sqrt(3) / 2)
    assert_almost_equal(
        hm.q_an[4], 0.000352283 * np.sqrt(3) / 2 / np.sqrt(np.sqrt(3) / 2 * 100)
    )


def test_stoch_sp_threshold_below_threshold():
    """
    Test the stochastic event model with stream power threshold in which
    the one core node is set up to not exceed erosion threshold for the value
    of Q that it attains. This can be checked by comparing the accumulated q
    to the threshold value needed for erosion Q0.
    """

    mg = RasterModelGrid((3, 3), xy_spacing=10.0)
    mg.set_status_at_node_on_edges(
        right=mg.BC_NODE_IS_CLOSED,
        top=mg.BC_NODE_IS_CLOSED,
        left=mg.BC_NODE_IS_CLOSED,
        bottom=mg.BC_NODE_IS_FIXED_VALUE,
    )
    elev = mg.add_ones("node", "topographic__elevation")
    mg.add_zeros("node", "aquifer_base__elevation")
    wt = mg.add_ones("node", "water_table__elevation")
    elev[4] += 0.01
    wt[:] = elev

    gdp = GroundwaterDupuitPercolator(mg, recharge_rate=1e-6)
    pd = PrecipitationDistribution(
        mg,
        mean_storm_duration=10,
        mean_interstorm_duration=100,
        mean_storm_depth=1e-3,
        total_t=100,
    )
    pd.seed_generator(seedval=1)
    hm = HydrologyEventThresholdStreamPower(
        mg,
        precip_generator=pd,
        groundwater_model=gdp,
        sp_coefficient=1e-5,
        sp_threshold=1e-10,
    )

    hm.run_step()

    assert_almost_equal(hm.q_eff[4], 0.0)


def test_stoch_sp_threshold_above_threshold():
    """
    Test the stochastic event model with stream power threshold in which
    the one core node is set up to exceed erosion threshold for the value
    of Q that it attains. This can be checked by comparing the accumulated q
    to the threshold value needed for erosion Q0.
    """

    mg = RasterModelGrid((3, 3), xy_spacing=10.0)
    mg.set_status_at_node_on_edges(
        right=mg.BC_NODE_IS_CLOSED,
        top=mg.BC_NODE_IS_CLOSED,
        left=mg.BC_NODE_IS_CLOSED,
        bottom=mg.BC_NODE_IS_FIXED_VALUE,
    )
    elev = mg.add_ones("node", "topographic__elevation")
    mg.add_zeros("node", "aquifer_base__elevation")
    wt = mg.add_ones("node", "water_table__elevation")
    elev[4] += 0.01
    wt[:] = elev

    gdp = GroundwaterDupuitPercolator(mg, recharge_rate=1e-6)
    pd = PrecipitationDistribution(
        mg,
        mean_storm_duration=10,
        mean_interstorm_duration=100,
        mean_storm_depth=1e-3,
        total_t=100,
    )
    pd.seed_generator(seedval=1)
    hm = HydrologyEventThresholdStreamPower(
        mg,
        precip_generator=pd,
        groundwater_model=gdp,
        sp_coefficient=1e-5,
        sp_threshold=1e-12,
    )

    hm.run_step()

    storm_dt = 1.4429106411  # storm duration
    storm_q = 0.0244046740  # accumulated q before threshold effect subtracted
    # interstorm_q = 0.0  # interstorm q is zero in this case
    assert_almost_equal(
        hm.q_eff[4],
        max(storm_q - hm.Q0[4], 0) * storm_dt / hm.T_h,
    )


def test_stoch_sp_threshold_record_state():
    """
    Initialize HydrologyEventThresholdStreamPower on a raster grid.
    Use several storm-interstorm pairs and make sure state recorded
    as expected.
    """

    mg = RasterModelGrid((3, 3), xy_spacing=10.0)
    mg.set_status_at_node_on_edges(
        right=mg.BC_NODE_IS_CLOSED,
        top=mg.BC_NODE_IS_CLOSED,
        left=mg.BC_NODE_IS_CLOSED,
        bottom=mg.BC_NODE_IS_FIXED_VALUE,
    )
    mg.add_ones("node", "topographic__elevation")
    mg.add_zeros("node", "aquifer_base__elevation")
    wt = mg.add_ones("node", "water_table__elevation")

    gdp = GroundwaterDupuitPercolator(mg, recharge_rate=1e-6)
    pd = PrecipitationDistribution(
        mg,
        mean_storm_duration=10,
        mean_interstorm_duration=100,
        mean_storm_depth=1e-3,
        total_t=200,
    )
    pd.seed_generator(seedval=1)
    hm = HydrologyEventThresholdStreamPower(
        mg, precip_generator=pd, groundwater_model=gdp
    )

    wt0 = wt.copy()
    hm.run_step_record_state()

    times = np.array(
        [
            0.0,
            hm.storm_dts[0],
            hm.storm_dts[0] + hm.interstorm_dts[0],
            hm.storm_dts[0] + hm.interstorm_dts[0] + hm.storm_dts[1],
            hm.storm_dts[0]
            + hm.interstorm_dts[0]
            + hm.storm_dts[1]
            + hm.interstorm_dts[1],
        ]
    )
    intensities = np.zeros(5)
    intensities[0] = hm.intensities[0]
    intensities[2] = hm.intensities[1]

    assert_equal(hm.time, times)
    assert_equal(hm.intensity, intensities)

    assert_equal(hm.qs_all.shape, (5, 9))
    assert_equal(hm.Q_all.shape, (5, 9))
    assert_equal(hm.wt_all.shape, (5, 9))

    assert_equal(hm.qs_all[0, :], np.zeros(9))
    assert_equal(hm.Q_all[0, :], np.zeros(9))
    assert_equal(hm.wt_all[0, :], wt0)


# HydrologyEventVadoseStreamPower
def test_stoch_sp_vadose_raster_null():
    """
    Initialize HydrologyEventVadoseStreamPower on a raster grid.
    Use single storm-interstorm pair and make sure it returns the
    quantity calculated. This is not an analytical solution, just
    the value that is returned when using gdp and adaptive
    timestep solver. Note that vadose bin number has to be large,
    because events smaller than bin size don't register.
    """

    mg = RasterModelGrid((3, 3), xy_spacing=10.0)
    mg.set_status_at_node_on_edges(
        right=mg.BC_NODE_IS_CLOSED,
        top=mg.BC_NODE_IS_CLOSED,
        left=mg.BC_NODE_IS_CLOSED,
        bottom=mg.BC_NODE_IS_FIXED_VALUE,
    )
    mg.add_ones("node", "topographic__elevation")
    mg.add_zeros("node", "aquifer_base__elevation")
    mg.add_ones("node", "water_table__elevation")

    gdp = GroundwaterDupuitPercolator(mg, porosity=0.2)
    pd = PrecipitationDistribution(
        mg,
        mean_storm_duration=10,
        mean_interstorm_duration=100,
        mean_storm_depth=1e-3,
        total_t=100,
    )
    pd.seed_generator(seedval=1)
    svm = SchenkVadoseModel(
        potential_evapotranspiration_rate=0.0,
        profile_depth=1.0,
        num_bins=int(1e6),
    )
    svm.sat_profile[:] = 1.0  # start initially saturated
    hm = HydrologyEventVadoseStreamPower(
        mg,
        precip_generator=pd,
        groundwater_model=gdp,
        vadose_model=svm,
    )
    hm.run_step()

    assert_almost_equal(hm.q_eff[4], 0.000352283)
    assert_almost_equal(hm.q_an[4], 0.000352283 / 10.0)


def test_stoch_sp_vadose_hex_null():
    """
    Initialize HydrologyEventVadoseStreamPower on a hex grid.
    Use single storm-interstorm pair and make sure it returns the
    quantity calculated. This is not an analytical solution, just
    the value that is returned when using gdp and adaptive
    timestep solver. Note that vadose bin number has to be large,
    because events smaller than bin size don't register.
    """

    mg = HexModelGrid((3, 3), node_layout="rect", spacing=10.0)
    mg.status_at_node[mg.status_at_node == 1] = 4
    mg.status_at_node[0] = 1

    mg.add_ones("node", "topographic__elevation")
    mg.add_zeros("node", "aquifer_base__elevation")
    mg.add_ones("node", "water_table__elevation")

    gdp = GroundwaterDupuitPercolator(mg, porosity=0.2)
    pd = PrecipitationDistribution(
        mg,
        mean_storm_duration=10,
        mean_interstorm_duration=100,
        mean_storm_depth=1e-3,
        total_t=100,
    )
    pd.seed_generator(seedval=1)
    svm = SchenkVadoseModel(
        potential_evapotranspiration_rate=0.0,
        profile_depth=1.0,
        num_bins=int(1e6),
    )
    svm.sat_profile[:] = 1.0  # start initially saturated
    hm = HydrologyEventVadoseStreamPower(
        mg,
        precip_generator=pd,
        groundwater_model=gdp,
        vadose_model=svm,
        routing_method="Steepest",
    )
    hm.run_step()

    assert_almost_equal(hm.q_eff[4], 0.000352283 * np.sqrt(3) / 2)
    assert_almost_equal(
        hm.q_an[4], 0.000352283 * np.sqrt(3) / 2 / np.sqrt(np.sqrt(3) / 2 * 100)
    )


def test_stoch_sp_vadose_record_state():
    """
    Initialize HydrologyEventVadoseStreamPower on a raster grid.
    Use several storm-interstorm pairs and make sure state recorded
    as expected.
    """

    mg = RasterModelGrid((3, 3), xy_spacing=10.0)
    mg.set_status_at_node_on_edges(
        right=mg.BC_NODE_IS_CLOSED,
        top=mg.BC_NODE_IS_CLOSED,
        left=mg.BC_NODE_IS_CLOSED,
        bottom=mg.BC_NODE_IS_FIXED_VALUE,
    )
    mg.add_ones("node", "topographic__elevation")
    mg.add_zeros("node", "aquifer_base__elevation")
    wt = mg.add_ones("node", "water_table__elevation")

    gdp = GroundwaterDupuitPercolator(mg, porosity=0.2)
    pd = PrecipitationDistribution(
        mg,
        mean_storm_duration=10,
        mean_interstorm_duration=100,
        mean_storm_depth=1e-3,
        total_t=200,
    )
    pd.seed_generator(seedval=1)
    svm = SchenkVadoseModel(
        potential_evapotranspiration_rate=0.0,
        profile_depth=1.0,
        num_bins=int(1e6),
    )
    svm.sat_profile[:] = 1.0  # start initially saturated
    hm = HydrologyEventVadoseStreamPower(
        mg,
        precip_generator=pd,
        groundwater_model=gdp,
        vadose_model=svm,
    )

    wt0 = wt.copy()
    hm.run_step_record_state()

    times = np.array(
        [
            0.0,
            hm.storm_dts[0],
            hm.storm_dts[0] + hm.interstorm_dts[0],
            hm.storm_dts[0] + hm.interstorm_dts[0] + hm.storm_dts[1],
            hm.storm_dts[0]
            + hm.interstorm_dts[0]
            + hm.storm_dts[1]
            + hm.interstorm_dts[1],
        ]
    )
    intensities = np.zeros(5)
    intensities[0] = hm.intensities[0]
    intensities[2] = hm.intensities[1]

    assert_equal(hm.time, times)
    assert_equal(hm.intensity, intensities)

    assert_equal(hm.qs_all.shape, (5, 9))
    assert_equal(hm.Q_all.shape, (5, 9))
    assert_equal(hm.wt_all.shape, (5, 9))

    assert_equal(hm.qs_all[0, :], np.zeros(9))
    assert_equal(hm.Q_all[0, :], np.zeros(9))
    assert_equal(hm.wt_all[0, :], wt0)


def test_stoch_sp_vadose_methods_same():
    """
    Initialize HydrologyEventVadoseStreamPower on a raster grid.
    Use several storm-interstorm pairs and make sure run_step_record_state
    method gives the same answer as run_step method.
    """

    mg = RasterModelGrid((3, 3), xy_spacing=10.0)
    mg.set_status_at_node_on_edges(
        right=mg.BC_NODE_IS_CLOSED,
        top=mg.BC_NODE_IS_CLOSED,
        left=mg.BC_NODE_IS_CLOSED,
        bottom=mg.BC_NODE_IS_FIXED_VALUE,
    )
    mg.add_ones("node", "topographic__elevation")
    mg.add_zeros("node", "aquifer_base__elevation")
    mg.add_ones("node", "water_table__elevation")

    gdp = GroundwaterDupuitPercolator(mg, porosity=0.2)
    pd = PrecipitationDistribution(
        mg,
        mean_storm_duration=10,
        mean_interstorm_duration=100,
        mean_storm_depth=1e-3,
        total_t=1000,
    )
    pd.seed_generator(seedval=1)
    svm = SchenkVadoseModel(
        potential_evapotranspiration_rate=0.0,
        profile_depth=1.0,
        num_bins=int(1e4),
    )
    svm.sat_profile[:] = 1.0  # start initially saturated
    hm = HydrologyEventVadoseStreamPower(
        mg,
        precip_generator=pd,
        groundwater_model=gdp,
        vadose_model=svm,
    )
    hm.run_step_record_state()

    mg1 = RasterModelGrid((3, 3), xy_spacing=10.0)
    mg1.set_status_at_node_on_edges(
        right=mg1.BC_NODE_IS_CLOSED,
        top=mg1.BC_NODE_IS_CLOSED,
        left=mg1.BC_NODE_IS_CLOSED,
        bottom=mg1.BC_NODE_IS_FIXED_VALUE,
    )
    mg1.add_ones("node", "topographic__elevation")
    mg1.add_zeros("node", "aquifer_base__elevation")
    mg1.add_ones("node", "water_table__elevation")

    gdp1 = GroundwaterDupuitPercolator(mg1, porosity=0.2)
    pd1 = PrecipitationDistribution(
        mg1,
        mean_storm_duration=10,
        mean_interstorm_duration=100,
        mean_storm_depth=1e-3,
        total_t=1000,
    )
    pd1.seed_generator(seedval=1)
    svm1 = SchenkVadoseModel(
        potential_evapotranspiration_rate=0.0,
        profile_depth=1.0,
        num_bins=int(1e4),
    )
    svm1.sat_profile[:] = 1.0  # start initially saturated
    hm1 = HydrologyEventVadoseStreamPower(
        mg1,
        precip_generator=pd1,
        groundwater_model=gdp1,
        vadose_model=svm1,
    )
    hm1.run_step()

    assert_equal(hm.q_eff, hm1.q_eff)


# HydrologyEventVadoseThresholdStreamPower
def test_stoch_sp_vadose_threshold_raster_null():
    """
    Initialize HydrologyEventVadoseThresholdStreamPower on a raster grid.
    Use single storm-interstorm pair and make sure it returns the
    quantity calculated. This is not an analytical solution, just
    the value that is returned when using gdp and adaptive
    timestep solver. Note that vadose bin number has to be large,
    because events smaller than bin size don't register. Threshold
    for incision is zero.
    """

    mg = RasterModelGrid((3, 3), xy_spacing=10.0)
    mg.set_status_at_node_on_edges(
        right=mg.BC_NODE_IS_CLOSED,
        top=mg.BC_NODE_IS_CLOSED,
        left=mg.BC_NODE_IS_CLOSED,
        bottom=mg.BC_NODE_IS_FIXED_VALUE,
    )
    mg.add_ones("node", "topographic__elevation")
    mg.add_zeros("node", "aquifer_base__elevation")
    mg.add_ones("node", "water_table__elevation")

    gdp = GroundwaterDupuitPercolator(mg, porosity=0.2)
    pd = PrecipitationDistribution(
        mg,
        mean_storm_duration=10,
        mean_interstorm_duration=100,
        mean_storm_depth=1e-3,
        total_t=100,
    )
    pd.seed_generator(seedval=1)
    svm = SchenkVadoseModel(
        potential_evapotranspiration_rate=0.0,
        profile_depth=1.0,
        num_bins=int(1e6),
    )
    svm.sat_profile[:] = 1.0  # start initially saturated
    hm = HydrologyEventVadoseThresholdStreamPower(
        mg,
        precip_generator=pd,
        groundwater_model=gdp,
        vadose_model=svm,
        routing_method="Steepest",
    )
    hm.run_step()

    assert_almost_equal(hm.q_eff[4], 0.000352283)
    assert_almost_equal(hm.q_an[4], 0.000352283 / 10.0)


def test_stoch_sp_vadose_threshold_hex_null():
    """
    Initialize HydrologyEventVadoseThresholdStreamPower on a hex grid.
    Use single storm-interstorm pair and make sure it returns the
    quantity calculated. This is not an analytical solution, just
    the value that is returned when using gdp and adaptive
    timestep solver. Note that vadose bin number has to be large,
    because events smaller than bin size don't register. Threshold
    for incision is zero.
    """

    mg = HexModelGrid((3, 3), node_layout="rect", spacing=10.0)
    mg.status_at_node[mg.status_at_node == 1] = 4
    mg.status_at_node[0] = 1

    mg.add_ones("node", "topographic__elevation")
    mg.add_zeros("node", "aquifer_base__elevation")
    mg.add_ones("node", "water_table__elevation")

    gdp = GroundwaterDupuitPercolator(mg, porosity=0.2)
    pd = PrecipitationDistribution(
        mg,
        mean_storm_duration=10,
        mean_interstorm_duration=100,
        mean_storm_depth=1e-3,
        total_t=100,
    )
    pd.seed_generator(seedval=1)
    svm = SchenkVadoseModel(
        potential_evapotranspiration_rate=0.0,
        profile_depth=1.0,
        num_bins=int(1e6),
    )
    svm.sat_profile[:] = 1.0  # start initially saturated
    hm = HydrologyEventVadoseThresholdStreamPower(
        mg,
        precip_generator=pd,
        groundwater_model=gdp,
        vadose_model=svm,
        routing_method="Steepest",
    )
    hm.run_step()
    assert_almost_equal(hm.q_eff[4], 0.000352283 * np.sqrt(3) / 2)
    assert_almost_equal(
        hm.q_an[4], 0.000352283 * np.sqrt(3) / 2 / np.sqrt(np.sqrt(3) / 2 * 100)
    )


def test_stoch_sp_vadose_threshold_below_threshold():
    """
    Test the stochastic event model with stream power threshold in which
    the one core node is set up to not exceed erosion threshold for the value
    of Q that it attains. This can be checked by comparing the accumulated q
    to the threshold value needed for erosion Q0. Vadose model is in a null
    setup where all infiltration becomes recharge.
    """

    mg = RasterModelGrid((3, 3), xy_spacing=10.0)
    mg.set_status_at_node_on_edges(
        right=mg.BC_NODE_IS_CLOSED,
        top=mg.BC_NODE_IS_CLOSED,
        left=mg.BC_NODE_IS_CLOSED,
        bottom=mg.BC_NODE_IS_FIXED_VALUE,
    )
    elev = mg.add_ones("node", "topographic__elevation")
    mg.add_zeros("node", "aquifer_base__elevation")
    wt = mg.add_ones("node", "water_table__elevation")
    elev[4] += 0.01
    wt[:] = elev

    gdp = GroundwaterDupuitPercolator(mg, porosity=0.2)
    pd = PrecipitationDistribution(
        mg,
        mean_storm_duration=10,
        mean_interstorm_duration=100,
        mean_storm_depth=1e-3,
        total_t=100,
    )
    pd.seed_generator(seedval=1)
    svm = SchenkVadoseModel(
        potential_evapotranspiration_rate=0.0,
        profile_depth=1.0,
        num_bins=int(1e6),
    )
    svm.sat_profile[:] = 1.0  # start initially saturated
    hm = HydrologyEventVadoseThresholdStreamPower(
        mg,
        precip_generator=pd,
        groundwater_model=gdp,
        vadose_model=svm,
        sp_coefficient=1e-5,
        sp_threshold=1e-10,
    )
    hm.run_step()

    assert_almost_equal(hm.q_eff[4], 0.0)


def test_stoch_sp_vadose_threshold_above_threshold():
    """
    Test the stochastic event model with stream power threshold in which
    the one core node is set up to exceed erosion threshold for the value
    of Q that it attains. This can be checked by comparing the accumulated q
    to the threshold value needed for erosion Q0. Vadose model is in a null
    setup where all infiltration becomes recharge.
    """

    mg = RasterModelGrid((3, 3), xy_spacing=10.0)
    mg.set_status_at_node_on_edges(
        right=mg.BC_NODE_IS_CLOSED,
        top=mg.BC_NODE_IS_CLOSED,
        left=mg.BC_NODE_IS_CLOSED,
        bottom=mg.BC_NODE_IS_FIXED_VALUE,
    )
    elev = mg.add_ones("node", "topographic__elevation")
    mg.add_zeros("node", "aquifer_base__elevation")
    wt = mg.add_ones("node", "water_table__elevation")
    elev[4] += 0.01
    wt[:] = elev

    gdp = GroundwaterDupuitPercolator(mg, porosity=0.2)
    pd = PrecipitationDistribution(
        mg,
        mean_storm_duration=10,
        mean_interstorm_duration=100,
        mean_storm_depth=1e-3,
        total_t=100,
    )
    pd.seed_generator(seedval=1)
    svm = SchenkVadoseModel(
        potential_evapotranspiration_rate=0.0,
        profile_depth=1.0,
        num_bins=int(1e6),
    )
    svm.sat_profile[:] = 1.0  # start initially saturated
    hm = HydrologyEventVadoseThresholdStreamPower(
        mg,
        precip_generator=pd,
        groundwater_model=gdp,
        vadose_model=svm,
        sp_coefficient=1e-5,
        sp_threshold=1e-12,
    )
    hm.run_step()

    storm_dt = 1.4429106411  # storm duration
    storm_q = 0.0244046740  # accumulated q before threshold effect subtracted
    # interstorm_q = 0.0  # interstorm q is zero in this case
    assert_almost_equal(
        hm.q_eff[4],
        max(storm_q - hm.Q0[4], 0) * storm_dt / hm.T_h,
    )
