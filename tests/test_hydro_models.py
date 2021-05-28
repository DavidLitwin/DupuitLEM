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
    HydrologyEventStreamPower,
    HydrologySteadyStreamPower,
)


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

    assert_almost_equal(hm.q_eff[4], 0.00017614)
    assert_almost_equal(hm.q_an[4], 0.00017614 / 10.0)


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

    assert_almost_equal(hm.q_eff[4], 0.00017614 * np.sqrt(3) / 2)
    assert_almost_equal(
        hm.q_an[4], 0.00017614 * np.sqrt(3) / 2 / np.sqrt(np.sqrt(3) / 2 * 100)
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
