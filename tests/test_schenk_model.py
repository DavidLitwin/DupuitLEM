"""
Tests of SchenkVadoseModel, implemented for DupuitLEM.

Date: 28 May 2021
"""

import numpy as np
from numpy.testing import assert_almost_equal, assert_equal

from DupuitLEM.auxiliary_models import SchenkVadoseModel


def test_init_depth_profile():
    """"Check depths and bin capacity. Depths correspond to the bottom
    of each bin. Bin capacity is bin depth * available_water_content."""

    sm = SchenkVadoseModel(num_bins=5)

    assert_equal(sm.depths, [1.0, 2.0, 3.0, 4.0, 5.0])
    assert_almost_equal(sm.bin_capacity, 0.15)


def test_generate_storm():
    """Make sure seed is set when generating a storm event."""

    sm1 = SchenkVadoseModel()
    sm1.generate_storm(1, 0.1, 1, random_seed=5)
    dr, tr, tb = sm1.Dr, sm1.Tr, sm1.Tb
    sm2 = SchenkVadoseModel()
    sm2.generate_storm(1, 0.1, 1, random_seed=5)

    assert_equal([dr, tr, tb], [sm2.Dr, sm2.Tr, sm2.Tb])


def test_run_event_0():
    """"Trivial case of an event. Event depth is less than half bin volume so
    sat profile and recharge at depth are unchanged."""

    sm = SchenkVadoseModel(num_bins=5, available_water_content=1.0)
    sm.sat_profile[:] = np.array([1.0, 1.0, 0.0, 0.0, 0.0])
    sm.run_event(0.49 * sm.bin_capacity)

    assert_equal(sm.sat_profile, [1.0, 1.0, 0.0, 0.0, 0.0])  # binary
    assert_equal(sm.recharge_at_depth, [0.0, 0.0, 0.0, 0.0, 0.0])


def test_run_event_1():
    """"simple case of an event. An event with depth 2 fills the top 2 bins
    when bins have unit depth and can be filled 100% with water. Recharge is
    calculated on a 'floor' basis."""

    sm = SchenkVadoseModel(num_bins=5, available_water_content=0.25)
    sm.run_event(0.5)

    assert_equal(sm.sat_profile, [1.0, 1.0, 0.0, 0.0, 0.0])  # binary
    assert_equal(sm.recharge_at_depth, [0.25, 0.0, 0.0, 0.0, 0.0])


def test_run_event_2():
    """Slightly less trivial example of an event where there is already
    some inital nonuniform saturation. Ensure that the topmost available bins
    are filled."""

    sm = SchenkVadoseModel(num_bins=5, available_water_content=1.0)
    sm.sat_profile[:] = np.array([0.0, 1.0, 0.0, 0.0, 1.0])
    sm.run_event(2.0)

    assert_equal(sm.sat_profile, [1.0, 1.0, 1.0, 0.0, 1.0])
    assert_equal(sm.recharge_at_depth, [1.0, 1.0, 0.0, 0.0, 0.0])


def test_run_event_3():
    """Edge case where event depth is added to a profile that is already
    saturated. In this case, the profile should stay saturated, but the
    recharge should reflect the total amount. That is, a water table
    at any depth will recieve recharge."""

    sm = SchenkVadoseModel(num_bins=5, available_water_content=1.0)
    sm.sat_profile[:] = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    sm.run_event(2.0)

    assert_equal(sm.sat_profile, [1.0, 1.0, 1.0, 1.0, 1.0])
    assert_equal(sm.recharge_at_depth, [2.0, 2.0, 2.0, 2.0, 2.0])


def test_run_event_4():
    """Event where there is already some inital nonuniform saturation and
    event is larger than storage. Ensure all bins are filled and recharge is
    varying correctly with depth."""

    sm = SchenkVadoseModel(num_bins=5, available_water_content=1.0)
    sm.sat_profile[:] = np.array([0.0, 1.0, 0.0, 0.0, 1.0])
    sm.run_event(5.0)

    assert_equal(sm.sat_profile, [1.0, 1.0, 1.0, 1.0, 1.0])
    assert_equal(sm.recharge_at_depth, [4.0, 4.0, 3.0, 2.0, 2.0])


def test_run_interevent_0():
    """ Trivial case of an interevent. PET over interval is less than
    half bin capacity so sat profile remains unchanged."""

    sm = SchenkVadoseModel(
        num_bins=5, available_water_content=1.0, potential_evapotranspiration_rate=1.0,
    )
    sm.sat_profile[:] = np.array([1, 1, 1, 0, 0])
    sm.run_interevent(0.49)

    assert_equal(sm.sat_profile, [1, 1, 1, 0, 0])


def test_run_interevent_1():
    """ Simple case of an interevent. With unit PET rate, the top two
    bins are drained after 2 units of time when bins have unit depth and
    can be filled 100% with water. Note recharge_at_depth is not reset."""

    sm = SchenkVadoseModel(
        num_bins=5,
        available_water_content=0.25,
        potential_evapotranspiration_rate=0.25,
    )
    sm.sat_profile[:] = np.array([1, 1, 1, 0, 0])
    sm.run_interevent(2.0)

    assert_equal(sm.sat_profile, [0.0, 0.0, 1, 0.0, 0.0])


def test_run_interevent_2():
    """Edge case where interevent PET exceeds available saturation. In this
    case, the remaining saturation is drained."""

    sm = SchenkVadoseModel(
        num_bins=5, available_water_content=1.0, potential_evapotranspiration_rate=1.0,
    )
    sm.sat_profile[:] = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
    sm.run_interevent(4.0)

    assert_equal(sm.sat_profile, [0.0, 0.0, 0.0, 0.0, 0.0])


def test_run_interevent_3():
    """Edge case where interevent PET exceeds available saturation. In this
    case, the remaining saturation is drained."""

    sm = SchenkVadoseModel(
        num_bins=5, available_water_content=1.0, potential_evapotranspiration_rate=1.0,
    )
    sm.sat_profile[:] = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    sm.run_interevent(4.0)

    assert_equal(sm.sat_profile, [0.0, 0.0, 0.0, 0.0, 0.0])


def test_recharge_event_1():
    """test_run_event_2 but now calculate the recharge for float and arrays
    of water table depths. Recharge at surface should be same as in first bin."""

    sm = SchenkVadoseModel(num_bins=5, available_water_content=1.0)
    sm.sat_profile[:] = np.array([0.0, 1.0, 0.0, 0.0, 1.0])
    sm.run_event(2.0)

    # sm.depths: [ 1.,  2.,  3.,  4.,  5.]
    # sm.sat_profile: [1.0, 1.0, 1.0, 0.0, 1.0]
    # sm.recharge_at_depth: [1.0, 1.0, 0.0, 0.0, 0.0]

    R = sm.calc_recharge_rate(np.array([0.0, 0.5, 1.5, 2.5, 3.5, 4.5]), 5.0)
    assert_equal(R, [0.2, 0.2, 0.2, 0.0, 0.0, 0.0])


def test_recharge_event_2():
    """test_run_event_3 but now calculate the recharge for float and arrays
    of water table depths. Ensure depths beyond profile don't receive recharge."""

    sm = SchenkVadoseModel(num_bins=5, available_water_content=1.0)
    sm.sat_profile[:] = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    sm.run_event(2.0)

    # sm.depths: [ 1.,  2.,  3.,  4.,  5.]
    # sm.sat_profile:  [1.0, 1.0, 1.0, 1.0, 1.0]
    # sm.recharge_at_depth: [2.0, 2.0, 2.0, 2.0, 2.0]

    R = sm.calc_recharge_rate(np.array([0.0, 3.5, 4.5, 5.0, 10.0]), 5.0)
    assert_equal(R, [0.4, 0.4, 0.4, 0.4, 0.4])
