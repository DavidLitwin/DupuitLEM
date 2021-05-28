"""
Tests of SchenkVadoseModel, implemented for DupuitLEM.

Date: 28 May 2021
"""

import numpy as np
from numpy.testing import assert_almost_equal, assert_equal

from DupuitLEM.auxiliary_models import SchenkVadoseModel


def test_init_depth_profile():
    """"Check depths and bin capacity. Depths correspond to the bottom 
    of each bin. Bin capacity is porosity * bin depth * difference in relative sat."""

    sm = SchenkVadoseModel(num_bins=5)
    
    assert_equal(sm.depths, [1., 2., 3., 4., 5.])
    assert_almost_equal(sm.bin_capacity, 0.04)


# check generated storm (make sure seed is set)
def test_generate_storm():
    """Make sure seed is set when generating a storm event."""
    
    sm1 = SchenkVadoseModel(random_seed=5)
    sm1.generate_storm()
    dr, tr, tb = sm1.Dr, sm1.Tr, sm1.Tb
    sm2 = SchenkVadoseModel(random_seed=5)
    sm2.generate_storm()
    
    assert_equal([dr, tr, tb], [sm2.Dr, sm2.Tr, sm2.Tb])


def test_run_event():
    """"trivial case of an event. An event with depth 2 fills the top 2 bins 
    when bins have unit depth and can be filled 100% with water. Recharge is 
    calculated on a 'floor' basis."""
    
    sm = SchenkVadoseModel(num_bins=5, 
                           upper_relative_saturation=1.0, 
                           lower_relative_saturation=0.0, 
                           porosity=1.0,
                           )
    sm.run_event(2.0)
    
    assert_equal(sm.sat_profile, [1., 1., 0., 0., 0.])
    assert_equal(sm.recharge_at_depth, [1., 0., 0., 0., 0.])

# trivial case check bins emptied after one interevent
def test_run_interevent():
    """ Trivial case of an interevent. With unit PET rate, the top two
    bins are drained after 2 units of time when bins have unit depth and 
    can be filled 100% with water. Note recharge_at_depth is not reset."""
    
    sm = SchenkVadoseModel(num_bins=5, 
                           upper_relative_saturation=1.0, 
                           lower_relative_saturation=0.0, 
                           porosity=1.0,
                           potential_evapotranspiration_rate=1.0,
                           )
    sm.sat_profile[:] = np.array([1., 1., 1., 0., 0.])
    sm.run_interevent(2.0)
    
    assert_equal(sm.sat_profile, [0., 0., 1., 0., 0.])


def test_run_event_2():
    """Slightly less trivial example of an event where there is already
    some inital nonuniform saturation. Ensure that the topmost available bins
    are filled."""
    
    sm = SchenkVadoseModel(num_bins=5, 
                           upper_relative_saturation=1.0, 
                           lower_relative_saturation=0.0, 
                           porosity=1.0,
                           )
    sm.sat_profile[:] = np.array([0., 1., 0., 0., 1.])
    sm.run_event(2.0)
    
    assert_equal(sm.sat_profile, [1., 1., 1., 0., 1.])
    assert_equal(sm.recharge_at_depth, [1., 1., 0., 0., 0.])
    
    
def test_run_event_3():
    """Edge case where event depth is added to a profile that is already
    saturated. In this case, the profile should stay saturated, but the 
    recharge should reflect the total amount. That is, a water table 
    at any depth will recieve recharge."""
    
    sm = SchenkVadoseModel(num_bins=5, 
                           upper_relative_saturation=1.0, 
                           lower_relative_saturation=0.0, 
                           porosity=1.0,
                           )
    sm.sat_profile[:] = np.array([1., 1., 1., 1., 1.])
    sm.run_event(2.0)
    
    assert_equal(sm.sat_profile, [1., 1., 1., 1., 1.])
    assert_equal(sm.recharge_at_depth, [2., 2., 2., 2., 2.])
    

def test_run_interevent_2():
    """Edge case where interevent PET exceeds available saturation. In this 
    case, the remaining saturation is drained."""
    
    sm = SchenkVadoseModel(num_bins=5, 
                           upper_relative_saturation=1.0, 
                           lower_relative_saturation=0.0, 
                           porosity=1.0,
                           potential_evapotranspiration_rate=1.0,
                           )
    sm.sat_profile[:] = np.array([0., 0., 0., 0., 1.])
    sm.run_interevent(2.0)
    
    assert_equal(sm.sat_profile, [0., 0., 0., 0., 0.])