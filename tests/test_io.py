"""
Test core functionality of IO functions for DupuitLEM.
This includes functions for reading and writing NetCDF files
that store model grid and field information.
"""


from fileinput import filename
import numpy as np
import xarray as xr

from landlab import RasterModelGrid, HexModelGrid
from DupuitLEM.io import (load_fields_from_dataset, 
                          load_grid_from_dataset, 
                          _find_last_non_nan, 
                          write_output_step,
                          initialize_output_dataset,
)
from numpy.testing import assert_array_equal, assert_equal


def create_example_dataset():
    """
    Create an example xarray Dataset with dimensions (time=5, node=25).
    The last 2 time steps are filled with NaN.
    
    Returns
    -------
    ds : xarray.Dataset
        Example dataset
    """
    times = np.arange(5)
    nodes = np.arange(25)
    
    # Create data: first 3 timesteps with values, last 2 all NaN
    np.random.seed(0)
    data = np.random.rand(5, 25)
    data[3:, :] = np.nan
    
    ds = xr.Dataset(
        {
            "topographic__elevation": (("time", "node"), data),
        },
        coords={
            "time": times,
            "node": nodes,
        }
    )
    
    return ds


def test_find_last_non_nan():
    """
    Test finding the last non-NaN time index in a dataset.
    """

    ds = create_example_dataset()
    
    last_index = _find_last_non_nan(ds, t_index=-1) # Check from the end (negative index)
    assert_equal(last_index, -3)

    last_index = _find_last_non_nan(ds, t_index=4) # Check from end (positive index)
    assert_equal(last_index, 2)


def test_load_fields_from_dataset():
    """
    Test loading fields from a dataset into a grid.
    """

    ds = create_example_dataset()
    
    # Create a RasterModelGrid with 25 nodes
    mg = RasterModelGrid((5, 5), xy_spacing=1.0)
    
    # Load fields at last non-NaN time index
    load_fields_from_dataset(ds, mg, last_non_nan=True)
    
    # Check that the field was loaded correctly
    expected_data = ds['topographic__elevation'].isel(time=2).values
    loaded_data = mg.at_node['topographic__elevation']
    
    assert_array_equal(loaded_data, expected_data)


def test_load_fields_from_dataset_specific_index():
    """
    Test loading fields from a dataset into a grid at a specific time index.
    """

    ds = create_example_dataset()
    
    # Create a RasterModelGrid with 25 nodes
    mg = RasterModelGrid((5, 5), xy_spacing=1.0)
    
    # Load fields at time index 1
    load_fields_from_dataset(ds, mg, t_index=1)
    
    # Check that the field was loaded correctly
    expected_data = ds['topographic__elevation'].isel(time=1).values
    loaded_data = mg.at_node['topographic__elevation']
    
    assert_array_equal(loaded_data, expected_data)

def test_create_output_dataset():

    """
    Test that an output dataset can be created and written to correctly.
    """

    mg = RasterModelGrid((5, 5), xy_spacing=1.0)
    mg.add_field('topographic__elevation', np.random.rand(25), at='node')
    mg.add_field('water__depth', np.random.rand(25), at='node')
    output_fields = ['at_node:topographic__elevation', 'at_node:water__depth']
    output = {'output_fields': output_fields}

    ds_out = initialize_output_dataset(mg, output, [10])
    write_output_step(ds_out, mg, output, 0)

    np.testing.assert_array_equal(ds_out['topographic__elevation'].isel(time=0).values, mg.at_node['topographic__elevation'])
    np.testing.assert_array_equal(ds_out['water__depth'].isel(time=0).values, mg.at_node['water__depth'])
    np.testing.assert_array_equal(ds_out['time'].values, [10])

    # temp_file = 'temp_output.nc'
    # ds_out.to_netcdf(temp_file, format="NETCDF4")