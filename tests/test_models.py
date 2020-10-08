import numpy as np

from numpy.testing import assert_equal

from landlab import HexModelGrid, RasterModelGrid
from landlab.components import (
    GroundwaterDupuitPercolator,
    LinearDiffuser,
    FastscapeEroder,
)
from landlab.io.netcdf import from_netcdf
from DupuitLEM import StreamPowerModel
from DupuitLEM.auxiliary_models import (
    HydrologySteadyStreamPower,
    RegolithConstantThickness,
)


def test_stream_power_run_step():

    mg = RasterModelGrid((3, 3), xy_spacing=10.0)
    mg.set_status_at_node_on_edges(
        right=mg.BC_NODE_IS_CLOSED,
        top=mg.BC_NODE_IS_CLOSED,
        left=mg.BC_NODE_IS_CLOSED,
        bottom=mg.BC_NODE_IS_FIXED_VALUE,
    )
    z = mg.add_ones("node", "topographic__elevation")
    z[1] = 0.0
    zb = mg.add_zeros("node", "aquifer_base__elevation")
    mg.add_ones("node", "water_table__elevation")

    gdp = GroundwaterDupuitPercolator(mg, recharge_rate=1e-4)
    hm = HydrologySteadyStreamPower(mg, groundwater_model=gdp)
    sp = FastscapeEroder(
        mg,
        K_sp=1e-10,
        m_sp=1,
        n_sp=1,
        discharge_field="surface_water_area_norm__discharge",
    )
    ld = LinearDiffuser(mg, linear_diffusivity=1e-10)
    rm = RegolithConstantThickness(mg, uplift_rate=0.0)

    mdl = StreamPowerModel(
        mg, hydrology_model=hm, diffusion_model=ld, erosion_model=sp, regolith_model=rm,
    )

    mdl.run_step(1e5)

    assert z[4] < 1.0
    assert_equal(z[4] - zb[4], 1.0)


def test_stream_power_save_output(tmpdir):

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

    gdp = GroundwaterDupuitPercolator(mg, recharge_rate=1e-4)
    hm = HydrologySteadyStreamPower(mg, groundwater_model=gdp)
    sp = FastscapeEroder(
        mg,
        K_sp=1e-10,
        m_sp=1,
        n_sp=1,
        discharge_field="surface_water_area_norm__discharge",
    )
    ld = LinearDiffuser(mg, linear_diffusivity=1e-10)
    rm = RegolithConstantThickness(mg, uplift_rate=0.0)

    output = {}
    output["output_interval"] = 1000
    output["output_fields"] = [
        "at_node:topographic__elevation",
        "at_node:aquifer_base__elevation",
        "at_node:water_table__elevation",
    ]
    output["base_output_path"] = tmpdir.strpath + "/"
    output["run_id"] = 0  # make this task_id if multiple runs

    mdl = StreamPowerModel(
        mg,
        hydrology_model=hm,
        diffusion_model=ld,
        erosion_model=sp,
        regolith_model=rm,
        total_morphological_time=1e8,
        output_dict=output,
    )

    mdl.run_model()

    file = tmpdir.join("0_grid_0.nc")
    mg1 = from_netcdf(file.strpath)
    keys = [
        "topographic__elevation",
        "aquifer_base__elevation",
        "water_table__elevation",
    ]
    assert isinstance(mg1, RasterModelGrid)
    assert set(mg1.at_node.keys()) == set(keys)
    assert_equal(mg1.status_at_node, mg.status_at_node)


def test_stream_power_save_output_hex(tmpdir):

    mg = HexModelGrid((3, 3), node_layout="rect", spacing=10.0)
    mg.status_at_node[mg.status_at_node == 1] = 4
    mg.status_at_node[1] = 1
    mg.add_ones("node", "topographic__elevation")
    mg.add_zeros("node", "aquifer_base__elevation")
    mg.add_ones("node", "water_table__elevation")

    gdp = GroundwaterDupuitPercolator(mg, recharge_rate=1e-4)
    hm = HydrologySteadyStreamPower(
        mg, groundwater_model=gdp, routing_method="Steepest"
    )
    sp = FastscapeEroder(
        mg,
        K_sp=1e-10,
        m_sp=1,
        n_sp=1,
        discharge_field="surface_water_area_norm__discharge",
    )
    ld = LinearDiffuser(mg, linear_diffusivity=1e-10)
    rm = RegolithConstantThickness(mg, uplift_rate=0.0)

    output = {}
    output["output_interval"] = 1000
    output["output_fields"] = [
        "at_node:topographic__elevation",
        "at_node:aquifer_base__elevation",
        "at_node:water_table__elevation",
    ]
    output["base_output_path"] = tmpdir.strpath + "/"
    output["run_id"] = 0  # make this task_id if multiple runs

    mdl = StreamPowerModel(
        mg,
        hydrology_model=hm,
        diffusion_model=ld,
        erosion_model=sp,
        regolith_model=rm,
        total_morphological_time=1e8,
        output_dict=output,
    )

    mdl.run_model()

    file = tmpdir.join("0_grid_0.nc")
    mg1 = from_netcdf(file.strpath)
    keys = [
        "topographic__elevation",
        "aquifer_base__elevation",
        "water_table__elevation",
    ]
    assert isinstance(mg1, HexModelGrid)
    assert set(mg1.at_node.keys()) == set(keys)
    assert_equal(mg1.status_at_node, mg.status_at_node)
