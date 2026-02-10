"""
I/O utilities for DupuitLEM.

This module implements a grid-agnostic xarray/netCDF output format
that supports both RasterModelGrid and HexModelGrid and allows clean
model restarts.
"""

import numpy as np
import xarray as xr

from landlab import RasterModelGrid, HexModelGrid


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

def parse_output_field(field_str):
    """
    Parse an output field string like 'at_node:topographic__elevation'.

    Returns
    -------
    location : str
        Field location (e.g., 'node')
    name : str
        Field name
    """
    loc, name = field_str.split(":")
    return loc.replace("at_", ""), name


# ------------------------------------------------------------
# Dataset initialization
# ------------------------------------------------------------

def initialize_output_dataset(mg, output, times):
    """
    Initialize an xarray Dataset for DupuitLEM output.

    Parameters
    ----------
    mg : ModelGrid
        Landlab grid (RasterModelGrid or HexModelGrid)
    output : dict
        Output specification dictionary
    times : array-like
        Times at which output will be written

    Returns
    -------
    ds : xarray.Dataset
        Initialized dataset with empty data variables
    """

    # ---- coordinates ----
    ds = xr.Dataset(
        coords={
            "time": np.asarray(times),
            "node": np.arange(mg.number_of_nodes),
        }
    )

    # spatial coordinates (useful for plotting & postprocessing)
    ds["x"] = ("node", mg.x_of_node)
    ds["y"] = ("node", mg.y_of_node)
    ds["status_at_node"] = ("node", mg.status_at_node)

    # ---- data variables ----
    for field in output["output_fields"]:
        loc, name = parse_output_field(field)

        if loc != "node":
            raise NotImplementedError(
                f"Output location '{loc}' not yet supported"
            )

        ds[name] = (
            ("time", "node"),
            np.full((len(times), mg.number_of_nodes), np.nan),
        )

    # ---- grid metadata ----
    ds.attrs.update(_grid_metadata_from_grid(mg))

    # ---- run metadata ----
    ds.attrs.update({
        "run_id": output.get("run_id", None),
        "output_interval": output.get("output_interval", None),
    })

    return ds


# ------------------------------------------------------------
# Writing output
# ------------------------------------------------------------

def write_output_step(ds, mg, output, t_index):
    """
    Write one output timestep into an existing Dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Open output dataset
    mg : ModelGrid
        Landlab grid
    output : dict
        Output specification dictionary
    t_index : int
        Index along the time dimension
    """

    for field in output["output_fields"]:
        loc, name = parse_output_field(field)

        if loc != "node":
            raise NotImplementedError(
                f"Output location '{loc}' not yet supported"
            )

        ds[name][t_index, :] = mg.at_node[name]


# ------------------------------------------------------------
# Grid reconstruction (restart support)
# ------------------------------------------------------------

def load_grid_from_dataset(ds):
    """
    Reconstruct a Landlab grid from Dataset metadata.

    Parameters
    ----------
    ds : xarray.Dataset

    Returns
    -------
    mg : ModelGrid
        Reconstructed grid
    """

    grid_type = ds.attrs["grid_type"]

    if grid_type == "RasterModelGrid":
        mg = RasterModelGrid(
            shape=tuple(ds.attrs["shape"]),
            xy_spacing=ds.attrs["spacing"],
            xy_of_lower_left=ds.attrs.get("origin", (0.0, 0.0)),
        )
        mg.status_at_node[:] = ds["status_at_node"].values

    elif grid_type == "HexModelGrid":
        mg = HexModelGrid(
            shape=tuple(ds.attrs["shape"]),
            spacing=ds.attrs["spacing"],
            xy_of_lower_left=ds.attrs.get("origin", (0.0, 0.0)),
            orientation=ds.attrs.get("orientation", "horizontal"),
            node_layout=ds.attrs.get("node_layout", "hex"),
        )
        mg.status_at_node[:] = ds["status_at_node"].values

    else:
        raise ValueError(f"Unsupported grid type '{grid_type}'")

    return mg


def load_fields_from_dataset(ds, mg, t_index=-1, last_non_nan=False):
    """
    Load fields from a Dataset into an existing grid.

    Parameters
    ----------
    ds : xarray.Dataset
    mg : ModelGrid
    t_index : int, optional
        Time index to load (default: last timestep)
    last_non_nan : bool, optional
        If True, search backward from t_index to find the last non-NaN data (default: False)
    """

    if last_non_nan:
        t_index = _find_last_non_nan(ds, t_index)

    for var in ds.data_vars:
        if "node" in ds[var].dims and var not in ("x", "y", "status_at_node"):
            mg.add_field(
                var,
                ds[var].isel(time=t_index).values,
                clobber=True,
            )

# ------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------

def _grid_metadata_from_grid(mg):
    """
    Extract minimal grid metadata needed for reconstruction.
    """
    if isinstance(mg, RasterModelGrid):
        spacing = mg.dx
    elif isinstance(mg, HexModelGrid):
        spacing = mg.spacing
    else:
        raise ValueError(f"Unsupported grid type '{type(mg)}'")

    meta = {
        "grid_type": type(mg).__name__,
        "shape": mg.shape,
        "spacing": spacing,
        "origin": mg.xy_of_lower_left,
    }

    if isinstance(mg, HexModelGrid):
        meta["orientation"] = mg.orientation
        meta["node_layout"] = mg.node_layout

    return meta

def _find_last_non_nan(ds, t_index):
    """
    Find the last time index where data variables are not all NaN.

    Parameters
    ----------
    ds : xarray.Dataset
    t_index : int
        Starting time index (searches backward from here)

    Returns
    -------
    last_index : int
        Last time index with non-NaN data
    """
    for i in range(t_index, -len(ds.time) - 1, -1):
        if not np.isnan(ds.isel(time=i).to_array()).all():
            return i
    raise ValueError("No non-NaN data found in dataset")