"""
Misc. functions that are useful or necessary for model operations.

Author: David Litwin

Date: 16 March 2020
"""


import numpy as np
from landlab.grid.mappers import map_mean_of_link_nodes_to_link


def bind_avg_exp_ksat(ks, k0, dk):
    """
    Function to create the bound_avg_exp_ksat function. This function
    is used when the hydraulic conductivity needs to vary with
    depth:
        k = k0 + (ks-k0)*exp(-(b-h)/dk)

    Parameters
    ----------   
    ks: float
        Hydraulic conductivity at the ground surface.
    k0: float
        Asymptotic permeability at infinite depth.
    dk: float
        Characteristic depth.

    Returns
    -------
    bound_avg_exp_ksat: function
        Function to supply to GroundwaterDupuitPercolator
    """

    def bound_avg_exp_ksat(grid):
        """
        Calculate the average hydraulic conductivity when hydraulic conductivity
        varies with depth as:

            k = k0 + (ks-k0)*exp(-(b-h)/dk)

        Parameters
        ----------
        grid: ModelGrid
            Landlab ModelGrid object. 

        Returns
        -------
        kavg: array
            Array of effective hydrualic conductivities given water table position
            and a depth-dependent hydraulic conductivity model.
        """

        h = grid.at_node["aquifer__thickness"]
        b = (
            grid.at_node["topographic__elevation"]
            - grid.at_node["aquifer_base__elevation"]
        )
        blink = map_mean_of_link_nodes_to_link(grid, b)
        hlink = map_mean_of_link_nodes_to_link(grid, h)
        b1 = blink[hlink > 0.0]
        h1 = hlink[hlink > 0.0]
        kavg = np.zeros_like(hlink)
        kavg[hlink > 0.0] = (
            dk * (ks - k0) / h1 * (np.exp(-(b1 - h1) / dk) - np.exp(-b1 / dk)) + k0
        )
        return kavg

    return bound_avg_exp_ksat


def bind_avg_recip_ksat(ks, d):
    """
    Function to creat the bound_avg_recip_ksat function. This function
    is used when the hydraulic conductivity needs to vary with
    depth:
        k = ks / ((b-h)/d+1)

    Parameters
    ----------   
    ks: float
        Hydraulic conductivity at the ground surface (b=h).
    d: float
        Decay length scale. Usually set d = x * b, where 0 < x < 1.

    Returns
    -------
    bound_avg_recip_ksat: function
        Function to supply to GroundwaterDupuitPercolator
    """

    def bound_avg_recip_ksat(grid):
        """
        Calculate the average hydraulic conductivity when hydraulic conductivity
        varies with depth as:

        k = ks / ((b-h)/d+1)

        Parameters
        ----------
        grid: ModelGrid
            Landlab ModelGrid object. 

        Returns
        -------
        kavg: array
            Array of effective hydrualic conductivities given water table position
            and a depth-dependent hydraulic conductivity model.
        """

        h = grid.at_node["aquifer__thickness"]
        b = (
            grid.at_node["topographic__elevation"]
            - grid.at_node["aquifer_base__elevation"]
        )

        hlink = map_mean_of_link_nodes_to_link(grid, h)
        h1 = hlink[hlink > 0.0]
        kavg = np.zeros_like(hlink)

        if np.allclose(b, b[0]):
            b1 = b[0]
        else:
            blink = map_mean_of_link_nodes_to_link(grid, b) 
            b1 = blink[hlink > 0.0]

        kavg[hlink > 0.0] = (
            (ks*d)/h1 * np.log((b1 + d)/(b1 + d - h1))
        )
        return kavg

    return bound_avg_recip_ksat

def bind_avg_dual_ksat(Ks_0, Ks_1, b_1):
    """
    Function to create the bound_avg_dual_ksat function. This function
    is used when the hydraulic conductivity has a two-layer structure,
    in which the lower layer is bounded at the bottom by the 
    aquifer_base__elevation, and the upper layer bounded by a surface
    a fixed distance b1 below topography z:

             ( Ks_0,    zwt <= z - b1
        k =  |
             ( Ks_1,    zwt > z - b1

    Parameters
    ----------   
    Ks_0: float
        Hydraulic conductivity of lower layer.
    ks_1: float
        Hydraulic conductivity of upper layer.
    b_1: float
        Depth from surface to upper layer boundary.

    Returns
    -------
    bound_avg_dual_ksat: function
        Function to supply to GroundwaterDupuitPercolator
    """

    def bound_avg_dual_ksat(grid):
        """
        Calculate the average hydraulic conductivity for a two-layer 
        subsurface structure:

             ( Ks_0,    zwt <= z - b1
        k =  |
             ( Ks_1,    zwt > z - b1

        Parameters
        ----------
        grid: ModelGrid
            Landlab ModelGrid object. 

        Returns
        -------
        kavg: array
            Array of effective hydrualic conductivities given water table position
            and a depth-dependent hydraulic conductivity model.
        """

        # grid fields
        z = grid.at_node["topographic__elevation"]
        z0 = grid.at_node["aquifer_base__elevation"]
        z1 = z - b_1
        zwt = grid.at_node["water_table__elevation"]

        # calc avg if wt in upper layer, then conditional set if in lower layer
        Ks_node = ((z1-z0)*Ks_0 + (zwt - z1)*Ks_1)/(zwt-z0)
        Ks_node[zwt<=z1] = Ks_0

        # map to links
        Kavg = map_mean_of_link_nodes_to_link(grid, Ks_node)

        return Kavg

    return bound_avg_dual_ksat
