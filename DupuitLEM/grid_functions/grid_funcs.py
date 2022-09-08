"""
Misc. functions that are useful or necessary for model operations.

Author: David Litwin

Date: 16 March 2020
"""


import numpy as np
from landlab.grid.mappers import map_mean_of_link_nodes_to_link


def bind_avg_hydraulic_conductivity(ks, k0, dk):
    def bound_avg_hydraulic_conductivity(grid):
        """
        Calculate the average hydraulic conductivity when hydraulic conductivity
        varies with depth as:

            k = k0 + (ks-k0)*exp(-(b-h)/dk)

        Parameters:
            h = aquifer thickness
            b = depth from surface to impermeable base
            k0: asymptotic permeability at infinite depth
            ks: hydraulic conductivity at the ground surface
            dk = characteristic depth

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

    return bound_avg_hydraulic_conductivity
