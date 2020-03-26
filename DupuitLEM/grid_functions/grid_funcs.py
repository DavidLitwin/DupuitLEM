import numpy as np
from landlab.grid.mappers import map_mean_of_link_nodes_to_link, map_max_of_node_links_to_node

def bind_avg_hydraulic_conductivity(ks,k0,dk):
    def bound_avg_hydraulic_conductivity(grid):
        """
        Calculate the average hydraulic conductivity when hydraulic conductivity
        varies with depth as:

            k = k0 + (ks-k0)*exp(-d/dk)

        Parameters:
            h = aquifer thickness
            b = depth from surface to impermeable base
            k0: asymptotic permeability at infinite depth
            ks: hydraulic conductivity at the ground surface
            dk = characteristic depth

        """
        h = grid.at_node['aquifer__thickness'],
        b = grid.at_node['topographic__elevation']-grid.at_node['aquifer_base__elevation']
        blink = map_mean_of_link_nodes_to_link(grid,b)
        hlink = map_mean_of_link_nodes_to_link(grid,h)
        b1 = blink[hlink>0.0]
        h1 = hlink[hlink>0.0]
        kavg = np.zeros_like(hlink)
        kavg[hlink>0.0] = dk*(ks-k0)/h1 * (np.exp(-(b1-h1)/dk) - np.exp(-b1/dk)) + k0
        return kavg
    return bound_avg_hydraulic_conductivity



def calc_shear_stress_at_node(grid, n_manning=0.05, rho = 1000, g = 9.81):
    r"""
    Calculate the shear stress :math:`\tau` based upon the equations: (N/m2)

    .. math::
        \tau = \rho g S d

    .. math::
        d = \bigg( \frac{n Q}{S^{1/2} dx} \bigg)^{3/2}

    where :math:`\rho` is the density of water, :math:`g` is the gravitational constant,
    :math:`S` is the topographic slope, :math:`d` is the water depth calculated with Manning's equation,
    :math:`n` is Manning's n, :math:`q` is surface water discharge, and :math:`dx` is the grid cell
    width.

    Parameters
    ----------
    n_manning: float or array of float (-)
        Manning's n at nodes, giving surface roughness.
    rho: density of water (kg/m3)
    g: gravitational acceleration constant (m/s2)
    """
    S = abs(grid.calc_grad_at_link(grid.at_node["topographic__elevation"]))
    S_node = map_max_of_node_links_to_node(grid, S)
    return (rho * g * S_node * (
            (n_manning * grid.at_node["surface_water__discharge"] / 3600)
            / (grid.dx * np.sqrt(S_node)) )**(3/5)
    )
