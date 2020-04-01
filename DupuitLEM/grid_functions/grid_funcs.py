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
    :math:`n` is Manning's n, :math:`Q` is surface water discharge, and :math:`dx` is the grid cell
    width.

    Parameters
    ----------
    n_manning: float or array of float (-)
        Manning's n at nodes, giving surface roughness.
    rho: density of water (kg/m3)
    g: gravitational acceleration constant (m/s2)
    Q: surface water discharge (m3/s)

    notes:
    In future, should allow n_manning and dx to be fields that are spatially variable

    """
    S = abs(grid.calc_grad_at_link(grid.at_node["topographic__elevation"]))
    S_node = map_max_of_node_links_to_node(grid, S)[grid.core_nodes]
    Q = grid.at_node["surface_water__discharge"][grid.core_nodes]

    tau = np.zeros(grid.number_of_nodes)
    tau[grid.core_nodes] = (rho * g * S_node * ( (n_manning * Q)/(grid.dx * np.sqrt(S_node)) )**(3/5)
    return tau


def calc_erosion_from_shear_stress(grid,tauc,K,b):
    r"""
    Calculate erosion rate dzdt (m) using the shear stress method:

    .. math::
        \frac{dz}{dt} = k (\tau - \tau_c)^b

    parameters
    ----------
    tau: shear stress array at node (N/m2)
    tauc: critical shear stress value (N/m2)
    k: shear stress erosion coefficient
    b: shear stress erosion exponent

    notes:
    Positive when eroding. Does not allow accumulation.
    In future allow for K, tauc and b to be spatially variable

    """
    tau = grid.at_node['surface_water__shear_stress']

    dzdt = np.zeros_like(tau)
    dzdt[grid.core_nodes] = K*np.power((tau[grid.core_nodes]-tauc),b)
    dzdt[dzdt<0.0] = 0.0

    return dzdt
