# -*- coding: utf-8 -*-
"""
Analytical solutions to the SchenkVadoseModel identified by Ciaran Harman.
"""


def recharge_freq(profile_depth, tb, ds, pet, Sawc):
    """
    Returns the frequency of recharge relative to the frequency of events
    occurring at depths in the vadose zone profile.

    Parameters
    ----------
    profile_depth : array
        Depths below surface, stored in svm as self.depths.
    tb : float
        Mean interstorm duration.
    ds : float
        Mean storm depth.
    pet : float
        Potential evapotranspiration rate.
    Sawc : float
        Plant available water content, as a proportion of total volume.

    Returns
    -------
    freq : array
        Frequency of recharge relative to frequency of events 1/(tr+tb).
    """
    a = (profile_depth * Sawc) / ds
    b = (profile_depth * Sawc) / (pet * tb)

    freq = np.zeros_like(profile_depth)
    T1 = np.exp(a - b) / (1 + b) + (1 - a / b) + ((a - b) * np.exp(-b)) / (b * (1 + b))
    T2 = np.exp(-a + b) * (
        (1 - b / a)
        + (b / a) ** 2 * (1 / (1 + b) + ((a - b) * np.exp(-a)) / (b * (1 + b)))
    )

    freq[a < b] = T1[a < b]
    freq[a > b] = T2[a > b]

    return freq


def extraction_freq(profile_depth, ds, tb, pet, Sawc):
    """
    Returns the frequency of uptake of soil water during interstorms relative
    to the frequency of events occurring at depths in the vadose zone profile.

    Parameters
    ----------
    profile_depth : array
        Depths below surface, stored in svm as self.depths.
    ds : float
        Mean storm depth.
    tb : float
        Mean interstorm duration.
    pet : float
        Potential evapotranspiration rate.
    Sawc : float
        Plant available water content, as a proportion of total volume.

    Returns
    -------
    freq : array
        Frequency of extraction relative to frequency of events 1/(tr+tb).
    """
    a = (profile_depth * Sawc) / (pet * tb)
    b = (profile_depth * Sawc) / ds

    freq = np.zeros_like(profile_depth)
    T1 = np.exp(a - b) / (1 + b) + (1 - a / b) + ((a - b) * np.exp(-b)) / (b * (1 + b))
    T2 = np.exp(-a + b) * (
        (1 - b / a)
        + (b / a) ** 2 * (1 / (1 + b) + ((a - b) * np.exp(-a)) / (b * (1 + b)))
    )

    freq[a < b] = T1[a < b]
    freq[a > b] = T2[a > b]

    return freq


def extraction_pdf(profile_depth, ds, tb, pet, Sawc):
    """
    Returns the probability density function describing where root water uptake
    occurs in the depth profile. This is equivalen to the rooting distribution
    in Schenk (2008).

    Parameters
    ----------
    profile_depth : array
        Depths below surface, stored in svm as self.depths.
    ds : float
        Mean storm depth.
    tb : float
        Mean interstorm duration.
    pet : float
        Potential evapotranspiration rate.
    Sawc : float
        Plant available water content, as a proportion of total volume.

    Returns
    -------
    out : array
        PDF of root water uptake at depths in profile_depth.
    """

    z = profile_depth
    a = (profile_depth * Sawc) / ds
    b = (profile_depth * Sawc) / (pet * tb)

    out = np.zeros_like(profile_depth)
    T1 = (b*np.exp(a - b)) / ((b + 1)*z) * (np.exp(-a) / (b + 1) - (1- (b*(b+2))/(a*(b+1)))*(1-np.exp(-a)) )
    T2 = (a*np.exp(b - a)) / ((b + 1)*z) * (b+1-b/a* (2*b-2*np.exp(-a)+2 - b/a*((b**2/(b+1)*np.exp(-a) + (b+2-b**2/(a*(b+1)))*(1-np.exp(-a))))))
    T3 = 1/(a+1)**2

    out[a < b] = T1[a < b]
    out[a > b] = T2[a > b]
    out[a==b] = T3[a==b]

    return out

def extraction_cdf(profile_depth, ds, tb, pet, Sawc):
    """
    Returns the cumulative distribution function describing where root water uptake
    occurs in the depth profile.

    Parameters
    ----------
    profile_depth : array
        Depths below surface, stored in svm as self.depths.
    ds : float
        Mean storm depth.
    tb : float
        Mean interstorm duration.
    pet : float
        Potential evapotranspiration rate.
    Sawc : float
        Plant available water content, as a proportion of total volume.

    Returns
    -------
    out : array
        PDF of root water uptake at depths in profile_depth.
    """

    z = profile_depth * Sawc
    a = (profile_depth * Sawc) / ds
    b = (profile_depth * Sawc) / (pet * tb)
    beta = pet * tb

    out = np.zeros_like(profile_depth)
    T1 = (ds - np.exp(z/ds)*ds - beta + np.exp(z/beta)*(z + beta))/(np.exp(z/beta)*beta*(z + beta))
    T2 = (np.exp(z/beta)*ds*(ds - beta) + np.exp((2*z)/ds)*beta*(z + beta) + np.exp(z*(1/ds + 1/beta))*(-ds**2 + z*(ds - beta) + ds*beta - beta**2))/(np.exp((2*z)/ds)*beta**2*(z + beta))

    out[a < b] = T1[a < b]
    out[a >= b] = T2[a >= b]

    return out
