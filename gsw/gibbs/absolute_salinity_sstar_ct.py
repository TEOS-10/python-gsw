# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np

from constants import SSO, r1
from gsw.utilities import match_args_return
from conversions import pt0_from_t, CT_from_pt
import library as lib

__all__ = [
           'SA_from_SP',  # FIXME: Incomplete and untested. (need lib.SAAR)
           'Sstar_from_SP',  # FIXME: Incomplete and untested. (need lib.SAAR)
           'CT_from_t'
           ]


def check_input(SP, p, lon, lat):
    r"""Check for out of range values."""
    lon, lat, p, SP = np.broadcast_arrays(lon, lat, p, SP)

    SP[(p < 100) & (SP > 120)] = np.NaN
    SP[(p >= 100) & (SP > 42)] = np.NaN

    lon = lon % 360

    # FIXME: Test these exceptions, they are probably broken!
    # The original also checks for 9999s, not sure why.
    if ((p < -1.5) | (p > 12000)).any():
        raise(Exception, 'Sstar_from_SP: pressure is out of range')
    if ((lon < 0) | (lon > 360)).any():
        raise(Exception, 'Sstar_from_SP: longitude is out of range')
    if (np.abs(lat) > 90).any():
        raise(Exception, 'Sstar_from_SP: latitude is out of range')

    SP = np.maximum(SP, 0)

    return SP, p, lon, lat


@match_args_return
def SA_from_SP(SP, p, lon, lat):
    r"""Calculates Absolute Salinity from Practical Salinity.

    Parameters
    ----------
    SP : array_like
         salinity (PSS-78) [unitless]
    p : array_like
        pressure [dbar]
    lon : array_like
          decimal degrees east [0..+360] or [-180..+180]
    lat : array_like
          decimal degrees (+ve N, -ve S) [-90..+90]

    Returns
    -------
    SA : masked array
         Absolute salinity [g kg :sup:`-1`]

    See Also
    --------
    FIXME
    _delta_SA, _SA_from_SP_Baltic

    Notes
    -----
    The mask is only set when the observation is well and truly on dry
    land; often the warning flag is not set until one is several hundred
    kilometers inland from the coast.

    Since SP is non-negative by definition, this function changes any negative
    input values of SP to be zero.

    Examples
    --------
    >>> import seawater.gibbs as gsw
    >>> SP = [34.5487, 34.7275, 34.8605, 34.6810, 34.5680, 34.5600]
    >>> p = [10, 50, 125, 250, 600, 1000]
    >>> lon, lat = 188, 4
    >>> gsw.SA_from_SP(SP, p, lon, lat)
    array([ 34.71177971,  34.89152372,  35.02554774,  34.84723008,
            34.7366296 ,  34.73236186])

    References
    ----------
    .. [1] IOC, SCOR and IAPSO, 2010: The international thermodynamic equation
    of seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp. See section 2.5 and appendices A.4 and A.5.

    .. [2] McDougall, T.J., D.R. Jackett and F.J. Millero, 2010: An algorithm
    for estimating Absolute Salinity in the global ocean. Submitted to Ocean
    Science. A preliminary version is available at Ocean Sci. Discuss.,
    6, 215-242.
    http://www.ocean-sci-discuss.net/6/215/2009/osd-6-215-2009-print.pdf

    Modifications:
    2011-05-31. David Jackett, Trevor McDougall & Paul Barker.
    """

    SP, p, lon, lat = check_input(SP, p, lon, lat)

    SAAR = lib.SAAR(p, lon, lat)
    #SAAR = lib.delta_SA(p, lon, lat)

    SA = (SSO / 35) * SP * (1 + SAAR)
    SA_baltic = lib.SA_from_SP_Baltic(SP, lon, lat)

    # The following function (SAAR) finds SAAR in the non-Baltic parts of
    # the world ocean.  (Actually, this SAAR look-up table returns values
    # of zero in the Baltic Sea since SAAR in the Baltic is a function of SP,
    # not space.
    if SA_baltic is not None:
        SA[~SA_baltic.mask] = SA_baltic[~SA_baltic.mask]

    return SA


@match_args_return
def Sstar_from_SP(SP, p, lon, lat):
    r"""Calculates Preformed Salinity from Absolute Salinity.

    Parameters
    ----------
    SP : array_like
         salinity (PSS-78) [unitless]
    p : array_like
        pressure [dbar]
    lon : array_like
          decimal degrees east [0..+360] or [-180..+180]
    lat : array_like
          decimal degrees (+ve N, -ve S) [-90..+90]

    Returns
    -------
    Sstar : masked array
            Preformed Salinity [g kg :sup:`-1`]

    See Also
    --------
    FIXME
    _delta_SA, _SA_from_SP_Baltic

    Notes
    -----
    The mask is only set when the observation is well and truly on dry
    land; often the warning flag is not set until one is several hundred
    kilometers inland from the coast.

    Since SP is non-negative by definition, this function changes any negative
    input values of SP to be zero.

    Examples
    --------
    >>> import seawater.gibbs as gsw
    >>> SP = [34.5487, 34.7275, 34.8605, 34.6810, 34.5680, 34.5600]
    >>> p = [10, 50, 125, 250, 600, 1000]
    >>> lon, lat =  188, 4
    >>> gsw.Sstar_from_SP(SP, p, lon, lat)
    array([ 34.7115532 ,  34.89116101,  35.02464926,  34.84359277,
            34.7290336 ,  34.71967638])

    References
    ----------
    .. [1] IOC, SCOR and IAPSO, 2010: The international thermodynamic equation
    of seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp. See section 2.5 and appendices A.4 and A.5.

    .. [2] McDougall, T.J., D.R. Jackett and F.J. Millero, 2010: An algorithm
    for estimating Absolute Salinity in the global ocean. Submitted to Ocean
    Science. A preliminary version is available at Ocean Sci. Discuss.,
    6, 215-242.

    Modifications:
    2011-03-27. David Jackett, Trevor McDougall and Paul Barker.
    """

    SP, p, lon, lat = check_input(SP, p, lon, lat)

    SAAR = lib.SAAR(p, lon, lat)
    #SAAR = lib.delta_SA(p, lon, lat)
    Sstar = (SSO / 35.) * SP - r1 * SAAR

    # In the Baltic Sea, Sstar==SA.
    Sstar_baltic = lib.SA_from_SP_Baltic(SP, lon, lat)

    # TODO: Create Baltic and non-Baltic test cases.
    if Sstar_baltic is not None:
        Sstar[~Sstar_baltic.mask] = Sstar_baltic[~Sstar_baltic.mask]

    return Sstar


@match_args_return
def CT_from_t(SA, t, p):
    r"""Calculates Conservative Temperature of seawater from in situ
    temperature.

    Parameters
    ----------
    SA : array_like
         Absolute salinity [g kg :sup:`-1`]
    CT : array_like
         Conservative Temperature [:math:`^\circ` C (ITS-90)]
    p : array_like
        pressure [dbar]

    Returns
    -------

    See Also
    --------
    TODO

    Notes
    -----
    TODO

    Examples
    --------
    TODO

    References
    ----------
    .. [1] IOC, SCOR and IAPSO, 2010: The international thermodynamic equation
    of seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp. See section 3.3.

    Modifications:
    2011-03-27. David Jackett, Trevor McDougall and Paul Barker
    """
    # Find values that are out of range, set them to NaN.
    invalid = np.logical_and(p < 100, np.logical_or(t > 80, t < -12))
    t[invalid] = np.NaN

    invalid = np.logical_and(p >= 100, np.logical_or(t > 40, t < -12))
    t[invalid] = np.NaN

    pt0 = pt0_from_t(SA, t, p)
    CT = CT_from_pt(SA, pt0)

    return CT


@match_args_return
def SA_CT_plot(SA, CT, isopycs, title_string):
    r"""Calculates Conservative Temperature of seawater from in situ
    temperature.

    Parameters
    ----------
    SA : array_like
         Absolute salinity [g kg :sup:`-1`]
    CT : array_like
         Conservative Temperature [:math:`^\circ` C (ITS-90)]

    See Also
    --------
    TODO

    Notes
    -----
    TODO

    Examples
    --------
    TODO

    References
    ----------
    .. [1] IOC, SCOR and IAPSO, 2010: The international thermodynamic equation
    of seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp. See section 3.3.

    Modifications:
    2011-03-27. David Jackett, Trevor McDougall and Paul Barker
    """

    return None

if __name__ == '__main__':
    import doctest
    doctest.testmod()
