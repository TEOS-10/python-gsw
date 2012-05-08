# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np

from constants import db2Pascal
from earth import grav
from density_enthalpy_48 import rho

__all__ = [
           'Nsquared',
           #'Turner_Rsubrho',  TODO
           #'IPV_vs_fNsquared_ratio'  TODO
           ]


def Nsquared(SA, CT, p, lat=None):
    r"""Calculates the buoyancy frequency squared (N^2)(i.e. the Brunt-Väisälä
    frequency squared) at the mid pressure from the equation,
    .. math::
        N^2 = g^2 \frac{\partial\rho}{\partial p}

    Parameters
    ----------
    SA : array_like
         Absolute Salinity  [g/kg]
    CT : array_like
         Conservative Temperature [deg C]
    p : array_like
        sea pressure  [dbar]
    lat : array_like, optional
          latitude in decimal degrees north [-90..+90]
          If lat is not supplied, a default gravitational acceleration of
          9.7963 m/s^2 (Griffies, 2004) will be used.

    Returns
    -------
    N2 : array_like
         Brunt-Väisälä frequency squared [1 s :math:`-2`]
    p_mid : array_like
            Mid pressure between p grid

    See Also
    --------
    TODO

    Notes
    -----
    This routine uses rho from the computationally efficient 48-term expression
    for density in terms of SA, CT and p.  Also that the pressure increment,
    :math:`\partial p`, in the above formula is in Pa, so that it is 10^4 times
    the pressure increment in dbar.

    The 48-term equation has been fitted in a restricted range of parameter
    space, and is most accurate inside the "oceanographic funnel" described in
    McDougall et al. (2011).  The GSW library function "infunnel(SA, CT, p)" is
    available to be used if one wants to test if some of one's data lies
    outside this "funnel".

    Examples
    --------
    TODO

    References
    ----------
    .. [1] IOC, SCOR and IAPSO, 2010: The international thermodynamic equation
    of seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp. See section 3.10 and Eqn. (3.10.2).

    ..[2] McDougall T.J., P.M. Barker, R. Feistel and D.R. Jackett, 2011:  A
    computationally efficient 48-term expression for the density of
    seawater in terms of Conservative Temperature, and related properties
    of seawater.

    ..[3] Griffies, S. M., 2004: Fundamentals of Ocean Climate Models.
    Princeton, NJ: Princeton University Press, 518 pp + xxxiv.

    Modifications:
    2011-04-22. Trevor McDougall & Paul Barker
    """

    if lat != None:
        g = grav(lat, p)
    else:
        g = 9.7963  # Standard value from Griffies (2004).

    SA, CT, p, g = np.broadcast_arrays(SA, CT, p, g)

    p_mid = 0.5 * (p[1:, ...] + p[:-1, ...])

    drho = (rho(SA[1:, ...], CT[1:, ...], p_mid) -
            rho(SA[:-1, ...], CT[:-1, ...], p_mid))

    grav_local = 0.5 * (g[1:, ...] + g[:-1, ...])
    dp = p[1:, ...] - p[:-1, ...]

    N2 = grav_local ** 2 * drho / (db2Pascal * dp)

    return N2, p_mid
