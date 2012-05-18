# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np
from gsw.utilities import match_args_return
#from conversions import geo_strf_dyn_height

__all__ = ['steric_height']


@match_args_return
def steric_height(SA, CT, p, p_ref):
    r"""Calculates steric height anomaly as the pressure integral of specific
    volume anomaly from the pressure p of the "bottle" to the reference
    pressure p_ref, divided by the constant value of the gravitational
    acceleration, 9.7963 m s^-2.  That is, this function returns the dynamic
    height anomaly divided by 9.7963 m s^-2; this being  the gravitational
    acceleration averaged over the surface of the global ocean (see page 46 of
    Griffies, 2004).  Hence, steric_height is the steric height anomaly with
    respect to a given reference pressure p_ref.

    Dynamic height anomaly is the geostrophic streamfunction for the difference
    between the horizontal velocity at the pressure concerned, p, and the
    horizontal velocity at p_ref.  Dynamic height anomaly is the exact
    geostrophic streamfunction in isobaric surfaces even though the
    gravitational acceleration varies with latitude and pressure.  Steric
    height anomaly, being simply proportional to dynamic height anomaly, is
    also an exact geostrophic streamfunction in an isobaric surface (up to the
    constant of proportionality, 9.7963 m s^-2).

    Note however that steric_height is not exactly the height (in meters) of an
    isobaric surface above a geopotential surface.  It is tempting to divide
    dynamic height anomaly by the local value of the gravitational
    acceleration, but doing so robs the resulting quantity of either being

    (i)  an exact geostrophic streamfunction, or
    (ii) exactly the height of an isobaric surface above a geopotential
    surface.

    By using a constant value of the gravitational acceleration, we have
    retained the first of these two properties.  So it should be noted that
    because of the variation of the gravitational acceleration with latitude,
    steric_height does not exactly represent the height of an isobaric surface
    above a geopotential surface under the assumption of geostropy.

    The reference values used for the specific volume anomaly are
    SSO = 35.16504 g/kg and CT = 0 deg C.  This function calculates specific
    volume anomaly using the computationally efficient 48-term expression for
    specific volume of McDougall et al. (2011).  Note that the 48-term equation
    has been fitted in a restricted range of parameter space, and is most
    accurate inside the "oceanographic funnel" described in McDougall et al.
    (2011) and IOC et al. (2010).  For dynamical oceanography we may take the
    48-term rational function expression for density as essentially reflecting
    the full accuracy of TEOS-10.  The GSW internal library function
    "infunnel(SA,CT,p)" is available to be used if one wants to test if some of
    one's data lies outside this "funnel".

    Parameters
    ----------
    SA : array_like
         Absolute salinity [g kg :sup:`-1`]
    CT : array_like
         Conservative Temperature [:math:`^\circ` C (ITS-90)]
    p : array_like
        pressure [dbar]
    p_ref : int, float, optional
        reference pressure, default = 0

    Returns
    -------
    steric_height : array_like
                    dynamic height anomaly divided by 9.7963 m s^-2  [m]

    Notes
    -----
    If p_ref exceeds the pressure of the deepest "bottle" on a vertical
    profile, the steric height anomaly for each "bottle" on the whole vertical
    profile is returned as NaN.

    See Also
    --------
    TODO

    Examples
    --------
    TODO

    References
    ----------
    .. [1] IOC, SCOR and IAPSO, 2010: The international thermodynamic equation
    of seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp. See section 3.27.

    .. [2] McDougall T.J., P.M. Barker, R. Feistel and D.R. Jackett, 2011:  A
    computationally efficient 48-term expression for the density of seawater
    in terms of Conservative Temperature, and related properties of seawater.

    .. [3] Griffies, S. M., 2004: Fundamentals of Ocean Climate Models.
    Princeton, NJ: Princeton University Press, 518 pp + xxxiv.

    Modifications:
    2010-05-20. Trevor McDougall and Paul Barker.
    """

    p_ref = np.asanyarray(p_ref)

    p_ref = np.unique(p_ref)

    if not np.isscalar(p_ref):
        raise ValueError('The reference pressure p_ref must be unique')

    if (p_ref < 0).any():
        raise ValueError('The reference pressure p_ref must be positive')

    if (SA < 0).any():
        raise ValueError('The Absolute Salinity must be positive!')

    # Start of the calculation.
    if p.max() < p_ref.max():
        raise ValueError('The reference pressure p_ref is deeper than bottles')

    dynamic_height_anomaly = geo_strf_dyn_height(SA, CT, p, p_ref)
    const_grav = 9.7963  # Griffies, 2004.

    return dynamic_height_anomaly / const_grav
