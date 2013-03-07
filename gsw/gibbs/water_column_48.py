# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np

from constants import db2Pascal
from earth import grav
from density_enthalpy_48 import rho
from gsw.utilities import match_args_return
from density_enthalpy_48 import rho_alpha_beta

__all__ = [
           'Nsquared',
           'Turner_Rsubrho',
           'IPV_vs_fNsquared_ratio'
           ]

# FIXME: match_args_return returns a
# ndarray instead of a tuple.
# Need to create a test for the match_args_return.


#@match_args_return
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
         Conservative Temperature [:math:`^\circ` C (ITS-90)]
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
            Mid pressure between p grid [dbar]

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
    2011-03-22. Trevor McDougall & Paul Barker
    """

    if lat is not None:
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


#@match_args_return
def Turner_Rsubrho(SA, CT, p):
    r"""Calculates the Turner angle and the Rsubrho as a function of pressure
    down a vertical water column.  These quantities express the relative
    contributions of the vertical gradients of Conservative Temperature and
    Absolute Salinity to the vertical stability (the square of the
    Brunt-Väisälä Frequency squared, N^2).  `Tu` and `Rsubrho` are evaluated at
    the mid pressure between the individual data points in the vertical.  This
    function uses computationally-efficient 48-term expression for density in
    terms of SA, CT and p (McDougall et al., 2011).  Note that in the
    double-diffusive literature, papers concerned with the "diffusive" form of
    double-diffusive convection often define the stability ratio as the
    reciprocal of what is defined here as the stability ratio.

    Parameters
    ----------
    SA : array_like
         Absolute Salinity  [g/kg]
    CT : array_like
         Conservative Temperature [:math:`^\circ` C (ITS-90)]
    p : array_like
        sea pressure  [dbar]

    Returns
    -------
    Tu : array_like
         Turner angle, on the same (M-1)xN grid as p_mid. [degrees of rotation]
    Rsubrho : array_like
              Stability Ratio, on the same (M-1)xN grid as p_mid. [unitless]
    p_mid : array_like
            Mid pressure between p grid [dbar]

    See Also
    --------
    TODO

    Notes
    -----
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
    UNESCO (English), 196 pp. See Eqns. (3.15.1) and (3.16.1).

    ..[2] McDougall T.J., P.M. Barker, R. Feistel and D.R. Jackett, 2011:  A
    computationally efficient 48-term expression for the density of
    seawater in terms of Conservative Temperature, and related properties
    of seawater.

    Modifications:
    2011-03-26. Trevor McDougall & Paul Barker
    """

    if SA.ndim == 1:
        raise ValueError('There must be at least 2 columns.')

    SA.clip(0, np.inf)

    SA, CT, p = np.broadcast_arrays(SA, CT, p)

    p_mid = 0.5 * (p[0:-1, ...] + p[1:, ...])
    SA_mid = 0.5 * (SA[0:-1, ...] + SA[1:, ...])
    CT_mid = 0.5 * (CT[0:-1, ...] + CT[1:, ...])

    dSA = SA[0:-1, ...] - SA[1:, ...]
    dCT = CT[0:-1, ...] - CT[1:, ...]

    [dummy, alpha, beta] = rho_alpha_beta(SA_mid, CT_mid, p_mid)

    """This function evaluates Tu and Rsubrho using the computationally
    efficient 48-term expression for density in terms of SA, CT and p. If one
    wanted to compute Tu and Rsubrho using the full TEOS-10 Gibbs function
    expression for density, the following lines of code would do that.

    pt_mid = pt_from_CT(SA_mid, CT_mid)
    pr0 = np.zeros_like(SA_mid)
    t_mid = pt_from_t(SA_mid, pt_mid, pr0, p_mid)
    beta = beta_const_CT_t_exact(SA_mid, t_mid, p_mid)
    alpha = alpha_wrt_CT_t_exact(SA_mid, t_mid, p_mid)
    """

    Tu = np.arctan2((alpha * dCT + beta * dSA), (alpha * dCT - beta * dSA))

    Tu = Tu * (180 / np.pi)

    Rsubrho = np.zeros_like(dSA) + np.NaN

    Inz = dSA != 0
    Rsubrho[Inz] = (alpha[Inz] * dCT[Inz]) / (beta[Inz] * dSA[Inz])

    return Tu, Rsubrho, p_mid


#@match_args_return
def IPV_vs_fNsquared_ratio(SA, CT, p, p_ref=0):
    r"""Calculates the ratio of the vertical gradient of potential density to
    the vertical gradient of locally-referenced potential density.  This
    ratio is also the ratio of the planetary Isopycnal Potential Vorticity
    (IPV) to f times N^2, hence the name for this variable,
    IPV_vs_fNsquared_ratio (see Eqn. (3.20.5) of IOC et al. (2010)).  The
    reference sea pressure, p_ref, of the potential density surface must
    have a constant value.

    IPV_vs_fNsquared_ratio is evaluated at the mid pressure between the
    individual data points in the vertical.  This function uses the
    computationally-efficient 48-term expression for density in terms of
    SA, CT and p (McDougall et al., 2011).

    Parameters
    ----------
    SA : array_like
         Absolute Salinity  [g/kg]
    CT : array_like
         Conservative Temperature [:math:`^\circ` C (ITS-90)]
    p : array_like
        sea pressure  [dbar]
    p_ref : int, float, optional
         reference pressure, default = 0

    Returns
    -------
    IPV_vs_fNsquared_ratio : array_like
         The ratio of the vertical gradient of potential density,
         on the same (M-1)xN grid as p_mid. [unitless]
         referenced to p_ref, to the vertical gradient of locally-
         referenced potential density.

    p_mid : array_like
            Mid pressure between p grid [dbar]

    See Also
    --------
    TODO

    Notes
    -----
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
    UNESCO (English), 196 pp. See Eqn. (3.20.5).

    ..[2] McDougall T.J., P.M. Barker, R. Feistel and D.R. Jackett, 2011:  A
    computationally efficient 48-term expression for the density of
    seawater in terms of Conservative Temperature, and related properties
    of seawater.

    Modifications:
    2011-03-23. Trevor McDougall & Paul Barker
    """
    p_ref = np.unique(np.asanyarray(p_ref))

    # BUG
    #if not np.isscalar(p_ref):
        #raise ValueError('The reference pressure p_ref must be unique')

    if SA.ndim == 1:
        raise ValueError('There must be at least 2 columns.')

    SA.clip(0, np.inf)

    SA, CT, p, p_ref = np.broadcast_arrays(SA, CT, p, p_ref)

    p_ref = p_ref[:-1, ...]

    p_mid = 0.5 * (p[0:-1, ...] + p[1:, ...])
    SA_mid = 0.5 * (SA[0:-1, ...] + SA[1:, ...])
    CT_mid = 0.5 * (CT[0:-1, ...] + CT[1:, ...])

    dSA = SA[0:-1, ...] - SA[1:, ...]
    dCT = CT[0:-1, ...] - CT[1:, ...]

    [dummy, alpha, beta] = rho_alpha_beta(SA_mid, CT_mid, p_mid)

    _, alpha, beta = rho_alpha_beta(SA_mid, CT_mid, p_mid)
    _, alpha_pref, beta_pref = rho_alpha_beta(SA_mid, CT_mid, p_ref)

    """This function calculates IPV_vs_fNsquared_ratio using the
    computationally efficient 48-term expression for density in terms of SA,
    CT and p.  If one wanted to compute this with the full TEOS-10 Gibbs
    function expression for density, the following lines of code will enable
    this.

    pt_mid = pt_from_CT(SA_mid, CT_mid)
    pr0 = np.zeros_like(SA_mid)
    t_mid = pt_from_t(SA_mid, pt_mid, pr0, p_mid)
    beta = beta_const_CT_t_exact(SA_mid, t_mid, p_mid)
    alpha = alpha_wrt_CT_t_exact(SA_mid, t_mid, p_mid)
    beta_pref = beta_const_CT_t_exact(SA_mid, t_mid, p_ref)
    alpha_pref = alpha_wrt_CT_t_exact(SA_mid, t_mid, p_ref)
    """

    numerator = dCT * alpha_pref - dSA * beta_pref
    denominator = dCT * alpha - dSA * beta

    """IPV_vs_fNsquared_ratio = np.zeros_like(SA_mid) * np.NaN
    I = denominator != 0.
    IPV_vs_fNsquared_ratio[I] = numerator[I] / denominator[I]"""

    IPV_vs_fNsquared_ratio = numerator / denominator

    return IPV_vs_fNsquared_ratio, p_mid


if __name__ == '__main__':
    import doctest
    doctest.testmod()
