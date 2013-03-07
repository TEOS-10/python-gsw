# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np

from constants import SSO, cp0, Kelvin, sfac, uPS
from constants import db2Pascal, gamma, P0, M_S, valence_factor
from library import entropy_part, entropy_part_zerop, gibbs
from library import gibbs_pt0_pt0, enthalpy_SSO_0_p, specvol_SSO_0_p
from gsw.utilities import match_args_return, strip_mask

__all__ = [
           #'deltaSA_from_SP',  TODO
           #'SA_Sstar_from_SP',  TODO
           'SR_from_SP',
           'SP_from_SR',
           #'SP_from_SA',  TODO: changed, check gsw_SAAR.m
           #'Sstar_from_SA',  TODO: changed, check gsw_SAAR.m
           #'SA_from_Sstar',  TODO: changed, check gsw_SAAR.m
           #'SP_from_Sstar',  TODO: changed, check gsw_SAAR.m
           'pt_from_CT',
           't_from_CT',
           'CT_from_pt',
           'pot_enthalpy_from_pt',
           'pt0_from_t',
           'pt_from_t',
           't90_from_t48',
           't90_from_t68',
           'z_from_p',  # TODO: New test case with geo_strf_dyn_height != None
           'p_from_z',
           'depth_from_z',
           'z_from_depth',
           'Abs_Pressure_from_p',
           'p_from_Abs_Pressure',
           'entropy_from_CT',
           'CT_from_entropy',
           'entropy_from_pt',
           'pt_from_entropy',
           'molality_from_SA',
           'ionic_strength_from_SA'
           ]

DEG2RAD = np.pi / 180.0
n0, n1, n2 = 0, 1, 2


@match_args_return
def pt_from_entropy(SA, entropy):
    r"""Calculates potential temperature with reference pressure p_ref = 0 dbar
    and with entropy as an input variable.

    Parameters
    ----------
    SA : array_like
         Absolute salinity [g kg :sup:`-1`]
    entropy : array_like
              specific entropy [J kg :sup:`-1` K :sup:`-1`]

    Returns
    -------
    pt : array_like
         potential temperature [:math:`^\circ` C (ITS-90)]
         with reference sea pressure (p_ref) = 0 dbar.

    See Also
    --------
    TODO

    Notes
    -----
    TODO

    Examples
    --------
    >>> import seawater.gibbs as gsw
    >>> SA = [34.7118, 34.8915, 35.0256, 34.8472, 34.7366, 34.7324]
    >>> entropy = [400.3892, 395.4378, 319.8668, 146.7910, 98.6471, 62.7919]
    >>> gsw.pt_from_entropy(SA, entropy)
    array([ 28.78317983,  28.42095483,  22.78495274,  10.23053207,
             6.82921333,   4.32453778])

    References
    ----------
    .. [1] IOC, SCOR and IAPSO, 2010: The international thermodynamic equation
    of seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp. See appendix  A.10.

    Modifications:
    2011-04-03. Trevor McDougall and Paul Barker.
    """

    SA = np.maximum(SA, 0)

    part1 = 1 - SA / SSO
    part2 = 1 - 0.05 * part1
    ent_SA = (cp0 / Kelvin) * part1 * (1 - 1.01 * part1)
    c = (entropy - ent_SA) * part2 / cp0
    pt = Kelvin * (np.exp(c) - 1)
    dentropy_dt = cp0 / ((Kelvin + pt) * part2)  # Initial dentropy_dt.

    for Number_of_iterations in range(0, 3):
        pt_old = pt
        dentropy = entropy_from_pt(SA, pt_old) - entropy
        # This is half way through the modified method
        # (McDougall and Wotherspoon, 2012)
        pt = pt_old - dentropy / dentropy_dt
        ptm = 0.5 * (pt + pt_old)
        dentropy_dt = -gibbs_pt0_pt0(SA, ptm)
        pt = pt_old - dentropy / dentropy_dt

    """maximum error of 2.2x10^-6 degrees C for one iteration. maximum error is
    1.4x10^-14 degrees C for two iterations (two iterations is the default,
    "for Number_of_iterations = 1:2")."""

    return pt


@match_args_return
def t_from_CT(SA, CT, p):
    r"""Calculates *in-situ* temperature from Conservative Temperature of
    seawater.

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
    t : array_like
        in situ temperature [:math:`^\circ` C (ITS-90)]

    See Also
    --------
    TODO

    Notes
    -----
    TODO

    Examples
    --------
    >>> import gsw
    >>> SA = [34.7118, 34.8915, 35.0256, 34.8472, 34.7366, 34.7324]
    >>> CT = [28.8099, 28.4392, 22.7862, 10.2262, 6.8272, 4.3236]
    >>> p = [10, 50, 125, 250, 600, 1000]
    >>> gsw.t_from_CT(SA, CT, p)
    array([ 28.78558023,  28.43287225,  22.81032309,  10.26001075,
             6.8862863 ,   4.40362445])

    References
    ----------
    .. [1] IOC, SCOR and IAPSO, 2010: The international thermodynamic equation
    of seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp. See sections 3.1 and 3.3.

    Modifications:
    2011-03-29. Trevor McDougall and Paul Barker
    """

    pt0 = pt_from_CT(SA, CT)
    return pt_from_t(SA, pt0, 0, p)


@match_args_return
def pt_from_t(SA, t, p, p_ref=0):
    r"""Calculates potential temperature with the general reference pressure,
    pr, from in situ temperature.

    Parameters
    ----------
    SA : array_like
         Absolute salinity [g kg :sup:`-1`]
    t : array_like
        in situ temperature [:math:`^\circ` C (ITS-90)]
    p : array_like
        pressure [dbar]
    p_ref : int, float, optional
         reference pressure, default = 0

    Returns
    -------
    pt : array_like
         potential temperature [:math:`^\circ` C (ITS-90)]

    See Also
    --------
    TODO

    Notes
    -----
    This function calls `entropy_part` which evaluates entropy except for the
    parts which are a function of Absolute Salinity alone. A faster routine
    exists pt0_from_t(SA,t,p) if p_ref is indeed zero dbar.

    Examples
    --------
    >>> import gsw
    >>> SA = [34.7118, 34.8915, 35.0256, 34.8472, 34.7366, 34.7324]
    >>> t = [28.7856, 28.4329, 22.8103, 10.2600, 6.8863, 4.4036]
    >>> p = [10, 50, 125, 250, 600, 1000]
    >>> gsw.pt_from_t(SA, t, p)
    array([ 28.78319682,  28.42098334,  22.7849304 ,  10.23052366,
             6.82923022,   4.32451057])
    >>> gsw.pt_from_t(SA, t, p, pr = 1000)
    array([ 29.02665528,  28.662375  ,  22.99149634,  10.35341725,
             6.92732954,   4.4036    ])

    References
    ----------
    .. [1] IOC, SCOR and IAPSO, 2010: The international thermodynamic equation
    of seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp. See section 3.1.

    .. [2] McDougall T.J., P.M. Barker, R. Feistel and D.R. Jackett, 2011:  A
    computationally efficient 48-term expression for the density of seawater
    in terms of Conservative Temperature, and related properties of seawater.

    Modifications:
    2011-03-29. Trevor McDougall, David Jackett, Claire Roberts-Thomson and
    Paul Barker.
    """

    p_ref = np.asanyarray(p_ref)

    SA = np.maximum(SA, 0)

    s1 = SA * 35. / SSO

    pt = (t + (p - p_ref) * (8.65483913395442e-6 -
                       s1 * 1.41636299744881e-6 -
              (p + p_ref) * 7.38286467135737e-9 +
                        t * (-8.38241357039698e-6 +
                       s1 * 2.83933368585534e-8 +
                        t * 1.77803965218656e-8 +
              (p + p_ref) * 1.71155619208233e-10)))

    dentropy_dt = cp0 / ((Kelvin + pt) * (1 - 0.05 *
                                       (1 - SA / SSO)))

    true_entropy_part = entropy_part(SA, t, p)

    for Number_of_iterations in range(0, 2, 1):
        pt_old = pt
        dentropy = entropy_part(SA, pt_old, p_ref) - true_entropy_part
        pt = pt_old - dentropy / dentropy_dt  # half way through the method
        ptm = 0.5 * (pt + pt_old)
        dentropy_dt = -gibbs(n0, n2, n0, SA, ptm, p_ref)
        pt = pt_old - dentropy / dentropy_dt

    """maximum error of 6.3x10^-9 degrees C for one iteration.  maximum error
    is 1.8x10^-14 degrees C for two iterations (two iterations is the default,
    "for Number_of_iterations = 1:2).  These errors are over the full
    "oceanographic funnel" of McDougall et al. (2010), which reaches down to
    p = 8000 dbar."""

    return pt


@match_args_return
def CT_from_pt(SA, pt):
    r"""Calculates Conservative Temperature of seawater from potential
    temperature (whose reference sea pressure is zero dbar).

    Parameters
    ----------
    SA : array_like
         Absolute salinity [g kg :sup:`-1`]
    pt : array_like
         potential temperature referenced to a sea pressure of zero dbar
         [:math:`^\circ` C (ITS-90)]

    Returns
    -------
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
    >>> import gsw
    >>> SA = [34.7118, 34.8915, 35.0256, 34.8472, 34.7366, 34.7324]
    >>> pt = [28.7832, 28.4209, 22.7850, 10.2305, 6.8292, 4.3245]
    >>> gsw.CT_from_pt(SA, pt)
    array([ 28.80992302,  28.43914426,  22.78624661,  10.22616561,
             6.82718342,   4.32356518])

    References
    ----------
    .. [1] IOC, SCOR and IAPSO, 2010: The international thermodynamic equation
    of seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp. See section 3.3.

    Modifications:
    2011-03-29. David Jackett, Trevor McDougall and Paul Barker.
    """

    SA, pt, mask = strip_mask(SA, pt)

    pot_enthalpy = pot_enthalpy_from_pt(SA, pt)

    CT = pot_enthalpy / cp0

    return np.ma.array(CT, mask=mask, copy=False)


@match_args_return
def pot_enthalpy_from_pt(SA, pt):
    r"""Calculates the potential enthalpy of seawater from potential
    temperature (whose reference sea pressure is zero dbar).

    Parameters
    ----------
    SA : array_like
         Absolute salinity [g kg :sup:`-1`]
    pt : array_like
         potential temperature referenced to a sea pressure of zero dbar
         [:math:`^\circ` C (ITS-90)]

    Returns
    -------
    pot_enthalpy : array_like
                   potential enthalpy [J kg :sup:`-1`]

    See Also
    --------
    TODO

    Notes
    -----
    TODO

    Examples
    --------
    >>> import gsw
    >>> SA = [34.7118, 34.8915, 35.0256, 34.8472, 34.7366, 34.7324]
    >>> pt = [28.7832, 28.4209, 22.7850, 10.2305, 6.8292, 4.3245]
    >>> gsw.pot_enthalpy_from_pt(SA, pt)
    array([ 115005.40853458,  113525.30870246,   90959.68769935,
             40821.50280454,   27253.21472227,   17259.10131183])

    References
    ----------
    .. [1] IOC, SCOR and IAPSO, 2010: The international thermodynamic equation
    of seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp. See section 3.2.

    Modifications:
    2011-03-29. David Jackett, Trevor McDougall and Paul Barker
    """

    SA, pt, mask = strip_mask(SA, pt)

    SA = np.maximum(SA, 0)

    x2 = sfac * SA
    x = np.sqrt(x2)
    y = pt * 0.025  # Normalize for F03 and F08

    pot_enthalpy = (61.01362420681071 + y * (168776.46138048015 +
    y * (-2735.2785605119625 + y * (2574.2164453821433 +
    y * (-1536.6644434977543 + y * (545.7340497931629 +
    (-50.91091728474331 - 18.30489878927802 * y) * y))))) +
    x2 * (268.5520265845071 + y * (-12019.028203559312 +
    y * (3734.858026725145 + y * (-2046.7671145057618 +
    y * (465.28655623826234 + (-0.6370820302376359 -
    10.650848542359153 * y) * y)))) +
    x * (937.2099110620707 + y * (588.1802812170108 +
    y * (248.39476522971285 + (-3.871557904936333 -
    2.6268019854268356 * y) * y)) +
    x * (-1687.914374187449 + x * (246.9598888781377 +
    x * (123.59576582457964 - 48.5891069025409 * x)) +
    y * (936.3206544460336 +
    y * (-942.7827304544439 + y * (369.4389437509002 +
    (-33.83664947895248 - 9.987880382780322 * y) * y)))))))

    """The above polynomial for pot_enthalpy is the full expression for
    potential enthalpy in terms of SA and pt, obtained from the Gibbs function
    as below.  It has simply collected like powers of x and y so that it is
    computationally faster than calling the Gibbs function twice as is done in
    the commented code below. When this code below is run, the results are
    identical to calculating pot_enthalpy as above, to machine precision.

    g000 = gibbs(n0, n0, n0, SA, pt, 0)
    g010 = gibbs(n0, n1, n0, SA, pt, 0)
    pot_enthalpy = g000 - (Kelvin + pt) * g010

    This is the end of the alternative code
    %timeit gsw.CT_from_pt(SA, pt)
    1000 loops, best of 3: 1.34 ms per loop <- calling gibbs
    1000 loops, best of 3: 254 us per loop <- standard
    """

    return np.ma.array(pot_enthalpy, mask=mask, copy=False)


@match_args_return
def pt0_from_t(SA, t, p):
    r"""Calculates potential temperature with reference pressure, pr = 0 dbar.
    The present routine is computationally faster than the more general
    function "pt_from_t(SA, t, p, pr)" which can be used for any reference
    pressure value.

    Parameters
    ----------
    SA : array_like
         Absolute salinity [g kg :sup:`-1`]
    t : array_like
        in situ temperature [:math:`^\circ` C (ITS-90)]
    p : array_like
        pressure [dbar]

    Returns
    -------
    pt0 : array_like
          potential temperature relative to 0 dbar [:math:`^\circ` C (ITS-90)]

    See Also
    --------
    entropy_part, gibbs_pt0_pt0, entropy_part_zerop

    Notes
    -----
    pt_from_t  has the same result (only slower)

    Examples
    --------
    >>> import gsw
    >>> SA = [34.7118, 34.8915, 35.0256, 34.8472, 34.7366, 34.7324]
    >>> t = [28.7856, 28.4329, 22.8103, 10.2600, 6.8863, 4.4036]
    >>> p = [10, 50, 125, 250, 600, 1000]
    >>> gsw.pt0_from_t(SA, t, p)
    array([ 28.78319682,  28.42098334,  22.7849304 ,  10.23052366,
             6.82923022,   4.32451057])

    References
    ----------
    .. [1] IOC, SCOR and IAPSO, 2010: The international thermodynamic equation
    of seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp. See section 3.1.

    .. [2] McDougall T. J., D. R. Jackett, P. M. Barker, C. Roberts-Thomson,
    R. Feistel and R. W. Hallberg, 2010:  A computationally efficient 25-term
    expression for the density of seawater in terms of Conservative
    Temperature, and related properties of seawater.

    Modifications:
    2011-03-29. Trevor McDougall, David Jackett, Claire Roberts-Thomson and
    Paul Barker.
    """

    SA = np.maximum(SA, 0)

    s1 = SA * (35. / SSO)

    pt0 = t + p * (8.65483913395442e-6 -
             s1 * 1.41636299744881e-6 -
              p * 7.38286467135737e-9 +
              t * (-8.38241357039698e-6 +
             s1 * 2.83933368585534e-8 +
              t * 1.77803965218656e-8 +
              p * 1.71155619208233e-10))

    dentropy_dt = cp0 / ((Kelvin + pt0) * (1 - 0.05 * (1 - SA / SSO)))

    true_entropy_part = entropy_part(SA, t, p)

    for Number_of_iterations in range(0, 2, 1):
        pt0_old = pt0
        dentropy = entropy_part_zerop(SA, pt0_old) - true_entropy_part
        # Half way the mod. method (McDougall and Wotherspoon, 2012).
        pt0 = pt0_old - dentropy / dentropy_dt
        pt0m = 0.5 * (pt0 + pt0_old)
        dentropy_dt = -gibbs_pt0_pt0(SA, pt0m)
        pt0 = pt0_old - dentropy / dentropy_dt

    """maximum error of 6.3x10^-9 degrees C for one iteration. maximum error is
    1.8x10^-14 degrees C for two iterations (two iterations is the default,
    "for Number_of_iterations = 1:2").  These errors are over the full
    "oceanographic funnel" of McDougall et al. (2010), which reaches down to
    p = 8000 dbar."""

    return pt0


@match_args_return
def pt_from_CT(SA, CT):
    r"""Calculates potential temperature (with a reference sea pressure of zero
    dbar) from Conservative Temperature.

    Parameters
    ----------
    SA : array_like
         Absolute salinity [g kg :sup:`-1`]
    CT : array_like
         Conservative Temperature [:math:`^\circ` C (ITS-90)]

    Returns
    -------
    pt : array_like
         potential temperature referenced to a sea pressure of zero dbar
         [:math:`^\circ` C (ITS-90)]

    See Also
    --------
    specvol_anom

    Notes
    -----
    This function uses 1.5 iterations through a modified Newton-Raphson (N-R)
    iterative solution procedure, starting from a rational-function-based
    initial condition for both pt and dCT_dpt.

    Examples
    --------
    >>> import gsw
    >>> SA = [34.7118, 34.8915, 35.0256, 34.8472, 34.7366, 34.7324]
    >>> CT = [28.8099, 28.4392, 22.7862, 10.2262, 6.8272, 4.3236]
    >>> gsw.pt_from_CT(SA, CT)
    array([ 28.78317705,  28.4209556 ,  22.78495347,  10.23053439,
             6.82921659,   4.32453484])

    References
    ----------
    .. [1] IOC, SCOR and IAPSO, 2010: The international thermodynamic equation
    of seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp. See sections 3.1 and 3.3.

    .. [2] McDougall T. J., D. R. Jackett, P. M. Barker, C. Roberts-Thomson,
    R. Feistel and R. W. Hallberg, 2010:  A computationally efficient 25-term
    expression for the density of seawater in terms of Conservative
    Temperature, and related properties of seawater.

    Modifications:
    2011-03-29. Trevor McDougall, David Jackett, Claire Roberts-Thomson and
    Paul Barker.
    """

    SA, CT, mask = strip_mask(SA, CT)
    SA = np.maximum(SA, 0)

    s1 = SA * 35. / SSO

    a0 = -1.446013646344788e-2
    a1 = -3.305308995852924e-3
    a2 = 1.062415929128982e-4
    a3 = 9.477566673794488e-1
    a4 = 2.166591947736613e-3
    a5 = 3.828842955039902e-3

    b0 = 1.000000000000000e+0
    b1 = 6.506097115635800e-4
    b2 = 3.830289486850898e-3
    b3 = 1.247811760368034e-6

    a5CT = a5 * CT
    b3CT = b3 * CT
    CT_factor = (a3 + a4 * s1 + a5CT)
    pt_num = a0 + s1 * (a1 + a2 * s1) + CT * CT_factor
    pt_den = b0 + b1 * s1 + CT * (b2 + b3CT)
    pt = pt_num / pt_den

    dCT_dpt = pt_den / (CT_factor + a5CT - (b2 + b3CT + b3CT) * pt)

    # 1.5 iterations through the modified Newton-Rapshon iterative method
    CT_diff = CT_from_pt(SA, pt) - CT
    pt_old = pt
    pt = pt_old - CT_diff / dCT_dpt  # 1/2-way through the 1st modified N-R.
    ptm = 0.5 * (pt + pt_old)

    # This routine calls gibbs_pt0_pt0(SA, pt0) to get the second derivative of
    # the Gibbs function with respect to temperature at zero sea pressure.

    dCT_dpt = -(ptm + Kelvin) * gibbs_pt0_pt0(SA, ptm) / cp0
    pt = pt_old - CT_diff / dCT_dpt  # End of 1st full modified N-R iteration.
    CT_diff = CT_from_pt(SA, pt) - CT
    pt_old = pt
    pt = pt_old - CT_diff / dCT_dpt  # 1.5 iterations of the modified N-R.
    # Abs max error of result is 1.42e-14 deg C.
    return np.ma.array(pt, mask=mask, copy=False)


@match_args_return
def z_from_p(p, lat, geo_strf_dyn_height=None):
    r"""Calculates height from sea pressure using the computationally-efficient
    48-term expression for density in terms of SA, CT and p (McDougall et
    al., 2011).  Dynamic height anomaly, geo_strf_dyn_height, if provided, must
    be computed with its pr=0 (the surface.)

    Parameters
    ----------
    p : array_like
        pressure [dbar]
    lat : array_like
          latitude in decimal degrees north [-90..+90]
    geo_strf_dyn_height : float, optional
                          dynamic height anomaly [ m :sup:`2` s :sup:`-2` ]

    Returns
    -------
    z : array_like
        height [m]

    See Also
    --------
    # FIXME: enthalpy_SSO_0_CT25, changed!


    Examples
    --------
    >>> import gsw
    >>> p = [10, 50, 125, 250, 600, 1000]
    >>> lat = 4
    >>> gsw.z_from_p(p, lat)
    array([  -9.94460074,  -49.71817465, -124.2728275 , -248.47044828,
           -595.82618014, -992.0931748 ])

    Notes
    -----
    At sea level z = 0, and since z (HEIGHT) is defined to be positive upwards,
    it follows that while z is positive in the atmosphere, it is NEGATIVE in
    the ocean.

    References
    ----------
    .. [1] IOC, SCOR and IAPSO, 2010: The international thermodynamic equation
    of seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.

    .. [2] McDougall T.J., P.M. Barker, R. Feistel and D.R. Jackett, 2011:  A
    computationally efficient 48-term expression for the density of seawater
    in terms of Conservative Temperature, and related properties of seawater.

    .. [3] Moritz (2000) Goedetic reference system 1980. J. Geodesy, 74,
    128-133.

    Modifications:
    2011-03-26. Trevor McDougall, Claire Roberts-Thomson and Paul Barker.
    """

    if not geo_strf_dyn_height:
        geo_strf_dyn_height = np.zeros_like(p)

    X = np.sin(lat * DEG2RAD)
    sin2 = X ** 2
    B = 9.780327 * (1.0 + (5.2792e-3 + (2.32e-5 * sin2)) * sin2)
    A = -0.5 * gamma * B
    C = enthalpy_SSO_0_p(p) - geo_strf_dyn_height

    return -2 * C / (B + np.sqrt(B ** 2 - 4 * A * C))


@match_args_return
def p_from_z(z, lat, geo_strf_dyn_height=0):
    r"""Calculates sea pressure from height using computationally-efficient
    48-term expression for density, in terms of SA, CT and p (McDougall et al.,
    2011).  Dynamic height anomaly, geo_strf_dyn_height, if provided, must be
    computed with its pr=0 (the surface.)

    Parameters
    ----------
    z : array_like
        height [m]
    lat : array_like
          latitude in decimal degrees north [-90..+90]
    geo_strf_dyn_height : float, optional
                          dynamic height anomaly [ m :sup:`2` s :sup:`-2` ]
                          The reference pressure (p_ref) of geo_strf_dyn_height
                          must be zero (0) dbar.

    Returns
    -------
    p : array_like
        pressure [dbar]

    See Also
    --------
    #FIXME: specvol_SSO_0_CT25, enthalpy_SSO_0_CT25, changed!

    Examples
    --------
    >>> import gsw
    >>> z = [-10., -50., -125., -250., -600., -1000.]
    >>> lat = 4.
    >>> gsw.p_from_z(z, lat)
    array([  10.05521794,   50.2711751,  125.6548857,  251.23284504,
            602.44050752, 1003.07609807])
    >>> z = [9.94460074, 49.71817465, 124.2728275, 248.47044828, 595.82618014,
    ...      992.0931748]
    >>> gsw.p_from_z(z, lat)
    array([   10.,    50.,   125.,   250.,   600.,  1000.])

    Notes
    -----
    Height (z) is NEGATIVE in the ocean. Depth is -z. Depth is not used in the
    gibbs library.

    References
    ----------
    .. [1] IOC, SCOR and IAPSO, 2010: The international thermodynamic equation
    of seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.

    .. [2] McDougall T.J., P.M. Barker, R. Feistel and D.R. Jackett, 2011: A
    computationally efficient 48-term expression for the density of seawater
    in terms of Conservative Temperature, and related properties of seawater.

    .. [3] Moritz (2000) Goedetic reference system 1980. J. Geodesy, 74,
    128-133.

    .. [4] Saunders, P. M., 1981: Practical conversion of pressure to depth.
    Journal of Physical Oceanography, 11, 573-574.

    Modifications:
    2010-08-26. Trevor McDougall, Claire Roberts-Thomson and Paul Barker.
    2011-03-26. Trevor McDougall, Claire Roberts-Thomson and Paul Barker
    """

    X = np.sin(lat * DEG2RAD)
    sin2 = X ** 2
    gs = 9.780327 * (1.0 + (5.2792e-3 + (2.32e-5 * sin2)) * sin2)

    # get the first estimate of p from Saunders (1981)
    c1 = 5.25e-3 * sin2 + 5.92e-3
    p = -2 * z / ((1 - c1) + np.sqrt((1 - c1) * (1 - c1) + 8.84e-6 * z))

    df_dp = db2Pascal * specvol_SSO_0_p(p)  # Initial value for f derivative.

    f = (enthalpy_SSO_0_p(p) + gs * (z - 0.5 * gamma * (z ** 2)) -
                                                          geo_strf_dyn_height)

    p_old = p
    p = p_old - f / df_dp
    p_mid = 0.5 * (p + p_old)
    df_dp = db2Pascal * specvol_SSO_0_p(p_mid)
    p = p_old - f / df_dp

    # After this one iteration through this modified Newton-Raphson iterative
    # procedure, the remaining error in p is at computer machine precision,
    # being no more than 1.6e-10 dbar.

    return p


@match_args_return
def t90_from_t48(t48):
    r"""Converts IPTS-48 temperature to International Temperature Scale 1990
    (ITS-90) temperature.  This conversion should be applied to all in-situ
    data collected prior to 31/12/1967.

    Parameters
    ---------
    t48 : array_like
          in-situ temperature [:math:`^\circ` C (ITPS-48)]

    Returns
    -------
    t90 : array_like
          in-situ temperature [:math:`^\circ` C (ITS-90)]

    References
    ----------
    .. [1] International Temperature Scales of 1948, 1968 and 1990, an ICES
    note, available from http://www.ices.dk/ocean/procedures/its.htm

    Modifications:
    2011-03-29. Paul Barker and Trevor McDougall.
    """

    return (t48 - (4.4e-6) * t48 * (100 - t48)) / 1.00024


@match_args_return
def t90_from_t68(t68):
    r"""Converts IPTS-68 temperature to International Temperature Scale 1990
    (ITS-90) temperature.  This conversion should be applied to all in-situ
    data collected between 1/1/1968 and 31/12/1989.

    Parameters
    ---------
    t68 : array_like
          in-situ temperature [:math:`^\circ` C (ITPS-68)]

    Returns
    -------
    t90 : array_like
          in-situ temperature [:math:`^\circ` C (ITS-90)]

    References
    ----------
    .. [1] International Temperature Scales of 1948, 1968 and 1990, an ICES
    note, available from http://www.ices.dk/ocean/procedures/its.htm

    Modifications:
    2011-03-29. Paul Barker and Trevor McDougall.
    """

    # t90 = t68 / 1.00024
    return t68 * 0.999760057586179


def depth_from_z(z):
    r"""Calculates depth from height, z.  Note that in general height is
    negative in the ocean.

    Parameters
    ---------
    z : array_like
        height [m]

    Returns
    -------
    depth : array_like
        depth [m]

    Modifications:
    2011-03-26. Winston.
    """

    return -z


def z_from_depth(depth):
    r"""Calculates height, z, from depth.  Note that in general height is
    negative in the ocean.

    Parameters
    ---------
    depth : array_like
        depth [m]

    Returns
    -------
    z : array_like
        height [m]

    Modifications:
    2011-03-26. Winston.
    """

    return -depth


@match_args_return
def Abs_Pressure_from_p(p):
    r"""Calculates Absolute Pressure from sea pressure.  Note that Absolute
    Pressure is in Pa NOT dbar.

    Parameters
    ---------
    p : array_like
        sea pressure [dbar]

    Returns
    -------
    Absolute_Pressure : array_like
        Absolute Pressure [Pa]

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
    UNESCO (English), 196 pp. See Eqn. (2.2.1).

        Modifications:
    2011-03-29. Trevor McDougall & Paul Barker
    """

    return p * db2Pascal + P0


@match_args_return
def p_from_Abs_Pressure(Absolute_Pressure):
    r"""Calculates sea pressure from Absolute Pressure. Note that Absolute
    Pressure is in Pa NOT dbar.

    Parameters
    ---------
    Absolute_Pressure : array_like
                        Absolute Pressure [Pa]

    Returns
    -------
    p : array_like
        sea pressure [dbar]

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
    UNESCO (English), 196 pp. See Eqn. (2.2.1).

    Modifications:
    2011-03-29. Trevor McDougall & Paul Barker
    """

    return (Absolute_Pressure - P0) * 1. / db2Pascal


@match_args_return
def SR_from_SP(SP):
    r"""Calculates Reference Salinity from Practical Salinity.

    Parameters
    ---------
    SP : array_like
        Practical Salinity (PSS-78) [unitless]

    Returns
    -------
    SR : array_like
        Reference Salinity [g kg :sup:`-1`]

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
    UNESCO (English), 196 pp.

    Modifications:
    2011-03-27. Trevor McDougall & Paul Barker
    """

    return uPS * SP


@match_args_return
def SP_from_SR(SR):
    r"""Calculates Practical Salinity from Reference Salinity.

    Parameters
    ---------
    SR : array_like
        Reference Salinity [g kg :sup:`-1`]

    Returns
    -------
    SP : array_like
        Practical Salinity (PSS-78) [unitless]

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
    UNESCO (English), 196 pp.

    Modifications:
    2011-03-27. Trevor McDougall & Paul Barker
    """

    return 1. / uPS * SR


@match_args_return
def ionic_strength_from_SA(SA):
    r"""Calculates the ionic strength of seawater from Absolute Salinity.

    Parameters
    ----------
    SA : array_like
         Absolute salinity [g kg :sup:`-1`]

    Returns
    -------
    ionic_strength : array_like
                     ionic strength of seawater [mol kg :sup:`-1`]

    See Also
    --------
    TODO

    Notes
    -----
    TODO

    Examples
    --------
    >>> import gsw
    >>> SA = [34.7118, 34.8915, 35.0256, 34.8472, 34.7366, 34.7324]
    >>> gsw.ionic_strength_from_SA(SA)
    array([ 0.71298118,  0.71680567,  0.71966059,  0.71586272,  0.71350891,
            0.71341953])

    References
    ----------
    .. [1] IOC, SCOR and IAPSO, 2010: The international thermodynamic equation
    of seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp. See Table L.1.

    .. [2] Millero, F. J., R. Feistel, D. G. Wright, and T. J. McDougall, 2008:
    The composition of Standard Seawater and the definition of the
    Reference-Composition Salinity Scale, Deep-Sea Res. I, 55, 50-72.
    See Eqns. 5.9 and 5.12.

    Modifications:
    2011-03-29. Trevor McDougall and Paul Barker
    """

    # Molality of seawater in mol kg :sup:`-1`
    molality = molality_from_SA(SA)

    return 0.5 * valence_factor * molality


@match_args_return
def molality_from_SA(SA):
    r"""Calculates the molality of seawater from Absolute Salinity.

    Parameters
    ----------
    SA : array_like
         Absolute salinity [g kg :sup:`-1`]

    Returns
    -------
    molality : array_like
            seawater molality [mol kg :sup:`-1`]

    See Also
    --------
    TODO

    Notes
    -----
    TODO

    Examples
    --------
    >>> import gsw
    >>> SA = [34.7118, 34.8915, 35.0256, 34.8472, 34.7366, 34.7324]
    >>> gsw.molality(SA)
    array([ 1.14508476,  1.15122708,  1.15581223,  1.14971265,  1.14593231,
            1.14578877])

    References
    ----------
    .. [1] IOC, SCOR and IAPSO, 2010: The international thermodynamic equation
    of seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.

    Modifications:
    2011-03-29. Trevor McDougall & Paul Barker
    """

    # Molality of seawater in mol kg :sup:`-1`.
    SA = np.maximum(SA, 0)
    molality = SA / (M_S * (1000 - SA))

    return molality


@match_args_return
def entropy_from_pt(SA, pt):
    r"""Calculates specific entropy of seawater.

    Parameters
    ----------
    SA : array_like
         Absolute salinity [g kg :sup:`-1`]
    pt : array_like
         potential temperature [:math:`^\circ` C (ITS-90)]

    Returns
    -------
    entropy : array_like
              specific entropy [J kg :sup:`-1` K :sup:`-1`]

    See Also
    --------
    TODO

    Notes
    -----
    TODO

    Examples
    --------
    >>> import gsw
    >>> SA = [34.7118, 34.8915, 35.0256, 34.8472, 34.7366, 34.7324]
    >>> pt = [28.7832, 28.4210, 22.7850, 10.2305, 6.8292, 4.3245]
    >>> gsw.entropy_from_pt(SA, pt)
    array([ 400.38946744,  395.43839949,  319.86743859,  146.79054828,
             98.64691006,   62.79135672])

    References
    ----------
    .. [1] IOC, SCOR and IAPSO, 2010: The international thermodynamic equation
    of seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp. See appendix A.10.

    Modifications:
    2011-04-03. Trevor McDougall & Paul Barker
    """

    SA = np.maximum(SA, 0)
    return -gibbs(n0, n1, n0, SA, pt, 0)


@match_args_return
def entropy_from_CT(SA, CT):
    r"""Calculates specific entropy of seawater.

    Parameters
    ----------
    SA : array_like
         Absolute salinity [g kg :sup:`-1`]
    CT : array_like
         Conservative Temperature [:math:`^\circ` C (ITS-90)]

    Returns
    -------
    entropy : array_like
              specific entropy [J kg :sup:`-1` K :sup:`-1`]

    See Also
    --------
    TODO

    Notes
    -----
    TODO

    Examples
    --------
    >>> import gsw
    >>> SA = [34.7118, 34.8915, 35.0256, 34.8472, 34.7366, 34.7324]
    >>> CT = [28.8099, 28.4392, 22.7862, 10.2262, 6.8272, 4.3236]
    >>> gsw.entropy_from_CT(SA, CT)
    array([ 400.38916315,  395.43781023,  319.86680989,  146.79103279,
             98.64714648,   62.79185763])

    References
    ----------
    .. [1] IOC, SCOR and IAPSO, 2010: The international thermodynamic equation
    of seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp. See appendix A.10.

    Modifications:
    2011-04-04. Trevor McDougall & Paul Barker
    """

    SA = np.maximum(SA, 0)
    pt0 = pt_from_CT(SA, CT)
    return -gibbs(n0, n1, n0, SA, pt0, 0)


@match_args_return
def CT_from_entropy(SA, entropy):
    r"""Calculates Conservative Temperature with entropy as an input variable.

    Parameters
    ----------
    SA : array_like
         Absolute salinity [g kg :sup:`-1`]
    entropy : array_like
              specific entropy [J kg :sup:`-1` K :sup:`-1`]

    Returns
    -------
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
    >>> import gsw
    >>> SA = [34.7118, 34.8915, 35.0256, 34.8472, 34.7366, 34.7324]
    >>> entropy = [400.3892, 395.4378, 319.8668, 146.7910, 98.6471, 62.7919]
    >>> gsw.CT_from_entropy(SA, entropy)
    array([ 28.80990279,  28.43919923,  22.78619927,  10.22619767,
             6.82719674,   4.32360295])

    References
    ----------
    .. [1] IOC, SCOR and IAPSO, 2010: The international thermodynamic equation
    of seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp. See appendix  A.10.

    Modifications:
    2011-03-03. Trevor McDougall and Paul Barker.
    """

    SA = np.maximum(SA, 0)
    pt = pt_from_entropy(SA, entropy)
    return CT_from_pt(SA, pt)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
