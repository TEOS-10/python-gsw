# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np

from conversions import t_from_CT
from gsw.utilities import match_args_return

__all__ = [
           'CT_freezing',
           't_freezing',
           #'brineSA_CT',  TODO
           #'brineSA_t'  TODO
           ]

# Constants:
c = (0.017947064327968736, -6.076099099929818, 4.883198653547851,
     -11.88081601230542, 13.34658511480257, -8.722761043208607,
     2.082038908808201, -7.389420998107497, -2.110913185058476,
     0.2295491578006229, -0.9891538123307282, -0.08987150128406496,
     0.3831132432071728, 1.054318231187074, 1.065556599652796,
     -0.7997496801694032, 0.3850133554097069, -2.078616693017569,
     0.8756340772729538, -2.079022768390933, 1.596435439942262,
     0.1338002171109174, 1.242891021876471)


@match_args_return
def CT_freezing(SA, p, saturation_fraction=1):
    r"""Calculates the Conservative Temperature at which seawater freezes.

    Parameters
    ----------
    SA : array_like
         Absolute Salinity  [g/kg]
    p : array_like
        sea pressure  [dbar]
    saturation_fraction : fraction between 0, 1.  The saturation fraction of
                          dissolved air in seawater.  Default is 0 or
                          completely saturated.

    Returns
    -------
    CT_freezing : array_like
          Conservative Temperature at freezing of
          seawater [:math:`^\circ` C (ITS-90)]

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
    UNESCO (English), 196 pp. See sections 3.33 and 3.34.

    Modifications:
    2011-11-04. Trevor McDougall, Paul Barker and Rainer Feistal.

    """

    SA, p, saturation_fraction = np.broadcast_arrays(SA, p,
                                                     saturation_fraction)
    if (SA < 0).any():
        raise ValueError('SA must be non-negative!')

    if np.logical_or(saturation_fraction < 0, saturation_fraction > 1).any():
        raise ValueError('Saturation_fraction MUST be between zero and one.')

    SA_r = SA * 1e-2
    x = np.sqrt(SA_r)
    p_r = p * 1e-4

    CT_freeze = c[0] + SA_r * (c[1] + x * (c[2] + x * (c[3] + x * (c[4] +
                x * (c[5] + c[6] * x))))) + p_r * (c[7] + p_r * (c[8] +
                c[9] * p_r)) + SA_r * p_r * (c[10] + p_r * (c[12] + p_r *
                (c[15] + c[21] * SA_r)) + SA_r * (c[13] + c[17] * p_r +
                c[19] * SA_r) + x * (c[11] + p_r * (c[14] + c[18] * p_r) +
                SA_r * (c[16] + c[20] * p_r + c[22] * SA_r)))

    """The error of this fit ranges between -5e-4 K and 6e-4 K when compared
    with the Conservative Temperature calculated from the exact in-situ
    freezing temperature which is found by a Newton-Raphson iteration of the
    equality of the chemical potentials of water in seawater and in ice.
    (Note that the in-situ freezing temperature can be found by this exact
    method using the function sea_ice_freezingtemperature_si in the SIA
    library)."""

    # Adjust for the effects of dissolved air.
    a, b = 0.014289763856964, 0.057000649899720
    # Note that a = 0.502500117621 / 35.16504

    CT_freeze = (CT_freeze - saturation_fraction * (1e-3) *
                  (2.4 - a * SA) * (1 + b * (1 - SA / 35.16504)))

    tmp = np.logical_or(p > 10000, SA > 120)
    out = np.logical_or(tmp, p + SA * 71.428571428571402 > 13571.42857142857)

    CT_freeze[out] = np.ma.masked

    return CT_freeze


@match_args_return
def t_freezing(SA, p, saturation_fraction=1):
    r"""Calculates the in-situ temperature at which seawater freezes.

    Parameters
    ----------
    SA : array_like
         Absolute Salinity  [g/kg]
    p : array_like
        sea pressure  [dbar]
    saturation_fraction : fraction between 0, 1.  The saturation fraction of
                          dissolved air in seawater.  Default is 0 or
                          completely saturated.

    Returns
    -------
    t_freezing : array_like
                  in-situ temperature at which seawater freezes
                  [:math:`^\circ` C (ITS-90)]

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
    UNESCO (English), 196 pp. See sections 3.33 and 3.34.

    Modifications:
    2011-11-03. Trevor McDougall, Paul Barker and Rainer Feistal.

    """

    """This function, t_freezing, calculates the in-situ freezing temperature,
    t_freezing, of seawater by first evaluating a polynomial of the
     Conservative Temperature at which seawater freezes, CT_freezing, using
    the GSW function CT_freezing.  The in-situ freezing temperature is then
    calculated using the GSW function t_from_CT.  However, if one wanted to
    compute the in-situ freezing temperature directly from a single polynomial
    expression without first calculating the Conservative Temperature at the
    freezing point, the following lines of code achieve this.  The error of the
    following fit is similar to that of the present function, t_freezing, and
    ranges between -8e-4 K and 3e-4 K when compared with the in-situ freezing
    temperature evaluated by Newton-Raphson iteration of the equality of the
    chemical potentials of water in seawater and in ice.  (Note that the
    in-situ freezing temperature can be found by this exact method using the
    function sea_ice_freezingtemperature_si in the SIA library).

    c = (0.002519, -5.946302841607319, 4.136051661346983,
        -1.115150523403847e1, 1.476878746184548e1, -1.088873263630961e1,
        2.961018839640730, -7.433320943962606, -1.561578562479883,
        4.073774363480365e-2, 1.158414435887717e-2, -4.122639292422863e-1,
        -1.123186915628260e-1, 5.715012685553502e-1, 2.021682115652684e-1,
        4.140574258089767e-2, -6.034228641903586e-1, -1.205825928146808e-2,
        -2.812172968619369e-1, 1.877244474023750e-2, -1.204395563789007e-1,
        2.349147739749606e-1, 2.748444541144219e-3)

    SA_r = SA * 1e-2
    x = np.sqrt(SA_r)
    p_r = p * 1e-4

    t_freeze = c[0] + SA_r * (c[1] + x * (c[2] + x * (c[3] + x * (c[4] + x *
               (c[5] + c[6] * x))))) + p_r * (c[7] + p_r * (c[8] + c[9] *
               p_r)) + SA_r * p_r * (c[10] + p_r * (c[12] + p_r * (c[15] +
               c[21] * SA_r)) + SA_r * (c[13] + c[17] * p_r + c[19] * SA_r) +
               x * (c[11] + p_r * (c[14] + c[18] * p_r)  + SA_r * (c[16] +
               c[20] * p_r + c[22] * SA_r)))

    Adjust for the effects of dissolved air
    t_freezing -= saturation_fraction * (1e-3) * (2.4 - SA / 70.33008)

    """

    SA, p, saturation_fraction = np.broadcast_arrays(SA, p,
                                                     saturation_fraction)
    if (SA < 0).any():
        raise ValueError('SA must be non-negative!')

    if np.logical_or(saturation_fraction < 0, saturation_fraction > 1).any():
        raise ValueError('Saturation_fraction MUST be between zero and one.')

    CT_freeze = CT_freezing(SA, p, saturation_fraction)

    t_freeze = t_from_CT(SA, CT_freeze, p)

    tmp = np.logical_or(p > 10000, SA > 120)
    out = np.logical_or(tmp, p + SA * 71.428571428571402 > 13571.42857142857)

    t_freeze[out] = np.ma.masked

    return t_freeze
