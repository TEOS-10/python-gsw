# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np

from constants import SSO
from conversions import t_from_CT
from gsw.utilities import match_args_return
from absolute_salinity_sstar_ct import CT_from_t

__all__ = [
           'CT_freezing',
           't_freezing',
           'brineSA_CT',
           'brineSA_t'
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

T = (0.002519, -5.946302841607319, 4.136051661346983,
    -1.115150523403847e1, 1.476878746184548e1, -1.088873263630961e1,
    2.961018839640730, -7.433320943962606, -1.561578562479883,
    4.073774363480365e-2, 1.158414435887717e-2, -4.122639292422863e-1,
    -1.123186915628260e-1, 5.715012685553502e-1, 2.021682115652684e-1,
    4.140574258089767e-2, -6.034228641903586e-1, -1.205825928146808e-2,
    -2.812172968619369e-1, 1.877244474023750e-2, -1.204395563789007e-1,
    2.349147739749606e-1, 2.748444541144219e-3)

    # Adjust for the effects of dissolved air.  Note that
# a = 0.502500117621 / 35.16504
a, b = 0.014289763856964, 0.057000649899720

P = (2.570124672768757e-1, -1.917742353032266e+1, -1.413382858617969e-2,
    -5.427484830917552e-1, -4.126621135193472e-4, -4.176407833276121e-7,
    4.688217641883641e-5, -3.039808885885726e-8, -4.990118091261456e-11,
    -9.733920711119464e-9, -7.723324202726337e-12, 7.121854166249257e-16,
    1.256474634100811e-12, 2.105103897918125e-15, 8.663811778227171e-19)


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

    SA_r = SA * 1e-2
    x = np.sqrt(SA_r)
    p_r = p * 1e-4

    t_freeze = T[0] + SA_r * (T[1] + x * (T[2] + x * (T[3] + x * (T[4] + x *
               (T[5] + T[6] * x))))) + p_r * (T[7] + p_r * (T[8] + T[9] *
               p_r)) + SA_r * p_r * (T[10] + p_r * (T[12] + p_r * (T[15] +
               T[21] * SA_r)) + SA_r * (T[13] + T[17] * p_r + T[19] * SA_r) +
               x * (T[11] + p_r * (T[14] + T[18] * p_r)  + SA_r * (T[16] +
               T[20] * p_r + T[22] * SA_r)))

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


@match_args_return
def brineSA_CT(CT, p, saturation_fraction=1):
    r"""Calculates the Absolute Salinity of seawater at the freezing
    temperature.  That is, the output is the Absolute Salinity of seawater,
    with the fraction saturation_fraction of dissolved air, that is in
    equilibrium with ice at Conservative Temperature CT and pressure p.  If the
    input values are such that there is no positive value of Absolute Salinity
    for which seawater is frozen, the output, brineSA_CT, is put equal to -99.

    Parameters
    ----------
    CT : array_like
         Conservative Temperature [:math:`^\circ` C (ITS-90)]
    p : array_like
        sea pressure  [dbar]
    saturation_fraction : fraction between 0, 1.  The saturation fraction of
                          dissolved air in seawater.  Default is 0 or
                          completely saturated.

    Returns
    -------
    brine_SA_CT : array_like
                 Absolute Salinity of seawater when it freezes [ g/kg ]

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
    UNESCO (English), 196 pp. See sections 3.33.

    Modifications:
    2011-28-03. Trevor McDougall and Paul Barker.

    """

    CT, p, saturation_fraction = np.broadcast_arrays(CT, p,
                                                     saturation_fraction)

    if np.logical_or(saturation_fraction < 0, saturation_fraction > 1).any():
        raise ValueError('Saturation_fraction MUST be between zero and one.')

    p_r = p * 1e-4
    # Form the first estimate of brine_SA_CT from a polynomial in CT and p_r.
    SA = -(CT + 9 * p_r) / 0.06  # A rough estimate to get the saturated CT.

    SA = np.maximum(SA, 0)

    CTsat = CT - (1 - saturation_fraction) * 1e-3 * (2.4 - a * SA) * (1 + b *
                 (1 - SA / SSO))

    SA = P[0] + p * (P[2] + P[4] * CTsat + p * (P[5] + CTsat * (P[7] + P[9] *
    CTsat) + p * (P[8] + CTsat * (P[10] + P[12] * CTsat) + p * (P[11] +
    P[13] * CTsat + P[14] * p)))) + CTsat * (P[1] + CTsat * (P[3] + P[6] * p))

    CT_freezing_zero_SA = (c[0] + p_r * (c[7] + p_r * (c[8] + c[9] * p_r)) -
                           saturation_fraction * 2.4e-3 * (1 + b))

    # Find CT > CT_freezing_zero_SA.  If this is the case, the input values
    # represent seawater that is not frozen (at any positive SA).
    Itw = (CT > CT_freezing_zero_SA)  # tw stands for "too warm"
    SA[Itw] = np.ma.masked

    # Find -SA_cut_off < SA < SA_cut_off, replace the above estimate of SA
    # with one based on (CT_freezing_zero_SA - CT).
    SA_cut_off = 2.5  # This is the band of SA within +- 2.5 g/kg of SA = 0,
                      # which we treat differently in calculating the initial
                      # values of both SA and dCT_dSA.

    Ico = (np.abs(SA) < SA_cut_off)
    Icoa = np.logical_and(SA < 0, SA >= -SA_cut_off)

    SA[Icoa] = 0

    # Find SA < -SA_cut_off, set them to NaN.
    SA[SA < -SA_cut_off] = np.ma.masked

    # Form the first estimate of dCT_dSA, the derivative of CT with respect
    # to SA at fixed p.
    SA_r = 0.01 * SA
    x = np.sqrt(SA_r)

    dCT_dSA_part = 2 * c[1] + x * (3 * c[2] + x * (4 * c[3] + x * (5 * c[4] +
                   x * (6 * c[5] + 7 * c[6] * x)))) + p_r * (2 * c[10] + p_r *
                   (2 * c[12] + p_r * (2 * c[15] + 4 * c[21] * x * x)) + x *
                   x * (4 * c[13] + 4 * c[17] * p_r + 6 * c[19] * x * x) + x *
                   (3 * c[11] + 3 * p_r * (c[14] + c[18] * p_r) + x * x * (5 *
                   c[16] + 5 * c[20] * p_r + 7 * c[22] * x * x)))

    dCT_dSA = 0.5 * 0.01 * dCT_dSA_part - saturation_fraction * 1e-3 * (-a *
              (1 + b * (1 - SA / SSO)) - b * (2.4 - a * SA) / SSO)

    # Now replace the estimate of SA with the one based on
    # (CT_freezing_zero_SA - CT) when (np.abs(SA) < SA_cut_off).
    SA[Ico] = (CT[Ico] - CT_freezing_zero_SA[Ico]) / dCT_dSA[Ico]

    # Begin the modified Newton-Raphson method to solve the root of
    # CT_freezing = CT for SA.
    Number_of_Iterations = 2
    for I_iter in range(0, Number_of_Iterations):
        # CT_freezing temperature function evaluation (the forward function
        # evaluation), the same as CT_freezing(SA, p, saturation_fraction).

        SA_r = 0.01 * SA
        x = np.sqrt(SA_r)
        SA_old = SA
        CT_freeze = (c[0] + SA_r * (c[1] + x * (c[2] + x * (c[3] + x * (c[4] +
                     x * (c[5] + c[6] * x))))) + p_r * (c[7] + p_r * (c[8] +
                     c[9] * p_r)) + SA_r * p_r * (c[10] + p_r * (c[12] + p_r *
                     (c[15] + c[21] * SA_r)) + SA_r * (c[13] + c[17] * p_r +
                     c[19] * SA_r) + x * (c[11] + p_r * (c[14] + c[18] * p_r) +
                     SA_r * (c[16] + c[20] * p_r + c[22] * SA_r))) -
                     saturation_fraction * 1e-3 * (2.4 - a * SA) * (1 + b *
                     (1 - SA / SSO)))

        SA = SA_old - (CT_freeze - CT) / dCT_dSA

        # Half-way point of the modified Newton-Raphson solution method.
        SA_r = 0.5 * 0.01 * (SA + SA_old)  # The mean value of SA and SA_old.
        x = np.sqrt(SA_r)

        dCT_dSA_part = 2 * c[1] + x * (3 * c[2] + x * (4 * c[3] + x * (5 *
                       c[4] + x * (6 * c[5] + 7 * c[6] * x)))) + p_r * (2 *
                       c[10] + p_r * (2 * c[12] + p_r * (2 * c[15] + 4 *
                       c[21] * x * x)) + x * x * (4 * c[13] + 4 * c[17] *
                       p_r + 6 * c[19] * x * x) + x * (3 * c[11] + 3 * p_r *
                       (c[14] + c[18] * p_r) + x * x * (5 * c[16] + 5 * c[20] *
                       p_r + 7 * c[22] * x * x)))

        dCT_dSA = (0.5 * 0.01 * dCT_dSA_part - saturation_fraction * 1e-3 *
                  (-a * (1 + b * (1 - SA / SSO)) - b * (2.4 - a * SA) / SSO))

        SA = SA_old - (CT_freeze - CT) / dCT_dSA

    """The following lines of code, if implemented, calculates the error of
    this function in terms of Conservative Temperature, CT_error.  With
    Number_of_Iterations = 1, the maximum error in CT is 2x10^-7 C.  With
    Number_of_Iterations = 2, the maximum error in CT is 7x10^-15 C, which is
    the machine precision of the computer.  Number_of_Iterations = 2 is what
    we recommend.

    SA_r = 0.01 * SA
    x = np.sqrt(SA_r)
    CT_freeze = c[0] + SA_r * (c[1] + x * (c[2] + x * (c[3] + x * (c[4] + x *
                (c[5] + c[6] * x))))) + p_r * (c[7] + p_r * (c[8] + c[9] *
                p_r)) + SA_r * p_r * (c[10] + p_r * (c[12] + p_r * (c[15] +
                c[21] * SA_r)) + SA_r * (c[13] + c[17] * p_r + c[19] * SA_r) +
                x * (c[11] + p_r * (c[14] + c[18] * p_r) + SA_r * (c[16] +
                c[20] * p_r + c[22] * SA_r))) - saturation_fraction * 1e-3 *
                (2.4 - a * SA) * (1 + b * (1 - SA / SSO))

    CT_error = np.abs(CT_freeze - CT)

    tmp = np.logical_or(p > 10000, SA > 120
    out = np.logical_and(tmp, p + SA * 71.428571428571402 > 13571.42857142857)
    CT_error[out] = np.ma.masked
    """

    brine_SA_CT = SA

    tmp = np.logical_or(p > 10000, SA > 120)
    out = np.logical_and(tmp, p + SA * 71.428571428571402 > 13571.42857142857)

    brine_SA_CT[out] = np.ma.masked

    # If the CT input is too warm, then there is no (positive) value of SA
    # that represents frozen seawater.
    brine_SA_CT[Itw] = -99  # NOTE: Mask these?

    return brine_SA_CT


@match_args_return
def brineSA_t(t, p, saturation_fraction=1):
    r"""Calculates the Absolute Salinity of seawater at the freezing
    temperature.  That is, the output is the Absolute Salinity of seawater,
    with the fraction saturation_fraction of dissolved air, that is in
    equilibrium with ice at in-situ temperature t and pressure p.  If the input
    values are such that there is no positive value of Absolute Salinity for
    which seawater is frozen, the output, brineSA_t, is put equal to -99.

    Parameters
    ----------
    t : array_like
        in situ temperature [:math:`^\circ` C (ITS-90)]
    p : array_like
        sea pressure  [dbar]
    saturation_fraction : fraction between 0, 1.  The saturation fraction of
                          dissolved air in seawater.  Default is 0 or
                          completely saturated.

    Returns
    -------
    brine_SA_t : array_like
                 Absolute Salinity of seawater when it freezes [ g/kg ]

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
    UNESCO (English), 196 pp. See sections 3.33.

    Modifications:
    2011-28-03. Trevor McDougall and Paul Barker.

    """

    t, p, saturation_fraction = np.broadcast_arrays(t, p, saturation_fraction)

    if np.logical_or(saturation_fraction < 0, saturation_fraction > 1).any():
        raise ValueError('Saturation_fraction MUST be between zero and one.')

    p_r = p * 1e-4

    # Form the first estimate of brine_SA_t, called SA here, from a polynomial
    # in CT and p_r.
    SA = -(t + 9 * p_r) / 0.06  # A rough estimate to get the saturated CT.

    SA = np.maximum(SA, 0)

    CT = CT_from_t(SA, t, p)
    CTsat = CT - (1 - saturation_fraction) * 1e-3 * (2.4 - a * SA) * (1 + b *
            (1 - SA / SSO))

    SA = P[0] + p * (P[2] + P[4] * CTsat + p * (P[5] + CTsat * (P[7] +
         P[9] * CTsat) + p * (P[8] + CTsat * (P[10] + P[12] * CTsat) + p *
         (P[11] + P[13] * CTsat + P[14] * p)))) + CTsat * (P[1] + CTsat *
         (P[3] + P[6] * p))

    t_freezing_zero_SA = t_freezing(np.zeros_like(t), p, saturation_fraction)

    # Find t > t_freezing_zero_SA.  If this is the case, the input values
    # represent seawater that is not frozen (at any positive SA).
    Itw = (t > t_freezing_zero_SA)  # Itw stands for "I_too_warm"
    SA[Itw] = np.ma.masked

    # Find -SA_cut_off < SA < SA_cut_off, replace the above estimate of SA
    # with one based on (t_freezing_zero_SA - t).
    SA_cut_off = 2.5  # This is the band of SA within +- 2.5 g/kg of SA = 0,
                      # which we treat differently in calculating the initial
                      # values of both SA and dCT_dSA.

    Ico = (np.abs(SA) < SA_cut_off)
    Icoa = np.logical_and(SA < 0, SA >= -SA_cut_off)
    SA[Icoa] = 0

    # Find SA < -SA_cut_off, set them to masked.
    SA[SA < -SA_cut_off] = np.ma.masked

    # Form the first estimate of dt_dSA, the derivative of t with respect
    # to SA at fixed p, using the coefficients, t0 ... t22 from t_freezing.
    SA_r = 0.01 * SA
    x = np.sqrt(SA_r)
    dt_dSA_part = 2 * T[1] + x * (3 * T[2] + x * (4 * T[3] + x * (5 * T[4] +
    x * (6 * T[5] + 7 * T[6] * x)))) + p_r * (2 * T[10] + p_r * (2 * T[12] +
    p_r * (2 * T[15] + 4 * T[21] * x * x)) + x * x * (4 * T[13] + 4 * T[17] *
    p_r + 6 * T[19] * x * x) + x * (3 * T[11] + 3 * p_r * (T[14] + T[18] *
    p_r) + x * x * (5 * T[16] + 5 * T[20] * p_r + 7 * T[22] * x * x)))

    dt_dSA = 0.5 * 0.01 * dt_dSA_part + saturation_fraction * 1e-3 / 70.33008

    # Now replace the estimate of SA with the one based on
    # (t_freezing_zero_SA - t) when (abs(SA) < SA_cut_off).
    SA[Ico] = (t[Ico] - t_freezing_zero_SA[Ico]) / dt_dSA[Ico]

    # Begin the modified Newton-Raphson method to find the root of
    # t_freeze = t for SA.
    Number_of_Iterations = 5
    for I_iter in range(0, Number_of_Iterations):
        SA_old = SA
        t_freeze = t_freezing(SA_old, p, saturation_fraction)
        SA = SA_old - (t_freeze - t) / dt_dSA
        # Half-way point of the modified Newton-Raphson solution method.
        SA_r = 0.5 * 0.01 * (SA + SA_old)  # Mean value of SA and SA_old.
        x = np.sqrt(SA_r)
        dt_dSA_part = (2 * T[1] + x * (3 * T[2] + x * (4 * T[3] + x * (5 *
                       T[4] + x * (6 * T[5] + 7 * T[6] * x)))) + p_r *
                       (2 * T[10] + p_r * (2 * T[12] + p_r * (2 * T[15] + 4 *
                       T[21] * x * x)) + x * x * (4 * T[13] + 4 * T[17] * p_r +
                       6 * T[19] * x * x) + x * (3 * T[11] + 3 * p_r * (T[14] +
                       T[18] * p_r) + x * x * (5 * T[16] + 5 * T[20] * p_r +
                       7 * T[22] * x * x))))

        dt_dSA = (0.5 * 0.01 * dt_dSA_part + saturation_fraction * 1e-3 /
                  70.33008)

        SA = SA_old - (t_freeze - t) / dt_dSA

    """The following lines of code, if implemented, calculate the error of this
    function in terms of in-situ temperature.  With Number_of_Iterations = 4,
    the max error in t is 3x10^-13 C.  With Number_of_Iterations = 5, the max
    error in t is 2x10^-14 C, which is the machine precision of the computer.
    Number_of_Iterations = 5 is what we recommend.

    SA[SA < 0] = np.ma.masked

    t_freeze = t_freezing(SA, p, saturation_fraction)
    t_error = np.abs(t_freeze - t)
    tmp = np.logical_or(p > 10000, SA > 120)
    out = np.logical_and(tmp, p + SA * 71.428571428571402 > 13571.42857142857)
    t_error[out] = np.ma.masked
    """

    brine_SA_t = SA
    tmp = np.logical_or(p > 10000, SA > 120)
    out = np.logical_and(tmp, p + SA * 71.428571428571402 > 13571.42857142857)
    brine_SA_t[out] = np.ma.masked

    brine_SA_t[Itw] = -99  # If the t input is too warm, then there is no
                           # (positive) value of SA that represents frozen
                           # seawater.

    return brine_SA_t
