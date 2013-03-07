# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np

from library import Hill_ratio_at_SP2
from gsw.utilities import match_args_return

__all__ = [
           'SP_from_C',
           'C_from_SP',
           'SP_from_R',
           'R_from_SP',
           'SP_salinometer',
           'SP_from_SK',
           'SK_from_SP'
           ]

# Constants:
a = (0.0080, -0.1692, 25.3851, 14.0941, -7.0261, 2.7081)

b = (0.0005, -0.0056, -0.0066, -0.0375, 0.0636, -0.0144)

c = (0.6766097, 2.00564e-2, 1.104259e-4, -6.9698e-7, 1.0031e-9)

d = (3.426e-2, 4.464e-4, 4.215e-1, -3.107e-3)

e = (2.070e-5, -6.370e-10, 3.989e-15)

P = (4.577801212923119e-3, 1.924049429136640e-1, 2.183871685127932e-5,
    -7.292156330457999e-3, 1.568129536470258e-4, -1.478995271680869e-6,
    9.086442524716395e-4, -1.949560839540487e-5, -3.223058111118377e-6,
    1.175871639741131e-7, -7.522895856600089e-5, -2.254458513439107e-6,
    6.179992190192848e-7, 1.005054226996868e-8, -1.923745566122602e-9,
    2.259550611212616e-6, 1.631749165091437e-7, -5.931857989915256e-9,
    -4.693392029005252e-9, 2.571854839274148e-10, 4.198786822861038e-12)

q = (5.540896868127855e-5, 2.015419291097848e-1, -1.445310045430192e-5,
    -1.567047628411722e-2, 2.464756294660119e-4, -2.575458304732166e-7,
    5.071449842454419e-3, -9.081985795339206e-5, -3.635420818812898e-6,
    2.249490528450555e-8, -1.143810377431888e-3, 2.066112484281530e-5,
    7.482907137737503e-7, 4.019321577844724e-8, -5.755568141370501e-10,
    1.120748754429459e-4, -2.420274029674485e-6, -4.774829347564670e-8,
    -4.279037686797859e-9, -2.045829202713288e-10, 5.025109163112005e-12)

r = (3.432285006604888e-3, 1.672940491817403e-1, 2.640304401023995e-5,
    1.082267090441036e-1, -6.296778883666940e-5, -4.542775152303671e-7,
    -1.859711038699727e-1, 7.659006320303959e-4, -4.794661268817618e-7,
    8.093368602891911e-9, 1.001140606840692e-1, -1.038712945546608e-3,
    -6.227915160991074e-6, 2.798564479737090e-8, -1.343623657549961e-10,
    1.024345179842964e-2, 4.981135430579384e-4, 4.466087528793912e-6,
    1.960872795577774e-8, -2.723159418888634e-10, 1.122200786423241e-12)

u = (5.180529787390576e-3, 1.052097167201052e-3, 3.666193708310848e-5,
    7.112223828976632, -3.631366777096209e-4, -7.336295318742821e-7,
    -1.576886793288888e+2, -1.840239113483083e-3, 8.624279120240952e-6,
    1.233529799729501e-8, 1.826482800939545e+3, 1.633903983457674e-1,
    -9.201096427222349e-5, -9.187900959754842e-8, -1.442010369809705e-10,
    -8.542357182595853e+3, -1.408635241899082, 1.660164829963661e-4,
    6.797409608973845e-7, 3.345074990451475e-10, 8.285687652694768e-13)

k = 0.0162

a, b, c, d, e, P, q, r, u, k = map(np.asarray, (a, b, c, d, e, P, q, r, u, k))


@match_args_return
def SP_from_C(C, t, p):
    r"""Calculates Practical Salinity, SP, from conductivity, C, primarily
    using the PSS-78 algorithm.  Note that the PSS-78 algorithm for Practical
    Salinity is only valid in the range 2 < SP < 42.  If the PSS-78 algorithm
    produces a Practical Salinity that is less than 2 then the Practical
    Salinity is recalculated with a modified form of the Hill et al. (1986)
    formula. The modification of the Hill et al. (1986) expression is to ensure
    that it is exactly consistent with PSS-78 at SP = 2.  Note that the input
    values of conductivity need to be in units of mS/cm (not S/m).

    Parameters
    ----------
    C : array
        conductivity [mS cm :sup:`-1`]
    t : array
        in-situ temperature [:math:`^\circ` C (ITS-90)]
    p : array
        sea pressure [dbar]
        (i.e. absolute pressure - 10.1325 dbar)

    Returns
    -------
    SP : array
         Practical Salinity [psu (PSS-78), unitless]

    Examples
    --------
    TODO

    See Also
    --------
    TODO

    Notes
    -----
    TODO

    References
    ----------
    .. [1] Culkin and Smith, 1980:  Determination of the Concentration of
    Potassium Chloride Solution Having the Same Electrical Conductivity, at
    15C and Infinite Frequency, as Standard Seawater of Salinity 35.0000
    (Chlorinity 19.37394), IEEE J. Oceanic Eng, 5, 22-23.

    .. [2] Hill, K.D., T.M. Dauphinee & D.J. Woods, 1986: The extension of the
    Practical Salinity Scale 1978 to low salinities. IEEE J. Oceanic Eng., 11,
    109 - 112.

    .. [3] IOC, SCOR and IAPSO, 2010: The international thermodynamic equation
    of seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Appendix E.

    .. [4] Unesco, 1983: Algorithms for computation of fundamental properties
    of seawater.  Unesco Technical Papers in Marine Science, 44, 53 pp.

    Modifications:
    2011-04-01. Paul Barker, Trevor McDougall and Rich Pawlowicz.
    """

    C, t, p = np.broadcast_arrays(C, t, p)

    t68 = t * 1.00024
    ft68 = (t68 - 15) / (1 + k * (t68 - 15))

    # The dimensionless conductivity ratio, R, is the conductivity input, C,
    # divided by the present estimate of C(SP=35, t_68=15, p=0) which is
    # 42.9140 mS/cm (=4.29140 S/m), (Culkin and Smith, 1980).

    R = 0.023302418791070513 * C  # 0.023302418791070513 = 1./42.9140

    # rt_lc corresponds to rt as defined in the UNESCO 44 (1983) routines.
    rt_lc = c[0] + (c[1] + (c[2] + (c[3] + c[4] * t68) * t68) * t68) * t68
    Rp = (1 + (p * (e[0] + e[1] * p + e[2] * p ** 2)) /
         (1 + d[0] * t68 + d[1] * t68 ** 2 + (d[2] + d[3] * t68) * R))
    Rt = R / (Rp * rt_lc)

    Rt[Rt < 0] = np.nan
    Rtx = np.sqrt(Rt)

    SP = a[0] + (a[1] + (a[2] + (a[3] + (a[4] + a[5] * Rtx) * Rtx) * Rtx) *
         Rtx) * Rtx + ft68 * (b[0] + (b[1] + (b[2] + (b[3] + (b[4] + b[5] *
         Rtx) * Rtx) * Rtx) * Rtx) * Rtx)

    # The following section of the code is designed for SP < 2 based on the
    # Hill et al. (1986) algorithm.  This algorithm is adjusted so that it is
    # exactly equal to the PSS-78 algorithm at SP = 2.

    I2, = np.nonzero(np.ravel(SP) < 2)
    if len(I2) > 0:
        Hill_ratio = Hill_ratio_at_SP2(t[I2])
        x = 400 * Rt[I2]
        sqrty = 10 * Rtx[I2]
        part1 = 1 + x * (1.5 + x)
        part2 = 1 + sqrty * (1 + sqrty * (1 + sqrty))
        SP_Hill_raw = SP[I2] - a[0] / part1 - b[0] * ft68[I2] / part2
        SP[I2] = Hill_ratio * SP_Hill_raw

    SP = np.maximum(SP, 0)  # Ensure that SP is non-negative.

    return SP


def C_from_SP(SP, t, p):
    r"""Calculates conductivity, C, from (SP, t, p) using PSS-78 in the range
    2 < SP < 42. If the input Practical Salinity is less than 2 then a modified
    form of the Hill et al. (1986) fomula is used for Practical Salinity. The
    modification of the Hill et al. (1986) expression is to ensure that it is
    exactly consistent with PSS-78 at SP = 2.

    The conductivity ratio returned by this function is consistent with the
    input value of Practical Salinity, SP, to 2x10^-14 psu over the full range
    of input parameters (from pure fresh water up to SP = 42 psu). This error
    of 2x10^-14 psu is machine precision at typical seawater salinities.  This
    accuracy is achieved by having four different polynomials for the starting
    value of Rtx (the square root of Rt) in four different ranges of SP, and by
    using one and a half iterations of a computationally efficient modified
    Newton-Raphson technique to find the root of the equation.

    Parameters
    ----------
    SP : array
         Practical Salinity [psu (PSS-78), unitless]
    t : array
        in-situ temperature [:math:`^\circ` C (ITS-90)]
    p : array
        sea pressure [dbar]
        (i.e. absolute pressure - 10.1325 dbar)

    Returns
    -------
    C : array
        conductivity [mS cm :sup:`-1`]

    See Also
    --------
    TODO

    Notes
    -----
    Note that strictly speaking PSS-78 (Unesco, 1983) defines Practical
    Salinity in terms of the conductivity ratio, R, without actually
    specifying the value of C(35,15,0) (which we currently take to be
    42.9140 mS/cm).

    Examples
    --------
    TODO

    References
    ----------
    .. [1] Hill, K.D., T.M. Dauphinee and D.J. Woods, 1986: The extension of
    the Practical Salinity Scale 1978 to low salinities. IEEE J. Oceanic Eng.,
    OE-11, 1, 109 - 112.

    .. [2] IOC, SCOR and IAPSO, 2010: The international thermodynamic equation
    of seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp. See appendix E.

    .. [3] Unesco, 1983: Algorithms for computation of fundamental properties
    of seawater.  Unesco Technical Papers in Marine Science, 44, 53 pp.
    """

    C = 42.9140 * R_from_SP(SP, t, p)

    return C


@match_args_return
def SP_from_R(R, t, p):
    r"""Calculates Practical Salinity, SP, from the conductivity ratio, R,
    primarily using the PSS-78 algorithm.  Note that the PSS-78 algorithm for
    Practical Salinity is only valid in the range 2 < SP < 42.  If the PSS-78
    algorithm produces a Practical Salinity that is less than 2 then the
    Practical Salinity is recalculated with a modified form of the Hill et al.
    (1986) formula.  The modification of the Hill et al. (1986) expression are
    to ensure that it is exactly consistent with PSS-78 at SP = 2.

    Parameters
    ----------
    R : array_like
        conductivity ratio [unitless]
    t : array_like
        in-situ temperature [:math:`^\circ` C (ITS-90)]
    p : array
        sea pressure [dbar]
        (i.e. absolute pressure - 10.1325 dbar)

    Returns
    -------
    SP : array
         Practical Salinity [psu (PSS-78), unitless]

    Examples
    --------
    TODO

    See Also
    --------
    TODO

    Notes
    -----
    TODO

    References
    ----------
    .. [1] Culkin and Smith, 1980:  Determination of the Concentration of
    Potassium Chloride Solution Having the Same Electrical Conductivity, at
    15C and Infinite Frequency, as Standard Seawater of Salinity 35.0000
    (Chlorinity 19.37394), IEEE J. Oceanic Eng, 5, 22-23.

    .. [2] Hill, K.D., T.M. Dauphinee & D.J. Woods, 1986: The extension of the
    Practical Salinity Scale 1978 to low salinities. IEEE J. Oceanic Eng.,
    11, 109 - 112.

    .. [3] IOC, SCOR and IAPSO, 2010: The international thermodynamic equation
    of seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Appendix E.

    .. [4] Unesco, 1983: Algorithms for computation of fundamental properties
    of seawater.  Unesco Technical Papers in Marine Science, 44, 53 pp.

    Modifications:
    2011-04-01. Paul Barker, Trevor McDougall and Rich Pawlowicz.
    """

    R, t, p = np.broadcast_arrays(R, t, p)

    t68 = t * 1.00024
    ft68 = (t68 - 15) / (1 + k * (t68 - 15))

    # rt_lc corresponds to rt as defined in the UNESCO 44 (1983) routines.
    rt_lc = c[0] + (c[1] + (c[2] + (c[3] + c[4] * t68) * t68) * t68) * t68
    Rp = (1 + (p * (e[0] + e[1] * p + e[2] * p ** 2)) /
         (1 + d[0] * t68 + d[1] * t68 ** 2 + (d[2] + d[3] * t68) * R))
    Rt = R / (Rp * rt_lc)

    Rt[Rt < 0] = np.nan
    Rtx = np.sqrt(Rt)

    SP = a[0] + (a[1] + (a[2] + (a[3] + (a[4] + a[5] * Rtx) * Rtx) * Rtx) *
         Rtx) * Rtx + ft68 * (b[0] + (b[1] + (b[2] + (b[3] + (b[4] + b[5] *
         Rtx) * Rtx) * Rtx) * Rtx) * Rtx)

    # The following section of the code is designed for SP < 2 based on the
    # Hill et al. (1986) algorithm.  This algorithm is adjusted so that it is
    # exactly equal to the PSS-78 algorithm at SP = 2.

    I2 = SP < 2
    if I2.any():
        Hill_ratio = Hill_ratio_at_SP2(t[I2])
        x = 400 * Rt[I2]
        sqrty = 10 * Rtx[I2]
        part1 = 1 + x * (1.5 + x)
        part2 = 1 + sqrty * (1 + sqrty * (1 + sqrty))
        SP_Hill_raw = SP[I2] - a[0] / part1 - b[0] * ft68[I2] / part2
        SP[I2] = Hill_ratio * SP_Hill_raw

    SP = np.maximum(SP, 0)  # Ensure that SP is non-negative.

    return SP


@match_args_return
def R_from_SP(SP, t, p):
    r"""Calculates conductivity ratio from (SP, t, p) using PSS-78 in the range
    2 < SP < 42.  If the input Practical Salinity is less than 2 then a
    modified form of the Hill et al. (1986) formula is used for Practical
    Salinity.  The modification of the Hill et al. (1986) expression is to
    ensure that it is exactly consistent with PSS-78 at SP = 2.

    The conductivity ratio returned by this function is consistent with the
    input value of Practical Salinity, SP, to 2x10^-14 psu over the full range
    of input parameters (from pure fresh water up to SP = 42 psu).  This error
    of 2x10^-14 psu is machine precision at typical seawater salinities.  This
    accuracy is achieved by having four different polynomials for the starting
    value of Rtx (the square root of Rt) in four different ranges of SP, and by
    using one and a half iterations of a computationally efficient modified
    Newton-Raphson technique to find the root of the equation.

    Parameters
    ----------
    SP : array
         Practical Salinity [psu (PSS-78), unitless]
    t : array_like
        in-situ temperature [:math:`^\circ` C (ITS-90)]
    p : array
        sea pressure [dbar]
        (i.e. absolute pressure - 10.1325 dbar)

    Returns
    -------
    R : array_like
        conductivity ratio [unitless]

    Examples
    --------
    TODO

    See Also
    --------
    TODO

    Notes
    -----
    Strictly speaking PSS-78 (Unesco, 1983) defines Practical Salinity in terms
    of the conductivity ratio, R, without actually specifying the value of
    C(35, 15, 0) (which we currently take to be 42.9140 mS cm^-1.
    Culkin and Smith, 1980).

    References
    ----------
    .. [1] Culkin and Smith, 1980:  Determination of the Concentration of
    Potassium Chloride Solution Having the Same Electrical Conductivity, at
    15C and Infinite Frequency, as Standard Seawater of Salinity 35.0000
    (Chlorinity 19.37394), IEEE J. Oceanic Eng, 5, 22-23.

    .. [2] Hill, K.D., T.M. Dauphinee & D.J. Woods, 1986: The extension of the
    Practical Salinity Scale 1978 to low salinities. IEEE J. Oceanic Eng.,
    11, 109 - 112.

    .. [3] IOC, SCOR and IAPSO, 2010: The international thermodynamic equation
    of seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Appendix E.

    .. [4] Unesco, 1983: Algorithms for computation of fundamental properties
    of seawater.  Unesco Technical Papers in Marine Science, 44, 53 pp.

    Modifications:
    2011-04-06. Paul Barker, Trevor McDougall and Rich Pawlowicz.
    """

    # These few lines ensure that SP is non-negative.
    if (SP < 0).any():
        raise ValueError('R_from_SP: SP must be non-negative!')

    SP, t, p = np.broadcast_arrays(SP, t, p)

    # Setting up the constants
    t68 = t * 1.00024
    ft68 = (t68 - 15) / (1 + k * (t68 - 15))

    x = np.sqrt(SP)
    Rtx = np.zeros_like(SP) * np.nan

    # Finding the starting value of Rtx, the square root of Rt, using four
    # different polynomials of SP and t68.
    # TODO: Test case that cover all those "ifs"
    I = SP >= 9
    if I.any():
        Rtx[I] = P[0] + x[I] * (P[1] + P[4] * t68[I] + x[I] * (P[3] + P[7] *
        t68[I] + x[I] * (P[6] + P[11] * t68[I] + x[I] * (P[10] + P[16] *
        t68[I] + x[I] * P[15])))) + t68[I] * (P[2] + t68[I] * (P[5] + x[I] *
        x[I] * (P[12] + x[I] * P[17]) + P[8] * x[I] + t68[I] * (P[9] + x[I] *
        (P[13] + x[I] * P[18]) + t68[I] * (P[14] + P[19] * x[I] + P[20] *
        t68[I]))))

    I = np.logical_and(SP >= 0.25, SP < 9)
    if I.any():
        Rtx[I] = q[0] + x[I] * (q[1] + q[4] * t68[I] + x[I] * (q[3] + q[7] *
        t68[I] + x[I] * (q[6] + q[11] * t68[I] + x[I] * (q[10] + q[16] *
        t68[I] + x[I] * q[15])))) + t68[I] * (q[2] + t68[I] * (q[5] + x[I] *
        x[I] * (q[12] + x[I] * q[17]) + q[8] * x[I] + t68[I] * (q[9] + x[I] *
        (q[13] + x[I] * q[18]) + t68[I] * (q[14] + q[19] * x[I] + q[20] *
        t68[I]))))

    I = np.logical_and(SP >= 0.003, SP < 0.25)
    if I.any():
        Rtx[I] = r[0] + x[I] * (r[1] + r[4] * t68[I] + x[I] * (r[3] + r[7] *
        t68[I] + x[I] * (r[6] + r[11] * t68[I] + x[I] * (r[10] + r[16] *
        t68[I] + x[I] * r[15])))) + t68[I] * (r[2] + t68[I] * (r[5] + x[I] *
        x[I] * (r[12] + x[I] * r[17]) + r[8] * x[I] + t68[I] * (r[9] + x[I] *
        (r[13] + x[I] * r[18]) + t68[I] * (r[14] + r[19] * x[I] + r[20] *
        t68[I]))))

    I = SP < 0.003
    if I.any():
        Rtx[I] = u[0] + x[I] * (u[1] + u[4] * t68[I] + x[I] * (u[3] + u[7] *
        t68[I] + x[I] * (u[6] + u[11] * t68[I] + x[I] * (u[10] + u[16] *
        t68[I] + x[I] * u[15])))) + t68[I] * (u[2] + t68[I] * (u[5] + x[I] *
        x[I] * (u[12] + x[I] * u[17]) + u[8] * x[I] + t68[I] * (u[9] + x[I] *
        (u[13] + x[I] * u[18]) + t68[I] * (u[14] + u[19] * x[I] + u[20] *
        t68[I]))))

    # Finding the starting value of dSP_dRtx, the derivative of SP with
    # respect to Rtx.
    dSP_dRtx = a[1] + (2 * a[2] + (3 * a[3] + (4 * a[4] + 5 * a[5] * Rtx) *
                Rtx) * Rtx) * Rtx + ft68 * (b[1] + (2 * b[2] + (3 * b[3] +
                (4 * b[4] + 5 * b[5] * Rtx) * Rtx) * Rtx) * Rtx)

    # TODO: Test case that cover all those "ifs"
    I2 = SP < 2
    if I2.any():
        x = 400 * (Rtx[I2] ** 2)
        sqrty = 10 * Rtx[I2]
        part1 = 1 + x * (1.5 + x)
        part2 = 1 + sqrty * (1 + sqrty * (1 + sqrty))
        Hill_ratio = Hill_ratio_at_SP2(t[I2])
        dSP_dRtx[I2] = (dSP_dRtx[I2] + a[0] * 800 * Rtx[I2] * (1.5 + 2 * x) /
                       (part1 ** 2) + b[0] * ft68[I2] * (10 + sqrty * (20 +
                       30 * sqrty)) / (part2 ** 2))

        dSP_dRtx[I2] = Hill_ratio * dSP_dRtx[I2]

    """One iteration through the modified Newton-Raphson method achieves an
    error in Practical Salinity of about 10^-12 for all combinations of the
    inputs.  One and a half iterations of the modified Newton-Raphson method
    achieves a maximum error in terms of Practical Salinity of better than
    2x10^-14 everywhere.

    We recommend one and a half iterations of the modified Newton-Raphson
    method."""

    # Begin the modified Newton-Raphson method.
    SP_est = (a[0] + (a[1] + (a[2] + (a[3] + (a[4] + a[5] * Rtx) * Rtx) *
              Rtx) * Rtx) * Rtx + ft68 * (b[0] + (b[1] + (b[2] + (b[3] +
              (b[4] + b[5] * Rtx) * Rtx) * Rtx) * Rtx) * Rtx))

    # TODO: Test case that cover all those "ifs"
    I2 = SP_est < 2
    if I2.any():
        x = 400 * (Rtx[I2] ** 2)
        sqrty = 10 * Rtx[I2]
        part1 = 1 + x * (1.5 + x)
        part2 = 1 + sqrty * (1 + sqrty * (1 + sqrty))
        SP_Hill_raw = SP_est[I2] - a[0] / part1 - b[0] * ft68[I2] / part2
        Hill_ratio = Hill_ratio_at_SP2(t[I2])
        SP_est[I2] = Hill_ratio * SP_Hill_raw

    Rtx_old = Rtx
    Rtx = Rtx_old - (SP_est - SP) / dSP_dRtx

    # This mean value of Rtx, Rtxm, is the value of Rtx at which the
    # derivative dSP_dRtx is evaluated.
    Rtxm = 0.5 * (Rtx + Rtx_old)

    dSP_dRtx = a[1] + (2 * a[2] + (3 * a[3] + (4 * a[4] + 5 * a[5] *
               Rtxm) * Rtxm) * Rtxm) * Rtxm + ft68 * (b[1] + (2 * b[2] +
               (3 * b[3] + (4 * b[4] + 5 * b[5] * Rtxm) * Rtxm) * Rtxm) * Rtxm)

    # TODO: Test case that cover all those "ifs"
    I2 = SP_est < 2
    if I2.any():
        x = 400 * (Rtxm[I2] ** 2)
        sqrty = 10 * Rtxm[I2]
        part1 = 1 + x * (1.5 + x)
        part2 = 1 + sqrty * (1 + sqrty * (1 + sqrty))
        dSP_dRtx[I2] = (dSP_dRtx[I2] + a[0] * 800 * Rtxm[I2] * (1.5 + 2 *
                        x) / (part1 ** 2) + b[0] * ft68[I2] * (10 + sqrty *
                        (20 + 30 * sqrty)) / (part2 ** 2))
        Hill_ratio = Hill_ratio_at_SP2(t[I2])
        dSP_dRtx[I2] = Hill_ratio * dSP_dRtx[I2]

    # End of the one full iteration of the modified Newton-Raphson technique.
    Rtx = Rtx_old - (SP_est - SP) / dSP_dRtx  # Updated Rtx

    #  Now we do another half iteration of the modified Newton-Raphson
    #  technique, making a total of one and a half modified N-R iterations.

    SP_est = a[0] + (a[1] + (a[2] + (a[3] + (a[4] + a[5] * Rtx) * Rtx) *
             Rtx) * Rtx) * Rtx + ft68 * (b[0] + (b[1] + (b[2] + (b[3] +
             (b[4] + b[5] * Rtx) * Rtx) * Rtx) * Rtx) * Rtx)

    # TODO: Test case that cover all those "ifs"
    I2 = SP_est < 2
    if I2.any():
        x = 400 * (Rtx[I2] ** 2)
        sqrty = 10 * Rtx[I2]
        part1 = 1 + x * (1.5 + x)
        part2 = 1 + sqrty * (1 + sqrty * (1 + sqrty))
        SP_Hill_raw = SP_est[I2] - a[0] / part1 - b[0] * ft68[I2] / part2
        Hill_ratio = Hill_ratio_at_SP2(t[I2])
        SP_est[I2] = Hill_ratio * SP_Hill_raw

    Rtx = Rtx - (SP_est - SP) / dSP_dRtx

    """ TODO: add this as a kw.
    Return the error, SP_error, in Rtx (in terms of psu).

    SP_est = (a[0] + (a[1] + (a[2] + (a[3] + (a[4] + a[5] * Rtx) * Rtx) *
            Rtx) * Rtx) * Rtx + ft68 * (b[0] + (b[1] + (b[2] + (b[3] +
            (b[4] + b[5] * Rtx) * Rtx) * Rtx) * Rtx) * Rtx))
    I2 = SP_est < 2
    if I2.any():
        x = 400 * (Rtx[I2] ** 2)
        sqrty = 10 * Rtx[I2]
        part1 = 1 + x * (1.5 + x)
        part2 = 1 + sqrty * (1 + sqrty * (1 + sqrty))
        SP_Hill_raw = SP_est[I2] - a[0] / part1 - b[0] * ft68[I2] / part2
        Hill_ratio = Hill_ratio_at_SP2(t[I2])
        SP_est[I2] = Hill_ratio * SP_Hill_raw

    SP_error = np.abs(SP - SP_est)

    This is the end of the error testing
    """

    # Now go from Rtx to Rt and then to the conductivity ratio R at pressure p.
    Rt = Rtx ** 2
    A = d[2] + d[3] * t68
    B = 1 + d[0] * t68 + d[1] * t68 ** 2
    C = p * (e[0] + e[1] * p + e[2] * p ** 2)
    # rt_lc (i.e. rt_lower_case) corresponds to rt as defined in the
    # UNESCO 44 (1983) routines.
    rt_lc = c[0] + (c[1] + (c[2] + (c[3] + c[4] * t68) * t68) * t68) * t68

    D = B - A * rt_lc * Rt
    E = rt_lc * Rt * A * (B + C)
    Ra = np.sqrt(D ** 2 + 4 * E) - D

    return 0.5 * Ra / A


@match_args_return
def SP_salinometer(Rt, t):
    r"""Calculates Practical Salinity SP from a salinometer, primarily using
    the PSS-78 algorithm.  Note that the PSS-78 algorithm for Practical
    Salinity is only valid in the range 2 < SP < 42.  If the PSS-78 algorithm
    produces a Practical Salinity that is less than 2 then the Practical
    Salinity is recalculated with a modified form of the Hill et al. (1986)
    formula. The modification of the Hill et al. (1986) expression is to
    ensure that it is exactly consistent with PSS-78 at SP = 2.

    A laboratory salinometer has the ratio of conductivities, Rt, as an output,
    and the present function uses this conductivity ratio and the temperature t
    of the salinometer bath as the two input variables.

    Parameters
    ----------
    Rt : array
         C(SP,t_68,0)/C(SP=35,t_68,0) [unitless]
         conductivity ratio
         :math:`R = \frac{C(S, t_68, 0)}{C(35, 15(IPTS-68),0)} [unitless]

    t : array
        Temperature of the bath of the salinometer [:math:`^\circ` C (ITS-90)]

    Returns
    -------
    SP : array
         Practical Salinity [psu (PSS-78), unitless]

    See Also
    --------
    TODO: sw.sals

    Notes
    -----
    TODO

    Examples
    --------
    TODO

    References
    -----------
    ..[1] Fofonoff, P. and R.C. Millard Jr. 1983: Algorithms for computation of
    fundamental properties of seawater.  Unesco Tech. Pap. in Mar. Sci., 44,
    53 pp.

    ..[2] Hill, K.D., T.M. Dauphinee & D.J. Woods, 1986: The extension of the
    Practical Salinity Scale 1978 to low salinities. IEEE J. Oceanic Eng., 11,
    109 - 112.

    .. [3] IOC, SCOR and IAPSO, 2010: The international thermodynamic equation
    of seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp. See appendix E of this TEOS-10 Manual, and in
    particular, Eqns. (E.2.1) and (E.2.6).

    Modifications:
    2011-04-30. Paul Barker, Trevor McDougall and Rich Pawlowicz. Version 3.0
    """

    Rt, t = np.broadcast_arrays(Rt, t)

    t68 = t * 1.00024
    ft68 = (t68 - 15) / (1 + k * (t68 - 15))

    Rt[Rt < 0] = np.NaN
    Rtx = np.sqrt(Rt)

    SP = a[0] + (a[1] + (a[2] + (a[3] + (a[4] + a[5] * Rtx) * Rtx) * Rtx) *
         Rtx) * Rtx + ft68 * (b[0] + (b[1] + (b[2] + (b[3] + (b[4] + b[5] *
         Rtx) * Rtx) * Rtx) * Rtx) * Rtx)

    """The following section of the code is designed for SP < 2 based on the
    Hill et al. (1986) algorithm.  This algorithm is adjusted so that it is
    exactly equal to the PSS-78 algorithm at SP = 2."""

    I2 = SP < 2
    if I2.any():
        Hill_ratio = Hill_ratio_at_SP2(t[I2])
        x = 400 * Rt[I2]
        sqrty = 10 * Rtx[I2]
        part1 = 1 + x * (1.5 + x)
        part2 = 1 + sqrty * (1 + sqrty * (1 + sqrty))
        SP_Hill_raw = SP[I2] - a[0] / part1 - b[0] * ft68[I2] / part2
        SP[I2] = Hill_ratio * SP_Hill_raw
    # Ensure that SP is non-negative.
    SP = np.maximum(SP, 0)
    return SP


@match_args_return
def SP_from_SK(SK):
    r"""Calculates Practical Salinity from Knudsen Salinity.

    Parameters
    ----------
    SK : array_like
         Knudsen Salinity [parts per thousand, ppt]

    Returns
    -------
    SP : array
         Practical Salinity [psu (PSS-78), unitless]

    Examples
    --------
    TODO

    See Also
    --------
    TODO

    Notes
    -----
    TODO

    References
    ----------
    .. [1] IOC, SCOR and IAPSO, 2010: The international thermodynamic equation
    of seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  See Appendix A.3.

    Modifications:
    2011-11-16. Trevor McDougall and Paul Barker.
    """

    SP = (SK - 0.03) * (1.80655 / 1.805)
    return np.maximum(SP, 0)  # Ensure that SP is non-negative.


@match_args_return
def SK_from_SP(SP):
    r"""Calculates Knudsen Salinity from Practical Salinity.

    Parameters
    ----------
    SP : array
         Practical Salinity [psu (PSS-78), unitless]

    Returns
    -------
    SK : array_like
         Knudsen Salinity [parts per thousand, ppt]

    Examples
    --------
    TODO

    See Also
    --------
    TODO

    Notes
    -----
    TODO

    References
    ----------
    .. [1] IOC, SCOR and IAPSO, 2010: The international thermodynamic equation
    of seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  See Appendix A.3.

    Modifications:
    2011-11-16. Trevor McDougall and Paul Barker.
    """
    SP = np.maximum(SP, 0)  # Ensure that SP is non-negative.

    return 0.03 + SP * (1.805 / 1.80655)

if __name__ == '__main__':
    import doctest
    doctest.testmod()
