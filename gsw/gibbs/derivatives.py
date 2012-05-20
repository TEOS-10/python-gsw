# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np

from library import gibbs
from constants import cp0, Kelvin
from gsw.utilities import match_args_return
from conversions import pt_from_CT, pt_from_t

__all__ = [
           'CT_first_derivatives',
           'CT_second_derivatives',
           'enthalpy_first_derivatives',
           'enthalpy_second_derivatives',
           'entropy_first_derivatives',
           'entropy_second_derivatives',
           'pt_first_derivatives',
           'pt_second_derivatives',
          ]


@match_args_return
def pt_first_derivatives(SA, CT):
    r"""Calculates the following two partial derivatives of potential
    temperature (the regular potential temperature whose reference sea
    pressure is 0 dbar)
    (1) pt_SA, the derivative with respect to Absolute Salinity at
        constant Conservative Temperature, and
    (2) pt_CT, the derivative with respect to Conservative Temperature at
        constant Absolute Salinity.

    Parameters
    ----------
    SA : array_like
         Absolute salinity [g kg :sup:`-1`]
    CT : array_like
         Conservative Temperature [:math:`^\circ` C (TEOS-10)]

    Returns
    -------
    pt_SA : array_like
            The derivative of potential temperature with respect to Absolute
            Salinity at constant Conservative Temperature. [K/(g/kg)]
    pt_CT : array_like
            The derivative of potential temperature with respect to
            Conservative Temperature at constant Absolute Salinity. [unitless]

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
    of seawater -  2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp. See Eqns. (A.12.6), (A.12.3), (P.6) and (P.8).

    .. [2] McDougall T.J., P.M. Barker, R. Feistel and D.R. Jackett, 2011: A
    computationally efficient 48-term expression for the density of seawater in
    terms of Conservative Temperature, and related properties of seawater.  To
    be submitted to Ocean Science Discussions.

    Modifications:
    2011-03-29. Trevor McDougall and Paul Barker.
    """

    n0, n1, n2 = 0, 1, 2
    pt = pt_from_CT(SA, CT)
    abs_pt = Kelvin + pt
    CT_pt = - (abs_pt * gibbs(n0, n2, n0, SA, pt, 0)) / cp0

    def pt_derivative_SA(SA, CT):
        CT_SA = (gibbs(n1, n0, n0, SA, pt, 0) - abs_pt *
                                           gibbs(n1, n1, n0, SA, pt, 0)) / cp0
        return - CT_SA / CT_pt

    def pt_derivative_CT(SA, CT):
        return 1.0 / CT_pt

    return (pt_derivative_SA(SA, CT),
            pt_derivative_CT(SA, CT))


@match_args_return
def pt_second_derivatives(SA, CT):
    r"""Calculates the following three second-order derivatives of potential
    temperature (the regular potential temperature which has a reference
    sea pressure of 0 dbar),
    (1) pt_SA_SA, the second derivative with respect to Absolute Salinity at
        constant Conservative Temperature,
    (2) pt_SA_CT, the derivative with respect to Conservative Temperature and
        Absolute Salinity, and
    (3) pt_CT_CT, the second derivative with respect to Conservative
        Temperature at constant Absolute Salinity.

    Parameters
    ----------
    SA : array_like
         Absolute salinity [g kg :sup:`-1`]
    CT : array_like
         Conservative Temperature [:math:`^\circ` C (TEOS-10)]

    Returns
    -------
    pt_SA_SA : array_like
               The second derivative of potential temperature (the regular
               potential temperature which has reference sea pressure of 0
               dbar) with respect to Absolute Salinity at constant Conservative
               Temperature. [K/((g/kg)^2)]
    pt_SA_CT : array_like
               The derivative of potential temperature with respect to Absolute
               Salinity and Conservative Temperature. [1/(g/kg)]
    pt_CT_CT : array_like
               The second derivative of potential temperature (the regular one
               with p_ref = 0 dbar) with respect to Conservative Temperature at
               constant SA. [1/K]

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
    of seawater -  2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp. See Eqns. (A.12.9) and (A.12.10).

    .. [2] McDougall T.J., P.M. Barker, R. Feistel and D.R. Jackett, 2011: A
    computationally efficient 48-term expression for the density of seawater in
    terms of Conservative Temperature, and related properties of seawater.  To
    be submitted to Ocean Science Discussions.

    Modifications:
    2011-03-29. Trevor McDougall and Paul Barker.
    """

    def pt_derivative_SA_SA(SA, CT):
        dSA = 1e-3
        SA_l = SA - dSA
        SA_l = SA_l.clip(0.0, np.inf)
        SA_u = SA + dSA
        pt_SA_l, pt_CT_l = pt_first_derivatives(SA_l, CT)
        pt_SA_u, pt_CT_u = pt_first_derivatives(SA_u, CT)
        return (pt_SA_u - pt_SA_l) / (SA_u - SA_l)

    dCT = 1e-2
    CT_l = CT - dCT
    CT_u = CT + dCT
    pt_SA_l, pt_CT_l = pt_first_derivatives(SA, CT_l)
    pt_SA_u, pt_CT_u = pt_first_derivatives(SA, CT_u)

    def pt_derivative_SA_CT(SA, CT):
        # Can calculate this either way;
        # pt_SA_CT = (pt_CT_u - pt_CT_l) / (SA_u - SA_l)
        return (pt_SA_u - pt_SA_l) / (CT_u - CT_l)

    def pt_derivative_CT_CT(SA, CT):
        return (pt_CT_u - pt_CT_l) / (CT_u - CT_l)

    return (pt_derivative_SA_SA(SA, CT),
            pt_derivative_SA_CT(SA, CT),
            pt_derivative_CT_CT(SA, CT))


@match_args_return
def CT_first_derivatives(SA, pt):
    r"""Calculates the following two derivatives of Conservative Temperature
    (1) CT_SA, the derivative with respect to Absolute Salinity at constant
        potential temperature (with pr = 0 dbar), and
    (2) CT_pt, the derivative with respect to potential temperature (the
        regular potential temperature which is referenced to 0 dbar) at
        constant Absolute Salinity.

    Parameters
    ----------
    SA : array_like
         Absolute salinity [g kg :sup:`-1`]
    pt : array_like
         potential temperature referenced to a sea pressure of zero dbar
         [:math:`^\circ` C (ITS-90)]

    Returns
    -------
    CT_SA : array_like
            The derivative of CT with respect to SA at constant potential
            temperature reference sea pressure of 0 dbar.
            [K (g kg :sup:`-1`) :sup:`-1`]

    CT_pt : array_like
            The derivative of CT with respect to pt at constant SA.
            [ unitless ]

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
    >>> gsw.CT_first_derivatives(SA, pt)
    array([[-0.04198109, -0.04155814, -0.03473921, -0.0187111 , -0.01407594,
            -0.01057172],
           [ 1.00281494,  1.00255482,  1.00164514,  1.00000377,  0.99971636,
             0.99947433]])

    References
    ----------
    .. [1] IOC, SCOR and IAPSO, 2010: The international thermodynamic equation
    of seawater -  2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp. See Eqns. (A.12.3) and (A.12.9a,b).

    .. [2] McDougall T. J., D. R. Jackett, P. M. Barker, C. Roberts-Thomson, R.
    Feistel and R. W. Hallberg, 2010:  A computationally efficient 25-term
    expression for the density of seawater in terms of Conservative
    Temperature, and related properties of seawater.

    Modifications:
    2010-08-05. Trevor McDougall and Paul Barker.
    """

    # FIXME: Matlab version 3.0 has a copy-and-paste of the gibbs function here
    # instead of a call. Why?

    n0, n1, n2 = 0, 1, 2
    abs_pt = Kelvin + pt

    g100 = gibbs(n1, n0, n0, SA, pt, 0)
    g110 = gibbs(n1, n1, n0, SA, pt, 0)
    g020 = gibbs(n0, n2, n0, SA, pt, 0)

    def CT_derivative_SA(SA, pt):
        return (g100 - abs_pt * g110) / cp0

    def CT_derivative_pt(SA, pt):
        return - (abs_pt * g020) / cp0

    return CT_derivative_SA(SA, pt), CT_derivative_pt(SA, pt)


@match_args_return
def CT_second_derivatives(SA, pt):
    r"""Calculates the following three, second-order derivatives of
    Conservative Temperature
    (1) CT_SA_SA, the second derivative with respect to Absolute Salinity at
        constant potential temperature (with p_ref = 0 dbar),
    (2) CT_SA_pt, the derivative with respect to potential temperature (the
        regular potential temperature which is referenced to 0 dbar) and
        Absolute Salinity, and
    (3) CT_pt_pt, the second derivative with respect to potential temperature
        (the regular potential temperature which is referenced to 0 dbar) at
        constant Absolute Salinity.

    Parameters
    ----------
    SA : array_like
         Absolute salinity [g kg :sup:`-1`]
    pt : array_like
         potential temperature [:math:`^\circ` C (ITS-90)]

    Returns
    -------
    CT_SA_SA : array_like
               The second derivative of Conservative Temperature with respect
               to Absolute Salinity at constant potential temperature (the
               regular potential temperature which has reference sea pressure
               of 0 dbar). [K/((g/kg)^2)]
    CT_SA_pt : array_like
               The derivative of Conservative Temperature with respect to
               potential temperature (the regular one with p_ref = 0 dbar) and
               Absolute Salinity. [1/(g/kg)]
    CT_pt_pt : array_like
               The second derivative of Conservative Temperature with respect
               to potential temperature (the regular one with p_ref = 0 dbar)
               at constant SA. [1/K]

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
    of seawater -  2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp. See appendix A.12.

    .. [2] McDougall T.J., P.M. Barker, R. Feistel and D.R. Jackett, 2011: A
    computationally efficient 48-term expression for the density of seawater in
    terms of Conservative Temperature, and related properties of seawater.  To
    be submitted to Ocean Science Discussions.

    Modifications:
    2011-03-29. Trevor McDougall.
    """

    def CT_derivative_SA_SA(SA, pt):
        dSA = 1e-3
        SA_l = SA - dSA
        SA_l = SA_l.clip(0.0, np.inf)
        SA_u = SA + dSA

        CT_SA_l, _ = CT_first_derivatives(SA_l, pt)
        CT_SA_u, _ = CT_first_derivatives(SA_u, pt)

        return (CT_SA_u - CT_SA_l) / (SA_u - SA_l)

    def CT_derivative_SA_pt(SA, pt):
        dpt = 1e-2
        pt_l = pt - dpt
        pt_u = pt + dpt

        CT_SA_l, CT_pt_l = CT_first_derivatives(SA, pt_l)
        CT_SA_u, CT_pt_u = CT_first_derivatives(SA, pt_u)

        return (CT_SA_u - CT_SA_l) / (pt_u - pt_l)

    def CT_derivative_pt_pt(SA, pt):
        dpt = 1e-2
        pt_l = pt - dpt
        pt_u = pt + dpt

        CT_SA_l, CT_pt_l = CT_first_derivatives(SA, pt_l)
        CT_SA_u, CT_pt_u = CT_first_derivatives(SA, pt_u)

        return (CT_pt_u - CT_pt_l) / (pt_u - pt_l)

    return (CT_derivative_SA_SA(SA, pt),
            CT_derivative_SA_pt(SA, pt),
            CT_derivative_pt_pt(SA, pt))


@match_args_return
def entropy_first_derivatives(SA, CT):
    r"""Calculates the following two partial derivatives of specific entropy
    (eta)
    (1) eta_SA, the derivative with respect to Absolute Salinity at constant
        Conservative Temperature, and
    (2) eta_CT, the derivative with respect to Conservative Temperature at
        constant Absolute Salinity.

    Parameters
    ----------
    SA : array_like
         Absolute salinity [g kg :sup:`-1`]
    CT : array_like
         Conservative Temperature [:math:`^\circ` C (TEOS-10)]

    Returns
    -------
    eta_SA : array_like
             The derivative of specific entropy with respect to SA at constant
             CT [J g :sup:`-1` K :sup:`-1`]
    eta_CT : array_like
             The derivative of specific entropy with respect to CT at constant
             SA [ J (kg K :sup:`-2`) :sup:`-1` ]

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
    >>> gsw.entropy_first_derivatives(SA, CT)
    array([[ -0.2632868 ,  -0.26397728,  -0.2553675 ,  -0.23806659,
             -0.23443826,  -0.23282068],
           [ 13.22103121,  13.23691119,  13.48900463,  14.08659902,
             14.25772958,  14.38642995]])

    References
    ----------
    .. [1] IOC, SCOR and IAPSO, 2010: The international thermodynamic equation
    of seawater -  2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp. See Eqns. (A.12.8) and (P.14a,c).

    Modifications:
    2011-03-29. Trevor McDougall.
    """

    n0, n1 = 0, 1
    pt = pt_from_CT(SA, CT)

    def entropy_derivative_SA(SA, CT):
        return -(gibbs(n1, n0, n0, SA, pt, 0)) / (Kelvin + pt)

    def entropy_derivative_CT(SA, CT):
        return cp0 / (Kelvin + pt)

    return entropy_derivative_SA(SA, CT), entropy_derivative_CT(SA, CT)


@match_args_return
def entropy_second_derivatives(SA, CT):
    r"""Calculates the following three second-order partial derivatives of
    specific entropy (eta)
    (1) eta_SA_SA, the second derivative with respect to Absolute Salinity at
        constant Conservative Temperature, and
    (2) eta_SA_CT, the derivative with respect to Absolute Salinity and
        Conservative Temperature.
    (3) eta_CT_CT, the second derivative with respect to Conservative
        Temperature at constant Absolute Salinity.

    Parameters
    ----------
    SA : array_like
         Absolute salinity [g kg :sup:`-1`]
    CT : array_like
         Conservative Temperature [:math:`^\circ` C (TEOS-10)]

    Returns
    -------
    eta_SA_SA : array_like
                The second derivative of specific entropy with respect to SA at
                constant CT [J (kg K (g kg :sup:`-1` ) :sup:`2`) :sup:`-1`]
    eta_SA_CT : array_like
                The second derivative of specific entropy with respect to
                SA and CT [J (kg (g kg :sup:`-1` ) K :sup:`2`) :sup:`-1` ]
    eta_CT_CT : array_like
                The second derivative of specific entropy with respect to CT at
                constant SA [J (kg K :sup:`3`) :sup:`-1` ]

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
    >>> gsw.entropy_second_derivatives(SA, CT)
    TODO

    References
    ----------
    .. [1] IOC, SCOR and IAPSO, 2010: The international thermodynamic equation
    of seawater -  2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp. See Eqns. (P.14b) and (P.15a,b).

    Modifications:
    2011-03-29. Trevor McDougall and Paul Barker.
    """

    n0, n1, n2 = 0, 1, 2

    pt = pt_from_CT(SA, CT)
    abs_pt = Kelvin + pt

    def entropy_derivative_CT_CT(SA, CT):
        CT_pt = -(abs_pt * gibbs(n0, n2, n0, SA, pt, 0)) / cp0

        return - cp0 / (CT_pt * abs_pt ** 2)

    def entropy_derivative_SA_CT(SA, CT):
        CT_SA = (gibbs(n1, n0, n0, SA, pt, 0) -
                        (abs_pt * gibbs(n1, n1, n0, SA, pt, 0))) / cp0

        return -CT_SA * entropy_derivative_CT_CT(SA, CT)

    def entropy_derivative_SA_SA(SA, CT):
        CT_SA = (gibbs(n1, n0, n0, SA, pt, 0) -
                        (abs_pt * gibbs(n1, n1, n0, SA, pt, 0))) / cp0

        eta_SA_CT = entropy_derivative_SA_CT(SA, CT)

        return -gibbs(n2, n0, n0, SA, pt, 0) / abs_pt - CT_SA * eta_SA_CT

    return (entropy_derivative_SA_SA(SA, CT),
            entropy_derivative_SA_CT(SA, CT),
            entropy_derivative_CT_CT(SA, CT),)


@match_args_return
def enthalpy_first_derivatives(SA, CT, p):
    r"""Calculates the following three derivatives of specific enthalpy (h)
    (1) h_SA, the derivative with respect to Absolute Salinity at
        constant CT and p, and
    (2) h_CT, derivative with respect to CT at constant SA and p.
    (3) h_P, derivative with respect to pressure (in Pa) at constant SA and CT.

    Parameters
    ----------
    SA : array_like
         Absolute salinity [g kg :sup:`-1`]
    CT : array_like
         Conservative Temperature [:math:`^\circ` C (TEOS-10)]
    p : array_like
        pressure [dbar]

    Returns
    -------
    h_SA : array_like
           The first derivative of specific enthalpy with respect to Absolute
           Salinity at constant CT and p. [J/(kg (g/kg))]  i.e. [J/g]
    h_CT : array_like
           The first derivative of specific enthalpy with respect to CT at
           constant SA and p. [J/(kg K)]
    h_P : array_like
          The first partial derivative of specific enthalpy with respect to
          pressure (in Pa) at fixed SA and CT.  Note that h_P is specific
          volume (1/rho.)

    See Also
    --------
    TODO

    Notes
    -----
    TODO


    References
    ----------
    .. [1] IOC, SCOR and IAPSO, 2010: The international thermodynamic equation
    of seawater -  2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp. See Eqns. (A.11.18), (A.11.15) and (A.11.12.)

    Modifications:
    2010-09-24. Trevor McDougall.
    """

    # FIXME: The gsw 3.0 has the gibbs derivatives "copy-and-pasted" here
    # instead of the calls to the library! Why?

    n0, n1 = 0, 1
    pt0 = pt_from_CT(SA, CT)
    t = pt_from_t(SA, pt0, 0, p)
    temp_ratio = (Kelvin + t) / (Kelvin + pt0)

    def enthalpy_derivative_SA(SA, CT, p):
        return (gibbs(n1, n0, n0, SA, t, p) -
                temp_ratio * gibbs(n1, n0, n0, SA, pt0, 0))

    def enthalpy_derivative_CT(SA, CT, p):
        return cp0 * temp_ratio

    def enthalpy_derivative_p(SA, CT, p):
        return gibbs(n0, n0, n1, SA, t, p)

    return (enthalpy_derivative_SA(SA, CT, p),
            enthalpy_derivative_CT(SA, CT, p),
            enthalpy_derivative_p(SA, CT, p),)


@match_args_return
def enthalpy_second_derivatives(SA, CT, p):
    r"""Calculates the following three second-order derivatives of specific
    enthalpy (h),
    (1) h_SA_SA, second-order derivative with respect to Absolute Salinity
        at constant CT & p.
    (2) h_SA_CT, second-order derivative with respect to SA & CT at
        constant p.
    (3) h_CT_CT, second-order derivative with respect to CT at constant SA
        and p.

    Parameters
    ----------
    SA : array_like
         Absolute salinity [g kg :sup:`-1`]
    CT : array_like
         Conservative Temperature [:math:`^\circ` C (TEOS-10)]
    p : array_like
        pressure [dbar]

    Returns
    -------
    h_SA_SA : array_like
              The second derivative of specific enthalpy with respect to
              Absolute Salinity at constant CT & p. [J/(kg (g/kg)^2)]
    h_SA_CT : array_like
              The second derivative of specific enthalpy with respect to SA and
              CT at constant p. [J/(kg K(g/kg))]
    h_CT_CT : array_like
              The second derivative of specific enthalpy with respect to CT at
              constant SA and p. [J/(kg K^2)]

    References
    ----------
    .. [1] IOC, SCOR and IAPSO, 2010: The international thermodynamic equation
    of seawater -  2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp. See Eqns. (A.11.18), (A.11.15) and (A.11.12.)

    .. [2] McDougall T.J., P.M. Barker, R. Feistel and D.R. Jackett, 2011:  A
    computationally efficient 48-term expression for the density of seawater in
    terms of Conservative Temperature, and related properties of seawater.  To
    be submitted to Ocean Science Discussions.

    Modifications:
    2011-03-29. Trevor McDougall.
    """

    # NOTE: The Matlab version 3.0 mentions that this function is unchanged,
    # but that's not true!

    n0, n1, n2 = 0, 1, 2
    pt0 = pt_from_CT(SA, CT)
    abs_pt0 = Kelvin + pt0
    t = pt_from_t(SA, pt0, 0, p)
    temp_ratio = (Kelvin + t) / abs_pt0

    rec_gTT_pt0 = 1.0 / gibbs(n0, n2, n0, SA, pt0, 0)
    rec_gTT_t = 1.0 / gibbs(n0, n2, n0, SA, t, p)
    gST_pt0 = gibbs(n1, n1, n0, SA, pt0, 0)
    gST_t = gibbs(n1, n1, n0, SA, t, p)
    gS_pt0 = gibbs(n1, n0, n0, SA, pt0, 0)

    part = ((temp_ratio * gST_pt0 * rec_gTT_pt0 - gST_t * rec_gTT_t) /
                                                                    (abs_pt0))

    factor = gS_pt0 / cp0

    # h_CT_CT is naturally well-behaved as SA approaches zero.
    def enthalpy_derivative_CT_CT(SA, CT, p):
        return (cp0 ** 2 * ((temp_ratio * rec_gTT_pt0 - rec_gTT_t) /
                                                         (abs_pt0 * abs_pt0)))

    # h_SA_SA has a singularity at SA = 0, and blows up as SA approaches zero.
    def enthalpy_derivative_SA_SA(SA, CT, p):
        SA[SA < 1e-100] = 1e-100  # NOTE: Here is the changes from 2.0 to 3.0.
        h_CT_CT = enthalpy_derivative_CT_CT(SA, CT, p)
        return (gibbs(n2, n0, n0, SA, t, p) -
                temp_ratio * gibbs(n2, n0, n0, SA, pt0, 0) +
                temp_ratio * gST_pt0 ** 2 * rec_gTT_pt0 -
                gST_t ** 2 * rec_gTT_t - 2.0 * gS_pt0 * part +
                factor ** 2 * h_CT_CT)

    """h_SA_CT should not blow up as SA approaches zero. The following lines of
    code ensure that the h_SA_CT output of this function does not blow up in
    this limit.  That is, when SA < 1e-100 g/kg, we force the h_SA_CT output to
    be the same as if SA = 1e-100 g/kg."""

    def enthalpy_derivative_SA_CT(SA, CT, p):
        h_CT_CT = enthalpy_derivative_CT_CT(SA, CT, p)
        return cp0 * part - factor * h_CT_CT

    return (enthalpy_derivative_SA_SA(SA, CT, p),
            enthalpy_derivative_SA_CT(SA, CT, p),
            enthalpy_derivative_CT_CT(SA, CT, p))


if __name__ == '__main__':
    import doctest
    doctest.testmod()
