# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np

from library import match_args_return
from constants import cp0
from conversions import t_from_CT, pt_from_CT
from absolute_salinity_sstar_ct import CT_from_t
from basic_thermodynamic_t import rho_t_exact, alpha_wrt_CT_t_exact
from basic_thermodynamic_t import beta_const_CT_t_exact, specvol_t_exact
from basic_thermodynamic_t import specvol_anom_t_exact, sound_speed_t_exact
from basic_thermodynamic_t import t_maxdensity_exact, enthalpy_t_exact
from basic_thermodynamic_t import internal_energy_t_exact, sigma0_pt0_exact
from basic_thermodynamic_t import t_from_rho_exact

__all__ = [
           'rho_CT_exact',
           'alpha_CT_exact',
           'beta_CT_exact',
           'rho_alpha_beta_CT_exact',
           'specvol_CT_exact',
           'specvol_anom_CT_exact',
           'sigma0_CT_exact',
           'sigma1_CT_exact',
           'sigma2_CT_exact',
           'sigma3_CT_exact',
           'sigma4_CT_exact',
           'sound_speed_CT_exact',
           'internal_energy_CT_exact',
           'enthalpy_CT_exact',
           'enthalpy_diff_CT_exact',
           'dynamic_enthalpy_CT_exact',
           'SA_from_rho_CT_exact',
           'CT_from_rho_exact',
           'CT_maxdensity_exact'
           ]


def rho_CT_exact(SA, CT, p):
    r"""Calculates in-situ density from Absolute Salinity and Conservative
    Temperature.

    Parameters
    ----------
    SA : array_like
         Absolute Salinity  [g/kg]
    CT : array_like
         Conservative Temperature [:math:`^\circ` C (ITS-90)]
    p : array_like
        sea pressure [dbar]

    Returns
    -------
    rho_CT_exact : array_like
                   in-situ density [kg/m**3]

    See Also
    --------
    TODO

    Notes
    -----
    The potential density with respect to reference pressure, p_ref, is
    obtained by calling this function with the pressure argument being p_ref
    (i.e. "rho_CT_exact(SA, CT, p_ref)").  This function uses the full Gibbs
    function.  There is an alternative to calling this function, namely
    rho_CT(SA, CT, p), which uses the computationally efficient 48-term
    expression for density in terms of SA, CT and p (McDougall et al., 2011).

    Examples
    --------
    TODO

    References
    ----------
    .. [1] IOC, SCOR and IAPSO, 2010: The international thermodynamic equation
    of seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp. See Eqn. (2.8.2).

    .. [2] McDougall T.J., P.M. Barker, R. Feistel and D.R. Jackett, 2011:  A
    computationally efficient 48-term expression for the density of
    seawater in terms of Conservative Temperature, and related properties
    of seawater.

    Modifications:
    2011-04-03. Trevor McDougall and Paul Barker.
    """

    t = t_from_CT(SA, CT, p)
    return rho_t_exact(SA, t, p)


def alpha_CT_exact(SA, CT, p):
    r"""Calculates the thermal expansion coefficient of seawater with respect
    to Conservative Temperature from Absolute Salinity and Conservative
    Temperature.

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
    alpha_CT_exact : array_like
                     thermal expansion coefficient [K :sup:`-1`]
                     with respect to Conservative Temperature
    See Also
    --------
    TODO

    Notes
    -----
    This function uses the full Gibbs function.  There is an alternative to
    calling this function, namely alpha_wrt_CT(SA, CT, p) which uses the
    computationally efficient 48-term expression for density in terms of SA,
    CT and p (McDougall et al., (2011)).

    Examples
    --------
    TODO

    References
    ----------
    .. [1] IOC, SCOR and IAPSO, 2010: The international thermodynamic equation
    of seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp. See Eqn. (2.18.3).

    .. [2] McDougall T.J., P.M. Barker, R. Feistel and D.R. Jackett, 2011:  A
    computationally efficient 48-term expression for the density of
    seawater in terms of Conservative Temperature, and related properties
    of seawater.

    Modifications:
    2011-03-23. David Jackett, Trevor McDougall and Paul Barker.
    """

    t = t_from_CT(SA, CT, p)
    return alpha_wrt_CT_t_exact(SA, t, p)


def beta_CT_exact(SA, CT, p):
    r"""Calculates the saline (i.e. haline) contraction coefficient of seawater
    at constant Conservative Temperature.

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
    beta_CT_exact : array_like
                    thermal expansion coefficient [K :sup:`-1`]
    See Also
    --------
    TODO

    Notes
    -----
    This function uses the full Gibbs function.  There is an alternative to
    calling this function, namely beta_const_CT(SA, CT, p) which uses the
    computationally efficient 48-term expression for density in terms of SA, CT
    and p (McDougall et al., (2011)).

    Examples
    --------
    TODO

    References
    ----------
    .. [1] IOC, SCOR and IAPSO, 2010: The international thermodynamic equation
    of seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp. See Eqn. (2.19.3).

    .. [2] McDougall T.J., P.M. Barker, R. Feistel and D.R. Jackett, 2011:  A
    computationally efficient 48-term expression for the density of
    seawater in terms of Conservative Temperature, and related properties
    of seawater.

    Modifications:
    2011-03-23. Trevor McDougall and Paul Barker.
    """

    t = t_from_CT(SA, CT, p)
    return beta_const_CT_t_exact(SA, t, p)


def rho_alpha_beta_CT_exact(SA, CT, p):
    r"""Calculates in-situ density, the appropriate thermal expansion
    coefficient and the appropriate saline contraction coefficient of seawater
    from Absolute Salinity and Conservative Temperature.

    See the individual functions rho_CT_exact, alpha_CT_exact, and
    beta_CT_exact.  Retained for compatibility with the Matlab GSW toolbox.
    """

    t = t_from_CT(SA, CT, p)
    rho_CT_exact = rho_t_exact(SA, t, p)
    alpha_CT_exact = alpha_wrt_CT_t_exact(SA, t, p)
    beta_CT_exact = beta_const_CT_t_exact(SA, t, p)

    return rho_CT_exact, alpha_CT_exact, beta_CT_exact


def specvol_CT_exact(SA, CT, p):
    r"""Calculates specific volume from Absolute Salinity, Conservative
    Temperature and pressure.

    Parameters
    ----------
    SA : array_like
         Absolute Salinity  [g/kg]
    CT : array_like
         Conservative Temperature [:math:`^\circ` C (ITS-90)]
    p : array_like
        sea pressure [dbar]

    Returns
    -------
    specvol_CT_exact : array_like
                       specific volume  [m**3/kg]


    See Also
    --------
    TODO

    Notes
    -----
    This function uses the full Gibbs function.  There is an alternative to
    calling this function, namely specvol_CT(SA, CT, p), which uses the
    computationally efficient 48-term expression for density in terms of SA, CT
    and p (McDougall et al., 2011).


    Examples
    --------
    TODO

    References
    ----------
    .. [1] IOC, SCOR and IAPSO, 2010: The international thermodynamic equation
    of seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp. See Eqn. (2.7.2).

    .. [2] McDougall T.J., P.M. Barker, R. Feistel and D.R. Jackett, 2011:  A
    computationally efficient 48-term expression for the density of
    seawater in terms of Conservative Temperature, and related properties
    of seawater.

    Modifications:
    2011-04-06. Trevor McDougall and Paul Barker.
    """

    t = t_from_CT(SA, CT, p)
    return specvol_t_exact(SA, t, p)


def specvol_anom_CT_exact(SA, CT, p):
    r"""Calculates specific volume anomaly from Absolute Salinity, Conservative
    Temperature and pressure.  The reference value of Absolute Salinity is SSO
    and the reference value of Conservative Temperature is equal to 0 deg C.

    Parameters
    ----------
    SA : array_like
         Absolute Salinity  [g/kg]
    CT : array_like
         Conservative Temperature [:math:`^\circ` C (ITS-90)]
    p : array_like
        sea pressure [dbar]

    Returns
    -------
    specvol_anom_CT_exact : array_like
                            specific volume anomaly [m**3/kg]

    See Also
    --------
    TODO

    Notes
    -----
    This function uses the full Gibbs function.  There is an alternative to
    calling this function, namely specvol_anom_CT(SA, CT, p), which uses the
    computationally efficient 48-term expression for density in terms of SA, CT
    and p (McDougall et al., 2011).

    Examples
    --------
    TODO

    References
    ----------
    .. [1] IOC, SCOR and IAPSO, 2010: The international thermodynamic equation
    of seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp. See Eqn. (3.7.3).

    .. [2] McDougall T.J., P.M. Barker, R. Feistel and D.R. Jackett, 2011:  A
    computationally efficient 48-term expression for the density of
    seawater in terms of Conservative Temperature, and related properties
    of seawater.

    Modifications:
    2011-04-06. Trevor McDougall and Paul Barker.
    """

    t = t_from_CT(SA, CT, p)
    return specvol_anom_t_exact(SA, t, p)


def sigma0_CT_exact(SA, CT):
    r"""Calculates potential density anomaly with reference pressure of 0 dbar,
    this being this particular potential density minus 1000 kg/m^3.  This
    function has inputs of Absolute Salinity and Conservative Temperature.


    Parameters
    ----------
    SA : array_like
         Absolute Salinity [g/kg]
    CT : array_like
         Conservative Temperature [:math:`^\circ` C (ITS-90)]

    Returns
    -------
    sigma0_CT_exact: array_like
                     Potential density anomaly with [kg/m**3]
                     respect to a reference pressure of 0 dbar
                     that is, this potential density - 1000 kg/m**3.

    Notes
    -----
    Note that this function uses the full Gibbs function.  There is an
    alternative to calling this function, namely gsw_sigma0_CT(SA,CT,p), which
    uses the computationally efficient 48-term expression for density in terms
    of SA, CT and p (McDougall et al., 2011).

    Examples
    --------
    TODO

    References
    ----------
    .. [1] IOC, SCOR and IAPSO, 2010: The international thermodynamic equation
    of seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp. See Eqn. (A.30.1).

    .. [2] McDougall T.J., P.M. Barker, R. Feistel and D.R. Jackett, 2011:  A
    computationally efficient 48-term expression for the density of
    seawater in terms of Conservative Temperature, and related properties
    of seawater.

    Modifications:
    2011-04-03. Trevor McDougall and Paul Barker.
    """

    pt0 = pt_from_CT(SA, CT)
    return sigma0_pt0_exact(SA, pt0)


def sigma1_CT_exact(SA, CT):
    r"""Calculates potential density anomaly with reference pressure of
    1000 dbar."""
    t = t_from_CT(SA, CT, 1000.)
    return rho_t_exact(SA, t, 1000.) - 1000


def sigma2_CT_exact(SA, CT):
    r"""Calculates potential density anomaly with reference pressure of
    2000 dbar."""
    t = t_from_CT(SA, CT, 2000.)
    return rho_t_exact(SA, t, 2000.) - 1000


def sigma3_CT_exact(SA, CT):
    r"""Calculates potential density anomaly with reference pressure of
    3000 dbar."""
    t = t_from_CT(SA, CT, 3000.)
    return rho_t_exact(SA, t, 3000.) - 1000


def sigma4_CT_exact(SA, CT):
    r"""Calculates potential density anomaly with reference pressure of
    4000 dbar."""
    t = t_from_CT(SA, CT, 4000.)
    return rho_t_exact(SA, t, 4000.) - 1000


def sound_speed_CT_exact(SA, CT, p):
    r"""Calculates the speed of sound in seawater from Absolute Salinity and
    Conservative Temperature and pressure.

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
    sound_speed_CT_exact : array_like
    Speed of sound in seawater [m s :sup:`-1`]


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
    UNESCO (English), 196 pp. See Eqn. (2.17.1).

    Modifications:
    2011-04-05. David Jackett, Paul Barker and Trevor McDougall.
    """

    t = t_from_CT(SA, CT, p)
    return sound_speed_t_exact(SA, t, p)


def internal_energy_CT_exact(SA, CT, p):
    r"""Calculates the specific internal energy of seawater from Absolute
    Salinity, Conservative Temperature and pressure.

    Parameters
    ----------
    SA : array_like
         Absolute Salinity [g/kg]
    CT : array_like
         Conservative Temperature [:math:`^\circ` C (ITS-90)]
    p : array_like
        sea pressure [dbar]

    Returns
    -------
    internal_energy_CT_exact: array_like
                              specific internal energy (u) [J/kg]

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
    UNESCO (English), 196 pp. See Eqn. (2.11.1).

    Modifications:
    2011-04-05. Trevor McDougall.
    """

    t = t_from_CT(SA, CT, p)
    return internal_energy_t_exact(SA, t, p)


def enthalpy_CT_exact(SA, CT, p):
    r"""Calculates specific enthalpy of seawater from Absolute Salinity and
    Conservative Temperature and pressure.

    Parameters
    ----------
    SA : array_like
         Absolute Salinity  [g/kg]
    CT : array_like
         Conservative Temperature [:math:`^\circ` C (ITS-90)]
    p : array_like
        sea pressure [dbar]

    Returns
    -------
    enthalpy_CT_exact : array_like
                        specific enthalpy  [J/kg]

    See Also
    --------
    TODO

    Notes
    -----
    This function uses the full Gibbs function.  There is an alternative to
    calling this function, namely enthalpy_CT(SA, CT, p), which uses the
    computationally-efficient 48-term expression for density in terms of SA, CT
    and p (McDougall et al., 2011).

    Examples
    --------
    TODO

    References
    ----------
    .. [1] IOC, SCOR and IAPSO, 2010: The international thermodynamic equation
    of seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp. See appendix A.11.

    .. [2] McDougall T.J., P.M. Barker, R. Feistel and D.R. Jackett, 2011:  A
    computationally efficient 48-term expression for the density of
    seawater in terms of Conservative Temperature, and related properties
    of seawater.

    Modifications:
    2011-04-06. Trevor McDougall and Paul Barker.
    """

    t = t_from_CT(SA, CT, p)
    return enthalpy_t_exact(SA, t, p)


def enthalpy_diff_CT_exact(SA, CT, p_shallow, p_deep):
    r"""Calculates the difference of the specific enthalpy of seawater between
    two different pressures, p_deep (the deeper pressure) and p_shallow (the
    shallower pressure), at the same values of SA and CT.  The output
    (enthalpy_diff_CT_exact) is the specific enthalpy evaluated at
    (SA, CT, p_deep) minus the specific enthalpy at (SA,CT,p_shallow).

    parameters
    ----------
    Parameters
    ----------
    SA : array_like
         Absolute Salinity  [g/kg]
    CT : array_like
         Conservative Temperature [:math:`^\circ` C (ITS-90)]
    p_shallow : array_like
                lower sea pressure [dbar]
    p_deep : array-like
             upper sea pressure [dbar]

    returns
    -------
    enthalpy_diff_CT_exact : array_like
                             difference of specific enthalpy [J/kg]
                             (deep minus shallow)

    See Also
    --------
    TODO

    Notes
    -----
    This function uses the full Gibbs function.  There is an alternative to
    calling this function, namely enthalpy_diff_CT(SA, CT, p), which uses the
    computationally efficient 48-term expression for density in terms of SA, CT
    and p (McDougall et al., 2011).

    Examples
    --------
    TODO

    References
    ----------
    .. [1] IOC, SCOR and IAPSO, 2010: The international thermodynamic equation
    of seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp. See Eqns (3.32.2).

    .. [2] McDougall T.J., P.M. Barker, R. Feistel and D.R. Jackett, 2011:  A
    computationally efficient 48-term expression for the density of
    seawater in terms of Conservative Temperature, and related properties
    of seawater.

    Modifications:
    2011-04-06. Trevor McDougall and Paul Barker.
    """

    t_shallow = t_from_CT(SA, CT, p_shallow)
    t_deep = t_from_CT(SA, CT, p_deep)
    return (enthalpy_t_exact(SA, t_deep, p_deep) -
            enthalpy_t_exact(SA, t_shallow, p_shallow))


def dynamic_enthalpy_CT_exact(SA, CT, p):
    r"""Calculates the dynamic enthalpy of seawater from Absolute Salinity and
    Conservative Temperature and pressure.  Dynamic enthalpy is defined as
    enthalpy minus potential enthalpy (Young, 2010).

    Parameters
    ----------
    SA : array_like
         Absolute Salinity  [g/kg]
    CT : array_like
         Conservative Temperature [:math:`^\circ` C (ITS-90)]
    p : array_like
        sea pressure [dbar]

    Returns
    -------
    dynamic_enthalpy_CT_exact : array_like
                                dynamic enthalpy [J/kg]

    See Also
    --------
    TODO

    Notes
    -----
    This function uses the full Gibbs function.  There is an alternative to
    calling this function, namely dynamic_enthalpy(SA, CT, p), which uses the
    computationally efficient 48-term expression for density in terms of SA, CT
    and p (McDougall et al., 2011).

    Examples
    --------
    TODO

    References
    ----------
    .. [1] IOC, SCOR and IAPSO, 2010: The international thermodynamic equation
    of seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp. See apendix A.30.

    .. [2] McDougall T.J., P.M. Barker, R. Feistel and D.R. Jackett, 2011:  A
    computationally efficient 48-term expression for the density of
    seawater in terms of Conservative Temperature, and related properties
    of seawater.

    .. [3] Young, W.R., 2010: Dynamic enthalpy, Conservative Temperature, and
    the seawater Boussinesq approximation. Journal of Physical Oceanography,
    40, 394-400.

    Modifications:
    2011-04-05. Trevor McDougall and Paul Barker.
    """

    t = t_from_CT(SA, CT, p)
    return enthalpy_t_exact(SA, t, p) - cp0 * CT


@match_args_return
def SA_from_rho_CT_exact(rho, CT, p):
    r"""Calculates the Absolute Salinity of a seawater sample, for given values
    of its density, Conservative Temperature and sea pressure (in dbar).

    Parameters
    ----------
    rho : array_like
          density of a seawater sample [kg/m**3]
          This input has not had 1000 kg/m^3 subtracted from it
          (e.g. 1026 kg m**-3), that is, it is density, NOT density anomaly.
    CT : array_like
         Conservative Temperature [:math:`^\circ` C (ITS-90)]
    p : array_like
        sea pressure [dbar]

    Returns
    -------
    SA : array_like
         Absolute Salinity  [g/kg]

    See Also
    --------
    TODO

    Notes
    -----
    This function uses the full Gibbs function.  There is an alternative to
    calling this function, namely SA_from_rho_CT(rho, CT, p), which uses the
    computationally efficient 48-term expression for density in terms of SA, CT
    and p (McDougall et al., 2011).

    Examples
    --------
    TODO

    References
    ----------
    .. [1] IOC, SCOR and IAPSO, 2010: The international thermodynamic equation
    of seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp. See section 2.5.

    .. [2] Millero, F. J., R. Feistel, D. G. Wright, and T. J. McDougall, 2008:
    The composition of Standard Seawater and the definition of the
    Reference-Composition Salinity Scale, Deep-Sea Res. I, 55, 50-72.

    Modifications:
    2011-04-05. Trevor McDougall and Paul Barker.
    """

    v_lab = 1. / rho
    v_0 = specvol_CT_exact(np.zeros_like(rho), CT, p)
    v_120 = specvol_CT_exact(120 * np.ones_like(rho), CT, p)

    SA = 120 * (v_lab - v_0) / (v_120 - v_0)  # Initial estimate of SA.

    SA[np.logical_or(SA < 0, SA > 120)] = np.NaN

    v_SA = (v_120 - v_0) / 120  # Initial v_SA estimate (SA derivative of v).

    # Begin the modified Newton-Raphson iterative procedure.
    for Number_of_iterations in range(0, 3):
        SA_old = SA
        delta_v = specvol_CT_exact(SA_old, CT, p) - v_lab
        # Half way the mod. N-R method (McDougall and Wotherspoon, 2012).
        SA = SA_old - delta_v / v_SA
        SA_mean = 0.5 * (SA + SA_old)
        rho, alpha, beta = rho_alpha_beta_CT_exact(SA_mean, CT, p)
        v_SA = -beta / rho
        SA = SA_old - delta_v / v_SA
        SA[np.logical_or(SA < 0, SA > 120)] = np.ma.masked

    """After two iterations of this modified Newton-Raphson iteration, the
    error in SA is no larger than 8x10^-13 g kg^-1, which is machine precision
    for this calculation."""

    return SA


def CT_from_rho_exact(rho, SA, p):
    r"""Calculates the in-situ temperature of a seawater sample, for given
    values of its density, Absolute Salinity and sea pressure (in dbar).

    Parameters
    ----------
    rho : array_like
          density of a seawater sample [kg/m**3]
          This input has not had 1000 kg/m^3 subtracted from it
          (e.g. 1026 kg m**-3), that is, it is density, NOT density anomaly.
    SA : array_like
         Absolute Salinity  [g/kg]
    p : array_like
        sea pressure [dbar]

    Returns
    -------
    CT : array_like
         Conservative Temperature [:math:`^\circ` C (ITS-90)]
    CT_multiple : array_like
                  Conservative Temperature [:math:`^\circ` C (ITS-90)]

    See Also
    --------
    TODO

    Notes
    -----
    At low salinities, in brackish water, there are two possible temperatures
    for a single density.  This program will output both valid solutions
    (t, t_multiple), if there is only one possible solution the second variable
    will be set to NaN.


    Examples
    --------
    TODO

    References
    ----------
    .. [1] IOC, SCOR and IAPSO, 2010: The international thermodynamic equation
    of seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp. See section 2.5.

    Modifications:
    2011-04-21. Trevor McDougall and Paul Barker.
    """

    t, t_multiple = t_from_rho_exact(rho, SA, p)
    return CT_from_t(SA, t, p), CT_from_t(SA, t_multiple, p)


def CT_maxdensity_exact(SA, p):
    r"""Calculates the Conservative Temperature of maximum density of seawater.
    This function returns the Conservative temperature at which the density of
    seawater is a maximum, at given Absolute Salinity, SA, and sea pressure,
    p (in dbar).


    Parameters
    ----------
    SA : array_like
         Absolute Salinity  [g/kg]
    p : array_like
        sea pressure [dbar]

    Returns
    -------
    CT_maxdensity_exact : array_like
         Conservative Temperature [:math:`^\circ` C (ITS-90)]
         at which the density of seawater is a maximum for given SA and p.

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
    UNESCO (English), 196 pp. See section 3.42.

    Modifications:
    2011-04-03. Trevor McDougall and Paul Barker.
    """

    t_max_exact = t_maxdensity_exact(SA, p)
    return CT_from_t(SA, t_max_exact, p)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
