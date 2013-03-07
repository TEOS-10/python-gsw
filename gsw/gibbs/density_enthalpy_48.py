# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np

from library import specvol_SSO_0_p
from constants import P0, db2Pascal, cp0
from gsw.utilities import match_args_return

__all__ = [
           'rho',
           'alpha',
           'beta',
           'rho_alpha_beta',
           'specvol',
           'specvol_anom',
           'sigma0',
           'sigma1',
           'sigma2',
           'sigma3',
           'sigma4',
           'sound_speed',
           'internal_energy',
           'enthalpy',
           'enthalpy_diff',
           'dynamic_enthalpy',
           'SA_from_rho'
           ]


# NOTE: 48-term equation?
v01 = 9.998420897506056e+2
v02 = 2.839940833161907
v03 = -3.147759265588511e-2
v04 = 1.181805545074306e-3
v05 = -6.698001071123802
v06 = -2.986498947203215e-2
v07 = 2.327859407479162e-4
v08 = -3.988822378968490e-2
v09 = 5.095422573880500e-4
v10 = -1.426984671633621e-5
v11 = 1.645039373682922e-7
v12 = -2.233269627352527e-2
v13 = -3.436090079851880e-4
v14 = 3.726050720345733e-6
v15 = -1.806789763745328e-4
v16 = 6.876837219536232e-7
v17 = -3.087032500374211e-7
v18 = -1.988366587925593e-8
v19 = -1.061519070296458e-11
v20 = 1.550932729220080e-10
v21 = 1.0
v22 = 2.775927747785646e-3
v23 = -2.349607444135925e-5
v24 = 1.119513357486743e-6
v25 = 6.743689325042773e-10
v26 = -7.521448093615448e-3
v27 = -2.764306979894411e-5
v28 = 1.262937315098546e-7
v29 = 9.527875081696435e-10
v30 = -1.811147201949891e-11
v31 = -3.303308871386421e-5
v32 = 3.801564588876298e-7
v33 = -7.672876869259043e-9
v34 = -4.634182341116144e-11
v35 = 2.681097235569143e-12
v36 = 5.419326551148740e-6
v37 = -2.742185394906099e-5
v38 = -3.212746477974189e-7
v39 = 3.191413910561627e-9
v40 = -1.931012931541776e-12
v41 = -1.105097577149576e-7
v42 = 6.211426728363857e-10
v43 = -1.119011592875110e-10
v44 = -1.941660213148725e-11
v45 = -1.864826425365600e-14
v46 = 1.119522344879478e-14
v47 = -1.200507748551599e-15
v48 = 6.057902487546866e-17

a01 = 2.839940833161907
a02 = -6.295518531177023e-2
a03 = 3.545416635222918e-3
a04 = -2.986498947203215e-2
a05 = 4.655718814958324e-4
a06 = 5.095422573880500e-4
a07 = -2.853969343267241e-5
a08 = 4.935118121048767e-7
a09 = -3.436090079851880e-4
a10 = 7.452101440691467e-6
a11 = 6.876837219536232e-7
a12 = -1.988366587925593e-8
a13 = -2.123038140592916e-11

a14 = 2.775927747785646e-3
a15 = -4.699214888271850e-5
a16 = 3.358540072460230e-6
a17 = 2.697475730017109e-9
a18 = -2.764306979894411e-5
a19 = 2.525874630197091e-7
a20 = 2.858362524508931e-9
a21 = -7.244588807799565e-11
a22 = 3.801564588876298e-7
a23 = -1.534575373851809e-8
a24 = -1.390254702334843e-10
a25 = 1.072438894227657e-11
a26 = -3.212746477974189e-7
a27 = 6.382827821123254e-9
a28 = -5.793038794625329e-12
a29 = 6.211426728363857e-10
a30 = -1.941660213148725e-11
a31 = -3.729652850731201e-14
a32 = 1.119522344879478e-14
a33 = 6.057902487546866e-17

b01 = -6.698001071123802
b02 = -2.986498947203215e-2
b03 = 2.327859407479162e-4
b04 = -5.983233568452735e-2
b05 = 7.643133860820750e-4
b06 = -2.140477007450431e-5
b07 = 2.467559060524383e-7
b08 = -1.806789763745328e-4
b09 = 6.876837219536232e-7
b10 = 1.550932729220080e-10
b11 = -7.521448093615448e-3
b12 = -2.764306979894411e-5
b13 = 1.262937315098546e-7
b14 = 9.527875081696435e-10
b15 = -1.811147201949891e-11
b16 = -4.954963307079632e-5
b17 = 5.702346883314446e-7
b18 = -1.150931530388857e-8
b19 = -6.951273511674217e-11
b20 = 4.021645853353715e-12
b21 = 1.083865310229748e-5
b22 = -1.105097577149576e-7
b23 = 6.211426728363857e-10
b24 = 1.119522344879478e-14

c01 = -2.233269627352527e-2
c02 = -3.436090079851880e-4
c03 = 3.726050720345733e-6
c04 = -1.806789763745328e-4
c05 = 6.876837219536232e-7
c06 = -6.174065000748422e-7
c07 = -3.976733175851186e-8
c08 = -2.123038140592916e-11
c09 = 3.101865458440160e-10

c10 = -2.742185394906099e-5
c11 = -3.212746477974189e-7
c12 = 3.191413910561627e-9
c13 = -1.931012931541776e-12
c14 = -1.105097577149576e-7
c15 = 6.211426728363857e-10
c16 = -2.238023185750219e-10
c17 = -3.883320426297450e-11
c18 = -3.729652850731201e-14
c19 = 2.239044689758956e-14
c20 = -3.601523245654798e-15
c21 = 1.817370746264060e-16


def v_hat_denominator(SA, CT, p):
    return (v01 + CT * (v02 + CT * (v03 + v04 * CT)) + SA *
           (v05 + CT * (v06 + v07 * CT) + np.sqrt(SA) *
           (v08 + CT * (v09 + CT * (v10 + v11 * CT)))) + p *
           (v12 + CT * (v13 + v14 * CT) + SA * (v15 + v16 * CT) + p *
           (v17 + CT * (v18 + v19 * CT) + v20 * SA)))


def v_hat_numerator(SA, CT, p):
    return (v21 + CT * (v22 + CT * (v23 + CT * (v24 + v25 * CT))) + SA *
           (v26 + CT * (v27 + CT * (v28 + CT * (v29 + v30 * CT))) + v36 * SA +
           np.sqrt(SA) * (v31 + CT * (v32 + CT * (v33 + CT *
           (v34 + v35 * CT))))) + p * (v37 + CT * (v38 + CT *
           (v39 + v40 * CT)) + SA * (v41 + v42 * CT) + p * (v43 + CT *
           (v44 + v45 * CT + v46 * SA) + p * (v47 + v48 * CT))))


@match_args_return
def rho(SA, CT, p):
    r"""Calculates in-situ density from Absolute Salinity and Conservative
    Temperature, using the computationally-efficient 48-term expression for
    density in terms of SA, CT and p (McDougall et al., 2011).

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
    rho : array_like
          in-situ density [kg/m**3]

    See Also
    --------
    TODO

    Notes
    -----
    The potential density with respect to reference pressure, pr, is obtained
    by calling this function with the pressure argument being pr (i.e.
    "rho(SA,CT,pr)").

    The 48-term equation has been fitted in a restricted range of parameter
    space, and is most accurate inside the "oceanographic funnel" described in
    McDougall et al. (2011).  The GSW library function "infunnel(SA,CT,p)" is
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
    UNESCO (English), 196 pp. See appendix A.20 and appendix K.

    .. [2] McDougall T.J., P.M. Barker, R. Feistel and D.R. Jackett, 2011:  A
    computationally efficient 48-term expression for the density of
    seawater in terms of Conservative Temperature, and related properties
    of seawater.

    Modifications:
    2011-04-18. Paul Barker and Trevor McDougall.
    """

    SA = np.maximum(SA, 0)

    """This function calculates rho using the computationally-efficient 48-term
    expression for density in terms of SA, CT and p. If one wanted to compute
    rho from SA, CT, and p with the full TEOS-10 Gibbs function, the following
    lines of code will enable this.

    pt0 = pt_from_CT(SA, CT)
    t = pt_from_t(SA, pt0, 0, p)
    rho = rho_t_exact(SA, t, p)

    or call the following, it is identical to the lines above.

    rho = rho_CT_exact(SA, CT, p)

    or call the following, it is identical to the lines above.

    rho,_ ,_ = rho_alpha_beta_CT_exact(SA, CT, p)
    """

    return v_hat_denominator(SA, CT, p) / v_hat_numerator(SA, CT, p)


@match_args_return
def alpha(SA, CT, p):
    r"""Calculates the thermal expansion coefficient of seawater with respect
    to Conservative Temperature using the computationally-efficient 48-term
    expression for density in terms of SA, CT and p (McDougall et al., 2011)


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
    alpha : array_like
            thermal expansion coefficient [K :math:`-1`]
            with respect to Conservative Temperature

    See Also
    --------
    TODO

    Notes
    -----
    The 48-term equation has been fitted in a restricted range of parameter
    space, and is most accurate inside the "oceanographic funnel"
    described in McDougall et al. (2011).  The GSW library function
    "infunnel(SA, CT, p)" is available to be used if one wants to test if
    some of one's data lies outside this "funnel".

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
    2011-03-23. Paul Barker and Trevor McDougall.
    """

    SA = np.maximum(SA, 0)

    sqrtSA = np.sqrt(SA)

    spec_vol = v_hat_numerator(SA, CT, p) / v_hat_denominator(SA, CT, p)

    dvhatden_dCT = (a01 + CT * (a02 + a03 * CT) + SA * (a04 + a05 * CT +
                   sqrtSA * (a06 + CT * (a07 + a08 * CT))) + p *
                   (a09 + a10 * CT + a11 * SA + p * (a12 + a13 * CT)))

    dvhatnum_dCT = (a14 + CT * (a15 + CT * (a16 + a17 * CT)) + SA *
                   (a18 + CT * (a19 + CT * (a20 + a21 * CT)) + sqrtSA *
                   (a22 + CT * (a23 + CT * (a24 + a25 * CT)))) + p *
                   (a26 + CT * (a27 + a28 * CT) + a29 * SA + p *
                   (a30 + a31 * CT + a32 * SA + a33 * p)))

    return ((dvhatnum_dCT - dvhatden_dCT * spec_vol) /
            v_hat_numerator(SA, CT, p))


@match_args_return
def beta(SA, CT, p):
    r"""Calculates the saline (i.e. haline) contraction coefficient of seawater
    at constant Conservative Temperature using the computationally-efficient
    48-term expression for density in terms of SA, CT and p (McDougall et al.,
    2011).

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
    beta : array_like
           saline contraction coefficient [kg g :math:`-1`]
           at constant Conservative Temperature.

    See Also
    --------
    TODO

    Notes
    -----
    The 48-term equation has been fitted in a restricted range of parameter
    space, and is most accurate inside the "oceanographic funnel"
    described in McDougall et al. (2011).  The GSW library function
    "infunnel(SA, CT, p)" is available to be used if one wants to test if some
    of one's data lies outside this "funnel".

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
    2011-03-23. Paul Barker and Trevor McDougall.
    """

    SA = np.maximum(SA, 0)

    sqrtSA = np.sqrt(SA)

    spec_vol = v_hat_numerator(SA, CT, p) / v_hat_denominator(SA, CT, p)

    dvhatden_dSA = (b01 + CT * (b02 + b03 * CT) + sqrtSA *
                   (b04 + CT * (b05 + CT * (b06 + b07 * CT))) + p *
                   (b08 + b09 * CT + b10 * p))

    dvhatnum_dSA = (b11 + CT * (b12 + CT * (b13 + CT * (b14 + b15 * CT))) +
                   sqrtSA * (b16 + CT * (b17 + CT * (b18 + CT * (b19 + b20 *
                   CT)))) + b21 * SA + p * (b22 + CT * (b23 + b24 * p)))

    return ((dvhatden_dSA * spec_vol - dvhatnum_dSA) /
            v_hat_numerator(SA, CT, p))


def rho_alpha_beta(SA, CT, p):
    r"""Calculates in-situ density, the appropriate thermal expansion
    coefficient and the appropriate saline contraction coefficient of seawater
    from Absolute Salinity and Conservative Temperature.  This function uses
    the computationally-efficient 48-term expression for density in terms of
    SA, CT and p (McDougall et al., 2011).

    The potential density (pot_rho) with respect to reference pressure p_ref is
    obtained by calling this function with the pressure argument being p_ref as
    in pot_rho, _, _] = rho_alpha_beta(SA, CT, p_ref).

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
    rho : array_like
          in-situ density [kg/m**3]
    alpha : array_like
            thermal expansion coefficient [K :math:`-1`]
            with respect to Conservative Temperature
    beta : array_like
           saline contraction coefficient [kg g :math:`-1`]
           at constant Conservative Temperature.

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
    UNESCO (English), 196 pp. See appendix A.20 and appendix K.

    .. [2] McDougall T.J., P.M. Barker, R. Feistel and D.R. Jackett, 2011:  A
    computationally efficient 48-term expression for the density of
    seawater in terms of Conservative Temperature, and related properties
    of seawater.

    Modifications:
    2011-04-03. Paul Barker and Trevor McDougall.
    """

    return rho(SA, CT, p), alpha(SA, CT, p), beta(SA, CT, p)


@match_args_return
def specvol(SA, CT, p):
    r"""Calculates specific volume from Absolute Salinity, Conservative
    Temperature and pressure, using the computationally-efficient 48-term
    expression for density (McDougall et al., 2011).


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
    specvol : array_like
              specific volume [m**3/kg]

    See Also
    --------
    TODO

    Notes
    -----
    The 48-term equation has been fitted in a restricted range of parameter
    space, and is most accurate inside the "oceanographic funnel" described in
    McDougall et al. (2011).  The GSW library function "infunnel(SA,CT,p)" is
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
    UNESCO (English), 196 pp. See Eqn. (2.7.2).

    .. [2] McDougall T.J., P.M. Barker, R. Feistel and D.R. Jackett, 2011:  A
    computationally efficient 48-term expression for the density of
    seawater in terms of Conservative Temperature, and related properties
    of seawater.

    Modifications:
    2011-03-18. Paul Barker and Trevor McDougall.
    """

    SA = np.maximum(SA, 0)

    """This function calculates specvol using the computationally-efficient
    48-term expression for density in terms of SA, CT and p. If one wanted to
    compute specvol from SA, CT, and p with the full TEOS-10 Gibbs function,
    the following lines of code will enable this.

    pt = pt_from_CT(SA, CT)
    t = pt_from_t(SA, pt, 0, p)
    specvol = specvol_t_exact(SA, t, p)

    or call the following, it is identical to the lines above.

    specvol = specvol_CT_exact(SA, CT, p)
    """

    return v_hat_numerator(SA, CT, p) / v_hat_denominator(SA, CT, p)


@match_args_return
def specvol_anom(SA, CT, p):
    r"""Calculates specific volume anomaly from Absolute Salinity, Conservative
    Temperature and pressure.  It uses the computationally-efficient 48-term
    expression for density as a function of SA, CT and p (McDougall et al.,
    2011).  The reference value of Absolute Salinity is SSO and the reference
    value of Conservative Temperature is equal to 0 degrees C.


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
    specvol_anom : array_like
                   specific volume anomaly [m**3/kg]

    See Also
    --------
    TODO

    Notes
    -----
    The 48-term equation has been fitted in a restricted range of parameter
    space, and is most accurate inside the "oceanographic funnel" described in
    McDougall et al. (2011).  The GSW library function "infunnel(SA,CT,p)" is
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
    UNESCO (English), 196 pp. See Eqn. (3.7.3).

    .. [2] McDougall T.J., P.M. Barker, R. Feistel and D.R. Jackett, 2011:  A
    computationally efficient 48-term expression for the density of
    seawater in terms of Conservative Temperature, and related properties
    of seawater.

    Modifications:
    2011-03-24. Paul Barker and Trevor McDougall.
    """

    SA = np.maximum(SA, 0)

    """This function calculates specvol_anom using the computationally-
    efficient 48-term expression for density in terms of SA, CT and p.  If
    one wanted to compute specvol_anom from SA, CT, and p with the full
    TEOS-10 Gibbs function, the following lines of code will enable this.

    pt = pt_from_CT(SA, CT)
    t = pt_from_t(SA, pt, 0, p)
    specvol_anom = specvol_anom_t_exact(SA, t, p)

    or call the following, it is identical to the lines above.

    specvol_anom = specvol_anom_CT_exact(SA, CT, p)
    """

    return (v_hat_numerator(SA, CT, p) / v_hat_denominator(SA, CT, p) -
                                                           specvol_SSO_0_p(p))


def sigma0(SA, CT):
    r"""Calculates potential density anomaly with reference pressure of 0 dbar,
    this being this particular potential density minus 1000 kg/m^3.  This
    function has inputs of Absolute Salinity and Conservative Temperature.
    This function uses the computationally-efficient 48-term expression for
    density in terms of SA, CT and p (McDougall et al., 2011).


    Parameters
    ----------
    SA : array_like
         Absolute Salinity  [g/kg]
    CT : array_like
         Conservative Temperature [:math:`^\circ` C (ITS-90)]

    Returns
    -------
    sigma0 : array_like
             potential density anomaly with [kg/m**3]
             respect to a reference pressure of 0 dbar

    See Also
    --------
    gsw.rho
    """

    """This function calculates sigma0 using the computationally-efficient
    48-term expression for density in terms of SA, CT and p.  If one wanted
    to compute sigma0 with the full TEOS-10 Gibbs function expression for
    density, the following lines of code will enable this.

    sigma0 = rho_CT_exact(SA, CT, 0) - 1000
    """

    return rho(SA, CT, 0.) - 1000


def sigma1(SA, CT):
    r"""Calculates potential density anomaly with reference pressure of 1000
    dbar, this being this particular potential density minus 1000 kg/m^3.
    This function has inputs of Absolute Salinity and Conservative Temperature.


    Parameters
    ----------
    SA : array_like
         Absolute Salinity  [g/kg]
    CT : array_like
         Conservative Temperature [:math:`^\circ` C (ITS-90)]

    Returns
    -------
    sigma1 : array_like
             potential density anomaly with [kg/m**3]
             respect to a reference pressure of 1000 dbar

    See Also
    --------
    gsw.rho
    """

    """This function calculates sigma1 using the computationally-efficient
    48-term expression for density in terms of SA, CT and p.  If one wanted
    to compute sigma1 with the full TEOS-10 Gibbs function expression for
    density, the following lines of code will enable this.

    rho1 = rho_CT_exact(SA, CT, 1000)
    """

    return rho(SA, CT, 1000.) - 1000


def sigma2(SA, CT):
    r"""Calculates potential density anomaly with reference pressure of 2000
    dbar, this being this particular potential density minus 1000 kg/m^3.
    This function has inputs of Absolute Salinity and Conservative Temperature.


    Parameters
    ----------
    SA : array_like
         Absolute Salinity  [g/kg]
    CT : array_like
         Conservative Temperature [:math:`^\circ` C (ITS-90)]

    Returns
    -------
    sigma2 : array_like
             potential density anomaly with [kg/m**3]
             respect to a reference pressure of 2000 dbar

    See Also
    --------
    gsw.rho
    """

    """This function calculates sigma2 using the computationally-efficient
    48-term expression for density in terms of SA, CT and p.  If one wanted
    to compute sigma2 with the full TEOS-10 Gibbs function expression for
    density, the following lines of code will enable this.

    rho2 = rho_CT_exact(SA, CT, 2000.)
    """

    return rho(SA, CT, 2000.) - 1000


def sigma3(SA, CT):
    r"""Calculates potential density anomaly with reference pressure of 3000
    dbar, this being this particular potential density minus 1000 kg/m^3.
    This function has inputs of Absolute Salinity and Conservative Temperature.


    Parameters
    ----------
    SA : array_like
         Absolute Salinity  [g/kg]
    CT : array_like
         Conservative Temperature [:math:`^\circ` C (ITS-90)]

    Returns
    -------
    sigma1 : array_like
             potential density anomaly with [kg/m**3]
             respect to a reference pressure of 3000 dbar

    See Also
    --------
    gsw.rho
    """

    """This function calculates sigma3 using the computationally-efficient
    48-term expression for density in terms of SA, CT and p.  If one wanted
    to compute sigma3 with the full TEOS-10 Gibbs function expression for
    density, the following lines of code will enable this.

    rho3 = rho_CT_exact(SA, CT, 3000.)
    """

    return rho(SA, CT, 3000.) - 1000


def sigma4(SA, CT):
    r"""Calculates potential density anomaly with reference pressure of 4000
    dbar, this being this particular potential density minus 1000 kg/m^3.
    This function has inputs of Absolute Salinity and Conservative Temperature.


    Parameters
    ----------
    SA : array_like
         Absolute Salinity  [g/kg]
    CT : array_like
         Conservative Temperature [:math:`^\circ` C (ITS-90)]

    Returns
    -------
    sigma1 : array_like
             potential density anomaly with [kg/m**3]
             respect to a reference pressure of 4000 dbar

    See Also
    --------
    gsw.rho
    """

    """This function calculates sigma3 using the computationally-efficient
    48-term expression for density in terms of SA, CT and p.  If one wanted
    to compute sigma3 with the full TEOS-10 Gibbs function expression for
    density, the following lines of code will enable this.

    rho4 = rho_CT_exact(SA, CT, 4000.)
    """

    return rho(SA, CT, 4000.) - 1000


@match_args_return
def sound_speed(SA, CT, p):
    r"""Calculates the speed of sound in seawater.  This function has inputs of
    Absolute Salinity and Conservative Temperature.  This function uses the
    computationally-efficient 48-term expression for density in terms of SA,
    CT and p (McDougall et al., 2011).


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
    sound_speed : array_like
                  speed of sound in seawater [m/s]

    See Also
    --------
    TODO

    Notes
    -----
    Approximate with a r.m.s. of 6.7 cm s^-1.

    The 48-term equation has been fitted in a restricted range of parameter
    space, and is most accurate inside the "oceanographic funnel"
    described in McDougall et al. (2011).  The GSW library function
    "infunnel(SA, CT, p)" is available to be used if one wants to test if
    some of one's data lies outside this "funnel".


    Examples
    --------
    TODO

    References
    ----------
    .. [1] IOC, SCOR and IAPSO, 2010: The international thermodynamic equation
    of seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp. See Eqn. (2.17.1).

    .. [2] McDougall T.J., P.M. Barker, R. Feistel and D.R. Jackett, 2011:  A
    computationally efficient 48-term expression for the density of
    seawater in terms of Conservative Temperature, and related properties
    of seawater.

    Modifications:
    2011-03-23. Paul Barker and Trevor McDougall.
    """

    SA = np.maximum(SA, 0)

    dvden_dp = (c01 + CT * (c02 + c03 * CT) + SA * (c04 + c05 * CT) + p *
                (c06 + CT * (c07 + c08 * CT) + c09 * SA))

    dvnum_dp = (c10 + CT * (c11 + CT * (c12 + c13 * CT)) + SA *
               (c14 + c15 * CT) + p * (c16 + CT *
               (c17 + c18 * CT + c19 * SA) + p * (c20 + c21 * CT)))

    drho_dp = ((dvden_dp * v_hat_numerator(SA, CT, p) - dvnum_dp *
               v_hat_denominator(SA, CT, p)) / (v_hat_numerator(SA, CT, p) *
               v_hat_numerator(SA, CT, p)))

    return 100 * np.sqrt(1. / drho_dp)


@match_args_return
def internal_energy(SA, CT, p):
    r"""Calculates specific internal energy of seawater using the
    computationally-efficient 48-term expression for density in terms of SA,
    CT and p (McDougall et al., 2011).


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
    internal_energy : array_like
                      specific internal energy [J/kg]

    See Also
    --------
    TODO

    Notes
    -----
    The 48-term equation has been fitted in a restricted range of parameter
    space, and is most accurate inside the "oceanographic funnel"
    described in McDougall et al. (2011).  The GSW library function
    "infunnel(SA, CT, p)" is available to be used if one wants to test if
    some of one's data lies outside this "funnel".


    Examples
    --------
    TODO

    References
    ----------
    .. [1] IOC, SCOR and IAPSO, 2010: The international thermodynamic equation
    of seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.

    .. [2] McDougall T.J., P.M. Barker, R. Feistel and D.R. Jackett, 2011:  A
    computationally efficient 48-term expression for the density of
    seawater in terms of Conservative Temperature, and related properties
    of seawater.

    Modifications:
    2011-04-04. Trevor McDougall and Paul Barker.
    """

    SA = np.maximum(SA, 0)

    """This function calculates enthalpy using the computationally-efficient
    48-term expression for density in terms of SA, CT and p. If one wanted to
    compute enthalpy from SA, CT, and p with the full TEOS-10 Gibbs function,
    the following line of code will enable this.

    internal_energy = internal_energy_CT_exact(SA, CT, p)
    """

    return (enthalpy(SA, CT, p) - (P0 + db2Pascal * p) * specvol(SA, CT, p))


@match_args_return
def enthalpy(SA, CT, p):
    r"""Calculates specific enthalpy of seawater using the computationally-
    efficient 48-term expression for density in terms of SA, CT and p
    (McDougall et al., 2011)


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
    enthalpy : array_like
               specific enthalpy [J/kg]

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
    UNESCO (English), 196 pp. See Eqn. (A.30.6).

    .. [2] McDougall T.J., P.M. Barker, R. Feistel and D.R. Jackett, 2011:  A
    computationally efficient 48-term expression for the density of
    seawater in terms of Conservative Temperature, and related properties
    of seawater.

    Modifications:
    2011-04-05. Trevor McDougall, David Jackett, Claire Roberts-Thomson and
                Paul Barker.
    """

    SA = np.maximum(SA, 0)

    sqrtSA = np.sqrt(SA)

    a0 = (v21 + CT * (v22 + CT * (v23 + CT * (v24 + v25 * CT))) + SA *
         (v26 + CT * (v27 + CT * (v28 + CT * (v29 + v30 * CT))) + v36 * SA +
         sqrtSA * (v31 + CT * (v32 + CT * (v33 + CT * (v34 + v35 * CT))))))

    a1 = v37 + CT * (v38 + CT * (v39 + v40 * CT)) + SA * (v41 + v42 * CT)

    a2 = v43 + CT * (v44 + v45 * CT + v46 * SA)

    a3 = v47 + v48 * CT

    b0 = (v01 + CT * (v02 + CT * (v03 + v04 * CT)) + SA * (v05 + CT * (v06 +
    v07 * CT) + sqrtSA * (v08 + CT * (v09 + CT * (v10 + v11 * CT)))))

    b1 = 0.5 * (v12 + CT * (v13 + v14 * CT) + SA * (v15 + v16 * CT))

    b2 = v17 + CT * (v18 + v19 * CT) + v20 * SA

    b1sq = b1 * b1

    sqrt_disc = np.sqrt(b1sq - b0 * b2)

    N = a0 + (2 * a3 * b0 * b1 / b2 - a2 * b0) / b2
    M = a1 + (4 * a3 * b1sq / b2 - a3 * b0 - 2 * a2 * b1) / b2

    A = b1 - sqrt_disc
    B = b1 + sqrt_disc

    part = (N * b2 - M * b1) / (b2 * (B - A))

    """This function calculates enthalpy using the computationally-efficient
    48-term expression for density in terms of SA, CT and p.  If one wanted to
    compute enthalpy from SA, CT, and p with the full TEOS-10 Gibbs function,
    the following lines of code will enable this.

    pt = pt_from_CT(SA, CT)
    t = pt_from_t(SA, pt, 0, p)
    enthalpy = enthalpy_t_exact(SA, t, p)

    or call the following, it is identical to the lines above.

    enthalpy = enthalpy_CT_exact(SA, CT, p)
    """

    return (cp0 * CT + db2Pascal *
            (p * (a2 - 2 * a3 * b1 / b2 + 0.5 * a3 * p) / b2 + (M / (2 * b2)) *
             np.log(1 + p * (2 * b1 + b2 * p) / b0) + part *
             np.log(1 + (b2 * p * (B - A)) / (A * (B + b2 * p)))))


@match_args_return
def enthalpy_diff(SA, CT, p_shallow, p_deep):
    r"""Calculates the difference of the specific enthalpy of seawater between
    two different pressures, p_deep (the deeper pressure) and p_shallow (the
    shallower pressure), at the same values of SA and CT.  This function uses
    the computationally-efficient 48-term expression for density in terms of
    SA, CT and p (McDougall et al., 2011).  The output (enthalpy_diff_CT) is
    the specific enthalpy evaluated at (SA, CT, p_deep) minus the specific
    enthalpy at (SA, CT, p_shallow).



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

    Returns
    -------
    enthalpy_diff : array_like
                    difference of specific enthalpy [J/kg]
                    (deep minus shallow)

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
    UNESCO (English), 196 pp. See Eqns. (3.32.2) and (A.30.6).

    .. [2] McDougall T.J., P.M. Barker, R. Feistel and D.R. Jackett, 2011:  A
    computationally efficient 48-term expression for the density of
    seawater in terms of Conservative Temperature, and related properties
    of seawater.

    Modifications:
    2011-03-21. Trevor McDougall & Paul Barker.
    """

    SA = np.maximum(SA, 0)

    sqrtSA = np.sqrt(SA)

    a0 = (v21 + CT * (v22 + CT * (v23 + CT * (v24 + v25 * CT))) + SA *
         (v26 + CT * (v27 + CT * (v28 + CT * (v29 + v30 * CT))) + v36 * SA +
         sqrtSA * (v31 + CT * (v32 + CT * (v33 + CT * (v34 + v35 * CT))))))

    a1 = v37 + CT * (v38 + CT * (v39 + v40 * CT)) + SA * (v41 + v42 * CT)

    a2 = v43 + CT * (v44 + v45 * CT + v46 * SA)

    a3 = v47 + v48 * CT

    b0 = (v01 + CT * (v02 + CT * (v03 + v04 * CT)) + SA * (v05 + CT * (v06 +
    v07 * CT) + sqrtSA * (v08 + CT * (v09 + CT * (v10 + v11 * CT)))))

    b1 = 0.5 * (v12 + CT * (v13 + v14 * CT) + SA * (v15 + v16 * CT))

    b2 = v17 + CT * (v18 + v19 * CT) + v20 * SA

    b1sq = b1 * b1

    sqrt_disc = np.sqrt(b1sq - b0 * b2)

    N = a0 + (2 * a3 * b0 * b1 / b2 - a2 * b0) / b2
    M = a1 + (4 * a3 * b1sq / b2 - a3 * b0 - 2 * a2 * b1) / b2

    A = b1 - sqrt_disc
    B = b1 + sqrt_disc

    delta_p = p_deep - p_shallow
    p_sum = p_deep + p_shallow

    part1 = b0 + p_shallow * (2 * b1 + b2 * p_shallow)
    part2 = (B + b2 * p_deep) * (A + b2 * p_shallow)
    part3 = (N * b2 - M * b1) / (b2 * (B - A))

    """This function calculates enthalpy_diff using the computationally
    efficient 48-term expression for density in terms of SA, CT and p.  If one
    wanted to compute the enthalpy difference using the full TEOS-10 Gibbs
    function, the following lines of code will enable this.

    pt = pt_from_CT(SA, CT)
    t_shallow = pt_from_t(SA, pt, 0, p_shallow)
    t_deep = pt_from_t(SA, pt, 0, p_deep)
    enthalpy_diff = (enthalpy_t_exact(SA, t_deep, p_deep) -
                     enthalpy_t_exact(SA, t_shallow, p_shallow))

    or call the following, it is identical to the lines above.

    enthalpy_diff = enthalpy_diff_CT_exact(SA, CT, p_shallow, p_deep)
    """

    return (db2Pascal * (delta_p * (a2 - 2 * a3 * b1 / b2 + 0.5 * a3 * p_sum) /
           b2 + (M / (2 * b2)) * np.log(1 + delta_p * (2 * b1 + b2 * p_sum) /
           part1) + part3 * np.log(1 + delta_p * b2 * (B - A) / part2)))


@match_args_return
def dynamic_enthalpy(SA, CT, p):
    r"""Calculates dynamic enthalpy of seawater using the computationally-
    efficient 48-term expression for density in terms of SA, CT and p
    (McDougall et al., 2011).  Dynamic enthalpy is defined as enthalpy minus
    potential enthalpy (Young, 2010).


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
    dynamic_enthalpy : array_like
                       dynamic enthalpy [J/kg]

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
    UNESCO (English), 196 pp. See section 3.2

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

    SA = np.maximum(SA, 0)

    sqrtSA = np.sqrt(SA)

    a0 = (v21 + CT * (v22 + CT * (v23 + CT * (v24 + v25 * CT))) + SA *
         (v26 + CT * (v27 + CT * (v28 + CT * (v29 + v30 * CT))) + v36 * SA +
         sqrtSA * (v31 + CT * (v32 + CT * (v33 + CT * (v34 + v35 * CT))))))

    a1 = v37 + CT * (v38 + CT * (v39 + v40 * CT)) + SA * (v41 + v42 * CT)

    a2 = v43 + CT * (v44 + v45 * CT + v46 * SA)

    a3 = v47 + v48 * CT

    b0 = (v01 + CT * (v02 + CT * (v03 + v04 * CT)) + SA * (v05 + CT * (v06 +
    v07 * CT) + sqrtSA * (v08 + CT * (v09 + CT * (v10 + v11 * CT)))))

    b1 = 0.5 * (v12 + CT * (v13 + v14 * CT) + SA * (v15 + v16 * CT))

    b2 = v17 + CT * (v18 + v19 * CT) + v20 * SA

    b1sq = b1 * b1

    sqrt_disc = np.sqrt(b1sq - b0 * b2)

    N = a0 + (2 * a3 * b0 * b1 / b2 - a2 * b0) / b2
    M = a1 + (4 * a3 * b1sq / b2 - a3 * b0 - 2 * a2 * b1) / b2

    A = b1 - sqrt_disc
    B = b1 + sqrt_disc

    part = (N * b2 - M * b1) / (b2 * (B - A))

    """This function calculates dynamic_enthalpy using the computationally-
    efficient 48-term expression for density in terms of SA, CT and p.  If one
    wanted to compute dynamic_enthalpy from SA, CT, and p with the full TEOS-10
    Gibbs function, the following lines of code will enable this.

    dynamic_enthalpy = dynamic_enthalpy_CT_exact(SA, CT, p)
    """

    return db2Pascal * (p * (a2 - 2 * a3 * b1 / b2 + 0.5 * a3 * p) / b2 +
           (M / (2 * b2)) * np.log(1 + p * (2 * b1 + b2 * p) / b0) + part *
           np.log(1 + (b2 * p * (B - A)) / (A * (B + b2 * p))))


@match_args_return
def SA_from_rho(rho, CT, p):
    r"""Calculates the Absolute Salinity of a seawater sample, for given values
    of its density, Conservative Temperature and sea pressure (in dbar).  This
    function uses the computationally-efficient 48-term expression for density
    in terms of SA, CT and p (McDougall et al., 2011).

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
    This is expressed on the Reference-Composition Salinity Scale of
    Millero et al. (2008).

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
    UNESCO (English), 196 pp. See section 2.5

    .. [2] McDougall T.J., P.M. Barker, R. Feistel and D.R. Jackett, 2011:  A
    computationally efficient 48-term expression for the density of
    seawater in terms of Conservative Temperature, and related properties
    of seawater.

    .. [3] Millero, F. J., R. Feistel, D. G. Wright, and T. J. McDougall, 2008:
    The composition of Standard Seawater and the definition of the
    Reference-Composition Salinity Scale, Deep-Sea Res. I, 55, 50-72.

    Modifications:
    2011-04-04. Trevor McDougall and Paul Barker.
    """

    v_lab = 1. / rho
    v_0 = specvol(np.zeros_like(rho), CT, p)
    v_50 = specvol(50 * np.ones_like(rho), CT, p)

    SA = 50 * (v_lab - v_0) / (v_50 - v_0)  # Initial estimate of SA.

    SA[np.logical_or(SA < 0, SA > 50)] = np.NaN

    v_SA = (v_50 - v_0) / 50.  # Initial v_SA estimate (SA derivative of v).

    # Begin the modified Newton-Raphson iterative procedure.
    for Number_of_iterations in range(0, 3):
        SA_old = SA
        delta_v = specvol(SA_old, CT, p) - v_lab
        # Half way the mod. N-R method (McDougall and Wotherspoon, 2012)
        SA = SA_old - delta_v / v_SA  # Half way through the mod. N-R method.
        SA_mean = 0.5 * (SA + SA_old)
        rho, alpha, beta = rho_alpha_beta(SA_mean, CT, p)
        v_SA = -beta / rho
        SA = SA_old - delta_v / v_SA
        SA[np.logical_or(SA < 0, SA > 50)] = np.NaN

    # After two iterations of this modified Newton-Raphson iteration,
    # the error in SA is no larger than 8x10^-13 g kg^-1, which
    # is machine precision for this calculation.
    return SA
