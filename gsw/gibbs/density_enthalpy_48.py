# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np

from gsw.utilities import match_args_return

__all__ = [
           'rho',
           #'alpha',  TODO
           #'beta',  TODO
           #'rho_alpha_beta',  TODO
           #'specvol',  TODO
           #'specvol_anom',  TODO
           #'sigma0',  TODO
           #'sigma1',  TODO
           #'sigma2',  TODO
           #'sigma3',  TODO
           #'sigma4',  TODO
           #'sound_speed',  TODO
           #'internal_energy',  TODO
           #'enthalpy',  TODO
           #'enthalpy_diff',  TODO
           #'dynamic_enthalpy',  TODO
           #'SA_from_rho',  TODO
           ]


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

    v_hat_denominator = (v01 + CT * (v02 + CT * (v03 + v04 * CT)) +
                   SA * (v05 + CT * (v06 + v07 * CT) +
                   np.sqrt(SA) * (v08 + CT * (v09 + CT * (v10 + v11 * CT)))) +
                   p * (v12 + CT * (v13 + v14 * CT) + SA * (v15 + v16 * CT) +
                   p * (v17 + CT * (v18 + v19 * CT) + v20 * SA)))

    v_hat_numerator = (v21 + CT * (v22 + CT * (v23 + CT * (v24 + v25 * CT))) +
      SA * (v26 + CT * (v27 + CT * (v28 + CT * (v29 + v30 * CT))) + v36 * SA +
      np.sqrt(SA) * (v31 + CT * (v32 + CT * (v33 + CT * (v34 + v35 * CT))))) +
      p * (v37 + CT * (v38 + CT * (v39 + v40 * CT)) +
      SA * (v41 + v42 * CT) +
      p * (v43 + CT * (v44 + v45 * CT + v46 * SA) +
      p * (v47 + v48 * CT))))

    """This function calculates rho using the computationally-efficient 48-term
    expression for density in terms of SA, CT and p. If one wanted to compute
    rho from SA, CT, and p with the full TEOS-10 Gibbs function, the following
    lines of code will enable this.

    pt0 = pt_from_CT(SA, CT)
    pr0 = np.zeros_like(SA)
    t = pt_from_t(SA, pt0, pr0, p)
    rho = rho_t_exact(SA, t, p)

    or call the following, it is identical to the lines above.

    rho = rho_CT_exact(SA, CT, p)

    or call the following, it is identical to the lines above.

    [rho, ~, ~] = rho_alpha_beta_CT_exact(SA, CT, p)
    """

    return v_hat_denominator / v_hat_numerator
