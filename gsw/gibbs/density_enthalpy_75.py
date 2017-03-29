# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np

from .constants import sfac, soffset
from ..utilities import match_args_return

__all__ = ['alpha',
           'rho',
           'specvol']


a000 = -1.5649734675e-5
a001 =  1.8505765429e-5
a002 = -1.1736386731e-6
a003 = -3.6527006553e-7
a004 =  3.1454099902e-7
a010 =  5.5524212968e-5
a011 = -2.3433213706e-5
a012 =  4.2610057480e-6
a013 =  5.7391810318e-7
a020 = -4.9563477777e-5
a021 =  2.37838968519e-5
a022 = -1.38397620111e-6
a030 =  2.76445290808e-5
a031 = -1.36408749928e-5
a032 = -2.53411666056e-7
a040 = -4.0269807770e-6
a041 =  2.5368383407e-6
a050 =  1.23258565608e-6
a100 =  3.5009599764e-5
a101 = -9.5677088156e-6
a102 = -5.5699154557e-6
a103 = -2.7295696237e-7
a110 = -7.4871684688e-5
a111 = -4.7356616722e-7
a112 =  7.8274774160e-7
a120 =  7.2424438449e-5
a121 = -1.03676320965e-5
a122 =  2.32856664276e-8
a130 = -3.50383492616e-5
a131 =  5.1826871132e-6
a140 = -1.6526379450e-6
a200 = -4.3592678561e-5
a201 =  1.1100834765e-5
a202 =  5.4620748834e-6
a210 =  7.1815645520e-5
a211 =  5.8566692590e-6
a212 = -1.31462208134e-6
a220 = -4.3060899144e-5
a221 =  9.4965918234e-7
a230 =  1.74814722392e-5
a300 =  3.4532461828e-5
a301 = -9.8447117844e-6
a302 = -1.3544185627e-6
a310 = -3.7397168374e-5
a311 = -9.7652278400e-7
a320 =  6.8589973668e-6
a400 = -1.1959409788e-5
a401 =  2.5909225260e-6
a410 =  7.7190678488e-6
a500 =  1.3864594581e-6

v000 =  1.0769995862e-3
v001 = -6.0799143809e-5
v002 =  9.9856169219e-6
v003 = -1.1309361437e-6
v004 =  1.0531153080e-7
v005 = -1.2647261286e-8
v006 =  1.9613503930e-9
v010 = -1.5649734675e-5
v011 =  1.8505765429e-5
v012 = -1.1736386731e-6
v013 = -3.6527006553e-7
v014 =  3.1454099902e-7
v020 =  2.7762106484e-5
v021 = -1.1716606853e-5
v022 =  2.1305028740e-6
v023 =  2.8695905159e-7
v030 = -1.6521159259e-5
v031 =  7.9279656173e-6
v032 = -4.6132540037e-7
v040 =  6.9111322702e-6
v041 = -3.4102187482e-6
v042 = -6.3352916514e-8
v050 = -8.0539615540e-7
v051 =  5.0736766814e-7
v060 =  2.0543094268e-7
v100 = -3.1038981976e-4
v101 =  2.4262468747e-5
v102 = -5.8484432984e-7
v103 =  3.6310188515e-7
v104 = -1.1147125423e-7
v110 =  3.5009599764e-5
v111 = -9.5677088156e-6
v112 = -5.5699154557e-6
v113 = -2.7295696237e-7
v120 = -3.7435842344e-5
v121 = -2.3678308361e-7
v122 =  3.9137387080e-7
v130 =  2.4141479483e-5
v131 = -3.4558773655e-6
v132 =  7.7618888092e-9
v140 = -8.7595873154e-6
v141 =  1.2956717783e-6
v150 = -3.3052758900e-7
v200 =  6.6928067038e-4
v201 = -3.4792460974e-5
v202 = -4.8122251597e-6
v203 =  1.6746303780e-8
v210 = -4.3592678561e-5
v211 =  1.1100834765e-5
v212 =  5.4620748834e-6
v220 =  3.5907822760e-5
v221 =  2.9283346295e-6
v222 = -6.5731104067e-7
v230 = -1.4353633048e-5
v231 =  3.1655306078e-7
v240 =  4.3703680598e-6
v300 = -8.5047933937e-4
v301 =  3.7470777305e-5
v302 =  4.9263106998e-6
v310 =  3.4532461828e-5
v311 = -9.8447117844e-6
v312 = -1.3544185627e-6
v320 = -1.8698584187e-5
v321 = -4.8826139200e-7
v330 =  2.2863324556e-6
v400 =  5.8086069943e-4
v401 = -1.7322218612e-5
v402 = -1.7811974727e-6
v410 = -1.1959409788e-5
v411 =  2.5909225260e-6
v420 =  3.8595339244e-6
v500 = -2.1092370507e-4
v501 =  3.0927427253e-6
v510 =  1.3864594581e-6
v600 =  3.1932457305e-5


@match_args_return
def alpha(SA, CT, p):
    """
    Thermal expansion coefficient from CT (75-term equation)

    Calculates the thermal expansion coefficient of seawater with respect to
    Conservative Temperature using the computationally-efficient expression
    for specific volume in terms of SA, CT and p (Roquet et al., 2015).

    Parameters
    ----------
    SA : array_like
         Absolute Salinity  [g/kg]
    CT : array_like
         Conservative Temperature [:math:`^\circ` C (ITS-90)]
    p : array_like
        sea pressure ( i.e. absolute pressure - 10.1325 dbar ) [dbar]

    Returns
    -------
    alpha : array_like
            thermal expansion coefficient [K :math:`-1`]
            with respect to Conservative Temperature

    Examples
    --------
    >>> import gsw
    >>> SA = [34.7118, 34.8915, 35.0256, 34.8472, 34.7366, 34.7324]
    >>> CT = [28.8099, 28.4392, 22.7861, 10.2261, 6.8272, 4.3235]
    >>> p = [10, 50, 125, 250, 600, 1000]
    >>> gsw.alpha(SA, CT, p)
    array([ 0.00032464,  0.00032266,  0.00028114,  0.0001732 ,  0.00014629,
            0.00012941])

    Notes
    -----
    Note that this 75-term equation has been fitted in a restricted range of
    parameter space, and is most accurate inside the "oceanographic funnel"
    described in McDougall et al. (2003).  The GSW library function
    "infunnel(SA,CT,p)" is avaialble to be used if one wants to test if some of
    one's data lies outside this "funnel".

    References
    ----------
    .. [1] IOC, SCOR and IAPSO, 2010: The international thermodynamic equation
       of seawater - 2010: Calculation and use of thermodynamic properties.
       Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
       UNESCO (English), 196 pp. Available from http://www.TEOS-10.org
       See Eqn. (2.18.3) of this TEOS-10 manual.

    .. [2] McDougall, T.J., D.R. Jackett, D.G. Wright and R. Feistel, 2003:
       Accurate and computationally efficient algorithms for potential
       temperature and density of seawater.  J. Atmosph. Ocean. Tech., 20,
       pp. 730-741.

    .. [3] Roquet, F., G. Madec, T.J. McDougall, P.M. Barker, 2015: Accurate
       polynomial expressions for the density and specifc volume of seawater
       using the TEOS-10 standard. Ocean Modelling.
    """

    SA = np.maximum(SA, 0)

    xs = np.sqrt(sfac * SA + soffset)
    ys = CT * 0.025
    z = p * 1e-4

    v_CT_part = (a000
        + xs * (a100 + xs * (a200 + xs * (a300 + xs * (a400 + a500 * xs))))
        + ys * (a010 + xs * (a110 + xs * (a210 + xs * (a310 + a410 * xs)))
            + ys * (a020 + xs * (a120 + xs * (a220 + a320 * xs))
                + ys * (a030 + xs * (a130 + a230 * xs)
                    + ys * (a040 + a140 * xs + a050 * ys))))
        + z * (a001 + xs * (a101 + xs * (a201 + xs * (a301 + a401 * xs)))
            + ys * (a011 + xs * (a111 + xs * (a211 + a311 * xs))
                + ys * (a021 + xs * (a121 + a221 * xs)
                    + ys * (a031 + a131 * xs + a041 * ys)))
            + z * (a002 + xs * (a102 + xs * (a202 + a302 * xs))
                + ys * (a012 + xs * (a112 + a212 * xs)
                    + ys * (a022 + a122 * xs + a032 * ys))
                + z * (a003 + a103 * xs + a013 * ys + a004 * z))))

    specific_volume = (v000
        + xs * (v100 + xs * (v200 + xs * (v300 + xs * (v400 + xs * (v500
            + xs * v600)))))
        + ys * (v010
            + xs * (v110 + xs * (v210 + xs * (v310 + xs * (v410 + xs * v510))))
            + ys * (v020 + xs * (v120 + xs * (v220 + xs * (v320 + xs * v420)))
                + ys * (v030 + xs * (v130 + xs * (v230 + xs * v330))
                    + ys * (v040 + xs * (v140 + xs * v240)
                        + ys * (v050 + xs * v150 + ys * v060)))))
        + z * (v001
            + xs * (v101 + xs * (v201 + xs * (v301 + xs * (v401 + xs * v501))))
            + ys * (v011 + xs * (v111 + xs * (v211 + xs * (v311 + xs * v411)))
                + ys * (v021 + xs * (v121 + xs * (v221 + xs * v321))
                    + ys * (v031 + xs * (v131 + xs * v231)
                        + ys * (v041 + xs * v141 + ys * v051))))
            + z * (v002
                + xs * (v102 + xs * (v202 + xs * (v302 + xs * v402)))
                + ys * (v012 + xs * (v112 + xs * (v212 + xs * v312))
                    + ys * (v022 + xs * (v122 + xs * v222)
                        + ys * (v032 + xs * v132 + ys * v042)))
                + z * (v003
                    + xs * (v103 + xs * v203)
                    + ys * (v013 + xs * v113 + ys * v023)
                    + z * (v004 + xs * v104 + ys * v014
                        + z * (v005 + z * v006))))))

    return 0.025 * v_CT_part / specific_volume


@match_args_return
def rho(SA, CT, p):
    """
    In-situ density from SA, CT & p (75-term equation)

    Calculates in-situ density from Absolute Salinity and Conservative
    Temperature, using the computationally-efficient expression for
    specific volume in terms of SA, CT and p  (Roquet et al., 2015).

    Parameters
    ----------
    SA : array_like
         Absolute Salinity  [g/kg]
    CT : array_like
         Conservative Temperature [:math:`^\circ` C (ITS-90)]
    p : array_like
        sea pressure ( i.e. absolute pressure - 10.1325 dbar ) [dbar]

    Returns
    -------
    rho : array_like
          in-situ density [kg/m**3]

    Examples
    --------
    >>> import gsw
    >>> SA = [34.7118, 34.8915, 35.0256, 34.8472, 34.7366, 34.7324]
    >>> CT = [28.8099, 28.4392, 22.7861, 10.2261, 6.8272, 4.3235]
    >>> p = [10, 50, 125, 250, 600, 1000]
    >>> gsw.rho(SA, CT, p)
    array([ 1021.83993574,  1022.26245797,  1024.42722421,  1027.79017056,
            1029.837779  ,  1032.00246658])

    Notes
    -----
    The potential density with respect to reference pressure, pr, is obtained
    by calling this function with the pressure argument being pr (i.e.
    "rho(SA,CT,pr)").

    Note that this 75-term equation has been fitted in a restricted range of
    parameter space, and is most accurate inside the "oceanographic funnel"
    described in McDougall et al. (2003).  The GSW library function
    "infunnel(SA,CT,p)" is avaialble to be used if one wants to test if some of
    one's data lies outside this "funnel".

    References
    ----------
    .. [1] IOC, SCOR and IAPSO, 2010: The international thermodynamic equation
       of seawater - 2010: Calculation and use of thermodynamic properties.
       Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
       UNESCO (English), 196 pp. Available from http://www.TEOS-10.org
       See Eqn. (2.18.3) of this TEOS-10 manual.

    .. [2] McDougall, T.J., D.R. Jackett, D.G. Wright and R. Feistel, 2003:
       Accurate and computationally efficient algorithms for potential
       temperature and density of seawater.  J. Atmosph. Ocean. Tech., 20,
       pp. 730-741.

    .. [3] Roquet, F., G. Madec, T.J. McDougall, P.M. Barker, 2015: Accurate
       polynomial expressions for the density and specifc volume of seawater
       using the TEOS-10 standard. Ocean Modelling.
    """

    SA = np.maximum(SA, 0)

    xs = np.sqrt(sfac * SA + soffset)
    ys = CT * 0.025
    z = p * 1e-4

    specific_volume = (v000
        + xs * (v100 + xs * (v200 + xs * (v300 + xs * (v400 + xs * (v500
            + xs * v600)))))
        + ys * (v010
            + xs * (v110 + xs * (v210 + xs * (v310 + xs * (v410 + xs * v510))))
            + ys * (v020 + xs * (v120 + xs * (v220 + xs * (v320 + xs * v420)))
                + ys * (v030 + xs * (v130 + xs * (v230 + xs * v330))
                    + ys * (v040 + xs * (v140 + xs * v240)
                        + ys * (v050 + xs * v150 + ys * v060)))))
        + z * (v001
            + xs * (v101 + xs * (v201 + xs * (v301 + xs * (v401 + xs * v501))))
            + ys * (v011 + xs * (v111 + xs * (v211 + xs * (v311 + xs * v411)))
                + ys * (v021 + xs * (v121 + xs * (v221 + xs * v321))
                    + ys * (v031 + xs * (v131 + xs * v231)
                        + ys * (v041 + xs * v141 + ys * v051))))
            + z * (v002
                + xs * (v102 + xs * (v202 + xs * (v302 + xs * v402)))
                + ys * (v012 + xs * (v112 + xs * (v212 + xs * v312))
                    + ys * (v022 + xs * (v122 + xs * v222)
                        + ys * (v032 + xs * v132 + ys * v042)))
                + z * (v003
                    + xs * (v103 + xs * v203)
                    + ys * (v013 + xs * v113 + ys * v023)
                    + z * (v004 + xs * v104 + ys * v014
                        + z * (v005 + z * v006))))))

    return 1. / specific_volume


@match_args_return
def specvol(SA, CT, p):
    """
    Specific volume from SA, CT & p (75-term equation)

    Calculates specific volume from Absolute Salinity, Conservative
    Temperature and pressure, using the computationally-efficient 75-term
    polynomial expression for specific volume (Roquet et al., 2015).

    Parameters
    ----------
    SA : array_like
         Absolute Salinity  [g/kg]
    CT : array_like
         Conservative Temperature [:math:`^\circ` C (ITS-90)]
    p : array_like
        sea pressure ( i.e. absolute pressure - 10.1325 dbar ) [dbar]

    Returns
    -------
    specvol : array_like
              in-situ density [m**3/kg]

    Examples
    --------
    >>> import gsw
    >>> SA = [34.7118, 34.8915, 35.0256, 34.8472, 34.7366, 34.7324]
    >>> CT = [28.8099, 28.4392, 22.7861, 10.2261, 6.8272, 4.3235]
    >>> p = [10, 50, 125, 250, 600, 1000]
    >>> gsw.specvol(SA, CT, p)
    array([ 0.00097863,  0.00097822,  0.00097616,  0.00097296,  0.00097103,
            0.00096899])

    Notes
    -----
    Note that this 75-term equation has been fitted in a restricted range of
    parameter space, and is most accurate inside the "oceanographic funnel"
    described in McDougall et al. (2003).  The GSW library function
    "infunnel(SA,CT,p)" is avaialble to be used if one wants to test if some of
    one's data lies outside this "funnel".

    References
    ----------
    .. [1] IOC, SCOR and IAPSO, 2010: The international thermodynamic equation
       of seawater - 2010: Calculation and use of thermodynamic properties.
       Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
       UNESCO (English), 196 pp. Available from http://www.TEOS-10.org
       See Eqn. (2.18.3) of this TEOS-10 manual.

    .. [2] McDougall, T.J., D.R. Jackett, D.G. Wright and R. Feistel, 2003:
       Accurate and computationally efficient algorithms for potential
       temperature and density of seawater.  J. Atmosph. Ocean. Tech., 20,
       pp. 730-741.

    .. [3] Roquet, F., G. Madec, T.J. McDougall, P.M. Barker, 2015: Accurate
       polynomial expressions for the density and specifc volume of seawater
       using the TEOS-10 standard. Ocean Modelling.
    """

    SA = np.maximum(SA, 0)

    xs = np.sqrt(sfac * SA + soffset)
    ys = CT * 0.025
    z = p * 1e-4

    specific_volume = (v000
        + xs * (v100 + xs * (v200 + xs * (v300 + xs * (v400 + xs * (v500
            + xs * v600)))))
        + ys * (v010
            + xs * (v110 + xs * (v210 + xs * (v310 + xs * (v410 + xs * v510))))
            + ys * (v020 + xs * (v120 + xs * (v220 + xs * (v320 + xs * v420)))
                + ys * (v030 + xs * (v130 + xs * (v230 + xs * v330))
                    + ys * (v040 + xs * (v140 + xs * v240)
                        + ys * (v050 + xs * v150 + ys * v060)))))
        + z * (v001
            + xs * (v101 + xs * (v201 + xs * (v301 + xs * (v401 + xs * v501))))
            + ys * (v011 + xs * (v111 + xs * (v211 + xs * (v311 + xs * v411)))
                + ys * (v021 + xs * (v121 + xs * (v221 + xs * v321))
                    + ys * (v031 + xs * (v131 + xs * v231)
                        + ys * (v041 + xs * v141 + ys * v051))))
            + z * (v002
                + xs * (v102 + xs * (v202 + xs * (v302 + xs * v402)))
                + ys * (v012 + xs * (v112 + xs * (v212 + xs * v312))
                    + ys * (v022 + xs * (v122 + xs * v222)
                        + ys * (v032 + xs * v132 + ys * v042)))
                + z * (v003
                    + xs * (v103 + xs * v203)
                    + ys * (v013 + xs * v113 + ys * v023)
                    + z * (v004 + xs * v104 + ys * v014
                        + z * (v005 + z * v006))))))

    return specific_volume


if __name__ == '__main__':
    import doctest
    doctest.testmod()
