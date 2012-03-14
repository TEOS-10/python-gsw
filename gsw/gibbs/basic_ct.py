# -*- coding: utf-8 -*-

"""
Basic thermodynamic properties in terms of CT

Functions:
  sound_speed_CT_exact(SA, CT, p)
      sound speed

This is part of the python Gibbs Sea Water library
http://code.google.com/p/python-gsw.

"""

from __future__ import division

from library import match_args_return
from basicsa_t_p import sound_speed_t_exact

__all__ = [
           #'rho_CT',
           #'alpha_CT',
           #'beta_CT',
           #'rho_alpha_beta_CT',
           #'specvol_CT',
           #'specvol_anom_CT',
           #'sigma0_CT',
           #'sigma1_CT',
           #'sigma2_CT',
           #'sigma3_CT',
           #'sigma4_CT',
           #'enthalpy_CT',
           #'enthalpy_diff_CT',
           #'entropy_from_pt',
           #'entropy_from_CT',
           #'pt_from_entropy',
           #'CT_from_entropy'
           'sound_speed_CT_exact'
           ]


@match_args_return
def sound_speed_CT_exact(SA, CT, p):
    r"""
    Calculates the speed of sound in gsw.from Absolute Salinity and
    Conservative Temperature and pressure.

    Parameters
    ----------
    SA : array_like
         Absolute salinity [g kg :sup:`-1`]
    CT : array_like
        in Conservative Temperature [:math:`^\circ` C (ITS-90)]
    p : array_like
        pressure [dbar]

    Returns
    -------
    sound_speed_CT_exact : array_like
    Speed of sound in gsw.[m s :sup:`-1`]


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
    of gsw.- 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp. See Eqn. (2.17.1).

    Modifications:
    2011-04-05. David Jackett, Paul Barker and Trevor McDougall.
    """

    t = t_from_CT(SA, CT, p)

    return  sound_speed_t_exact(SA, t, p)
