# -*- coding: utf-8 -*-

"""Constants"""

from __future__ import division

C3515 = 42.9140
r"""Conductivity of 42.914 [mmho cm :sup:`-1` == mS cm :sup:`-1`] at Salinity
35 psu, Temperature 15 :math:`^\\circ` C [ITPS 68] and Pressure 0 db.

References
----------
.. [1] Culkin and Smith, 1980:  Determination of the Concentration of Potassium
Chloride Solution Having the Same Electrical Conductivity, at 15C and Infinite
Frequency, as Standard Seawater of Salinity 35.0000 (Chlorinity 19.37394),
IEEE J. Oceanic Eng, 5, 22-23.

.. [2] Unesco, 1983: Algorithms for computation of fundamental properties of
seawater. Unesco Technical Papers in Marine Science, 44, 53 pp."""

earth_radius = 6371000.
r"""Mean radius of earth  A.E. Gill."""

OMEGA = 7.292115e-5
r""":math:`\\Omega = \\frac{2\\pi}{\\textrm{sidereal day}}` =
                                                 7.292e-5.radians sec :sup:`-1`

1 sidereal day = 23.9344696 hours

Changed to a more precise value at Groten 2004

References
----------
.. [1] A.E. Gill 1982. p.54  eqn 3.7.15 "Atmosphere-Ocean Dynamics" Academic
Press: New York. ISBN: 0-12-283522-0. page: 597
.. [2] Groten, E., 2004: Fundamental Parameters and Current (2004) Best
Estimates of the Parameters of Common Relevance to Astronomy, Geodesy, and
Geodynamics. Journal of Geodesy, 77, pp. 724-797."""

gdef = 9.8
r"""Acceleration of gravity [m s :sup:`2`] used by sw.swvel and bfrq without
lat info."""

DEG2NM, NM2KM = 60., 1.8520
r"""1 nm = 1.8520 km

Used by sw.dist() to convert nautical miles to kilometers.

References
----------
.. [1] S. Pond & G.Pickard 2nd Edition 1986 Introductory Dynamical
Oceanography Pergamon Press Sydney. ISBN 0-08-028728-X. page: 303."""

T0 = Kelvin = 273.15
r"""The Celsius zero point; 273.15 K.  That is T = t + T0 where T is the
Absolute Temperature (in degrees K) and t is temperature in degrees C."""

db2Pascal = 1e4
r"""Decibar to pascal."""

gamma = 2.26e-7
r"""Gamma (A.E. Gill)."""

M_S = 0.0314038218
r"""Mole-weighted average atomic weight of the elements of
Reference-Composition sea salt, in units of kg mol :sup:`-1`. Strictly
speaking, the formula below applies only to seawater of Reference Composition.
If molality is required to an accuracy of better than 0.1% we suggest you
contact the authors for further guidance."""

cp0 = 3991.86795711963
r"""The "specific heat" for use with Conservative Temperature. cp0 is the ratio
of potential enthalpy to Conservative Temperature.
See Eqn. (3.3.3) and Table D.5 from IOC et al. (2010)."""

SSO = 35.16504
r"""SSO is the Standard Ocean Reference Salinity (35.16504 g/kg.)

SSO is the best estimate of the Absolute Salinity of Standard Seawater
when the seawater sample has a Practical Salinity, SP, of 35
(Millero et al., 2008), and this number is a fundamental part of the
TEOS-10 definition of seawater.

References:
-----------
.. [1] IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
seawater - 2010: Calculation and use of thermodynamic properties.
Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
UNESCO (English), 196 pp. See appendices A.3, A.5 and Table D.4.

.. [2] Millero, F. J., R. Feistel, D. G. Wright, and T. J. McDougall, 2008:
The composition of Standard Seawater and the definition of the
Reference-Composition Salinity Scale, Deep-Sea Res. I, 55, 50-72.
See Table 4 and section 5."""

sfac = 0.0248826675584615
r"""sfac = 1 / (40. * (SSO / 35.))"""

R = 8.314472
r"""The molar gas constant = 8.314472 m :sup:`2` kg s:sup:`-21 K :sup:`-1`
mol :sup:`-1`."""

r1 = 0.35
""" TODO """

uPS = SSO / 35.0
r"""The unit conversion factor for salinities (35.16504/35) g/kg (Millero et
al., 2008). Reference Salinity SR is uPS times Practical Salinity SP.

Ratio, unit conversion factor for salinities [g kg :sup:`-1`]

References
----------
Millero, F. J., R. Feistel, D. G. Wright, and T. J. McDougall, 2008: The
composition of Standard Seawater and the definition of the
Reference-Composition Salinity Scale, Deep-Sea Res. I, 55, 50-72. See
section 6, Eqn.(6.1)."""


P0 = 101325
r"""Absolute Pressure of one standard atmosphere in Pa, 101325 Pa."""

SonCl = 1.80655
r"""The ratio of Practical Salinity, SP, to Chlorinity, 1.80655 kg/g for
Reference Seawater (Millero et al., 2008). This is the ratio that was used by
the JPOTS committee in their construction of the 1978 Practical Salinity Scale
(PSS-78) to convert between the laboratory measurements of seawater samples
(which were measured in Chlorinity) to Practical Salinity.

References:
-----------
.. [1] Millero, F. J., R. Feistel, D. G. Wright, and T. J. McDougall, 2008:
The composition of Standard Seawater and the definition of the
Reference-Composition Salinity Scale, Deep-Sea Res. I, 55, 50-72. See section
5 below Eqn. (5.5)."""

atomic_weight = 31.4038218
r"""This function returns the mole-weighted atomic weight of sea salt of
Reference Composition, which is 31.4038218 g/mol.  This has been
defined as part of the Reference-Composition Salinity Scale of 2008
(Millero et al., 2008).

References:
-----------
.. [1] IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
seawater - 2010: Calculation and use of thermodynamic properties.
Intergovernmental Oceanographic Commission, Manuals and Guides No. 56, UNESCO
(English), 196 pp. See Table D.4 of this TEOS-10 Manual.

.. [2] Millero, F. J., R. Feistel, D. G. Wright, and T. J. McDougall, 2008:
The composition of Standard Seawater and the definition of the
Reference-Composition Salinity Scale, Deep-Sea Res. I, 55, 50-72.
See Eqn. (5.3)"""

valence_factor = 1.2452898
r"""This function returns the valence factor of sea salt of Reference
Composition, 1.2452898.  This valence factor is exact, and follows from
the definition of the Reference-Composition Salinity Scale 2008 of
Millero et al. (2008).  The valence factor is the mole-weighted square
of the charges, Z, of the ions comprising Reference Composition sea salt.

References:
-----------
.. [1] IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
seawater - 2010: Calculation and use of thermodynamic properties.
Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
UNESCO (English), 196 pp. See Table D.4 of this TEOS-10 Manual.

.. [2] Millero, F. J., R. Feistel, D. G. Wright, and T. J. McDougall, 2008:
The composition of Standard Seawater and the definition of the
Reference-Composition Salinity Scale, Deep-Sea Res. I, 55, 50-72.
See Eqn. (5.9)."""
