# -*- coding: utf-8 -*-

# TODO: Move some of these to the README.txt
"""CAVEAT: These modules do not adhere to strict basic-SI units but rather
oceanographic units are adopted.

Sea pressure is absolute pressure - 10.1325 dbar (or minus atmospheric
pressure)

For the first time the influence of the spatially varying composition of
seawater can be systematically taken into account through the use of Absolute
Salinity. In the open ocean, this has a non-trivial effect on the horizontal
density gradient, and thereby on the ocean velocities and heat transports
calculated via the thermal wind relationship.

The new salinity variable, Absolute Salinity, is measured in SI units
(e.g. g/kg).

The thermodynamic quantities available from TEOS-10 are totally consistent
with each other, while this was not the case with EOS-80

The new equation incorporates a more accurate representation of salinity known
as Absolute Salinity. The main justification for preferring Absolute Salinity
over the most recent salinity definition, Practical Salinity, is that the
thermodynamic properties of seawater are directly influenced by the total mass
of dissolved constituents (Absolute Salinity). However, the mass of dissolved
constituents are regionally variable and are not always accurately represented
when using conductivity measurements of seawater, the key parameter used in
the calculation of Practical Salinity.

In addition, it would be useful to have a conservative tracer for salinity.
The Absolute Salinity SA is not conservative, because it slowly increases in
the ocean due to biogeochemical processes, which remineralize carbon and
nutrients. The Reference Salinity SR is also not conservative, since it is
measured with conductivity which will also slowly increase due to these
increases in concentrations of nutrient and carbon system ions.

However, we can "remove" these effects and construct a conservative salinity
tracer Sstar..

Typically this PREFORMED SALINITY Sstar will equal SR (and SA) in Standard
Seawater (SSW), but as ions are added it will become smaller than either SR
and SA (as illustrated by the example in Table 1).

CT == \BigTheta = frac{h_o}{C^o_p}.

The scale factor C^o_p is a constant carefully chosen so that potential
temperature \smalltheta and Conservative Temperature \BigTheta will be
numerically similar for typical seawater at SP = 35, or near t = 0 degC.
However, the difference between the two can exceed 1degC when salinities are
low and temperatures high (for details, see IOC et al., 2010).
"""

__authors__ = ['Eric Firing', 'Bjørn Ådlandsvik', 'Filipe Fernandes']
__license__ = ["MIT"]
__version__ = ["3.0.0"]
__maintainer__ = ["Filipe Fernandes"]
__email__ = ["ocefpaf@gmail.com"]
__status__ = ["Production"]
__created_ = ["14-Jan-2010"]
__modified__ = ["19-Feb-2011"]
__all__ = ["gsw"]

from gibbs import *
