==========
python gsw
==========

Python implementation of the Thermodynamic Equation Of Seawater - 2010 (TEOS-10)
--------------------------------------------------------------------------------

For more information go to:
    http://www.teos-10.org/


gsw vs. csiro
^^^^^^^^^^^^^

.. role:: raw-math(raw)
    :format: latex html

This table shows some function names in the gibbs library and the corresponding function names in the csiro library.

+-------------------------------------------+-------------------------------------+-------------------------------------------------------+
| **Variable**                              | **SeaWater (EOS 80)**               | **Gibbs SeaWater (GSW TEOS 10)**                      |
+-------------------------------------------+-------------------------------------+-------------------------------------------------------+
| Absolute Salinity                         |          NA                         | gsw.SA_from_SP(SP,p,long,lat)                         |
+-------------------------------------------+-------------------------------------+-------------------------------------------------------+
| Conservative Temperature                  |          NA                         | gsw.CT_from_t(SA,t,p)                                 |
+-------------------------------------------+-------------------------------------+-------------------------------------------------------+
| density (i.e. in situ density)            |  sw.dens(SP,t,p)                    | gsw.rho_CT(SA,CT,p), or gsw.rho(SA,t,p), or           |
|                                           |                                     | gsw.rho_CT25(SA,CT,p)                                 |
+-------------------------------------------+-------------------------------------+-------------------------------------------------------+
| potential density                         |  sw.pden(SP,t,p,pr)                 | gsw.rho_CT(SA,CT,pr), or                              |
|                                           |                                     | gsw.rho_CT25(SA,CT,pr)                                |
+-------------------------------------------+-------------------------------------+-------------------------------------------------------+
| potential temperature                     |  sw.ptmp(SP,t,p,pr)                 | gsw.pt_from_t(SA,t,p,pr)                              |
+-------------------------------------------+-------------------------------------+-------------------------------------------------------+
| :math:`\sigma_0`, using                   |  sw.dens(SP, :math:`\theta_o`, 0)   | gsw.sigma0_CT(SA,CT)                                  |
|  :math:`\theta_o` = sw.ptmp(SP,t,p,0)     |  -1000 kg m :sup:`-3`               |                                                       |
+-------------------------------------------+-------------------------------------+-------------------------------------------------------+
| :math:`\sigma_2`, using                   |  sw.dens(SP,:math:`\theta_2`, 2000) | gsw.sigma2_CT(SA,CT)                                  |
|  :math:`\theta_2` = sw.ptmp(SP,t,p,2000)  |  -1000 kg m :sup:`-3`               |                                                       |
+-------------------------------------------+-------------------------------------+-------------------------------------------------------+
| :math:`\sigma_4`, using                   |  sw.dens(SP,:math:`\theta_4`, 4000) | gsw.sigma2_CT(SA,CT)                                  |
|  :math:`\theta_4` = sw.ptmp(SP,t,p,2000)  |  -1000 kg m :sup:`-3`               |                                                       |
+-------------------------------------------+-------------------------------------+-------------------------------------------------------+
| specific volume anomaly                   |  sw.svan(SP,t,p)                    | gsw.specvol_anom_CT(SA,CT,p)  or                      |
|                                           |                                     | gsw.specvol_anom_CT25(SA,CT,p)                        |
+-------------------------------------------+-------------------------------------+-------------------------------------------------------+
| dynamic height anomaly                    | -sw.gpan(SP,t,p)                    | gsw.geo_strf_dyn_height(SA,CT,p,delta_p,interp_style) |
+-------------------------------------------+-------------------------------------+-------------------------------------------------------+
| geostrophic velocity                      |  sw.gvel(ga,lat,long)               | gsw.geostrophic_velocity(geo_str,long,lat,p)          |
+-------------------------------------------+-------------------------------------+-------------------------------------------------------+
| N :sup:`2`                                |  sw.bfrq(SP,t,p,lat)                | gsw.Nsquared_CT25(SA,CT,p,lat)                        |
+-------------------------------------------+-------------------------------------+-------------------------------------------------------+
| pressure from height                      |  sw.pres(-z,lat)                    | gsw.p_from_z(z,lat)                                   |
| (SW uses depth, not height)               |                                     |                                                       |
+-------------------------------------------+-------------------------------------+-------------------------------------------------------+
| height from pressure                      |  z =  -sw.dpth(p,lat)               | gsw.z_from_p(p,lat)                                   |
| (SW outputs depth, not height)            |                                     |                                                       |
+-------------------------------------------+-------------------------------------+-------------------------------------------------------+
| in situ temperature from pt               |  sw.temp(SP,pt,p,pr)                | gsw.pt_from_t(SA,pt,pr,p)                             |
+-------------------------------------------+-------------------------------------+-------------------------------------------------------+
| sound speed                               |  sw.svel(SP,t,p)                    | gsw.sound_speed(SA,t,p)                               |
+-------------------------------------------+-------------------------------------+-------------------------------------------------------+
| isobaric heat capacity                    |  sw.cp(SP,t,p)                      | gsw.cp(SA,t,p)                                        |
+-------------------------------------------+-------------------------------------+-------------------------------------------------------+
| adiabatic lapse rate*                     |  sw.adtg(SP,t,p)                    | gsw.adiabatic_lapse_rate(SA,t,p)                      |
+-------------------------------------------+-------------------------------------+-------------------------------------------------------+
| SP from cndr,  (PSS 78)                   |  sw.salt(cndr,t,p)                  | gsw.SP_from_cndr(cndr,t,p)                            |
+-------------------------------------------+-------------------------------------+-------------------------------------------------------+
| cndr from SP,  (PSS 78)                   |  sw.cndr(SP,t,p)                    | gsw.cndr_from_SP(SP,t,p)                              |
+-------------------------------------------+-------------------------------------+-------------------------------------------------------+
| distance                                  |  sw.dist(lat,long,units)            | gsw.distance(long,lat,p)                              |
+-------------------------------------------+-------------------------------------+-------------------------------------------------------+
| gravitational acceleration                |  sw.g(lat,z)                        | gsw.grav(lat,p)                                       |
+-------------------------------------------+-------------------------------------+-------------------------------------------------------+
| Coriolis parameter                        |  sw.f(lat)                          | gsw.f(lat)                                            |
+-------------------------------------------+-------------------------------------+-------------------------------------------------------+
| testing of all functions                  |  sw.test()                          | gsw.test()                                            |
+-------------------------------------------+-------------------------------------+-------------------------------------------------------+

\* The SW and GSW functions output the adiabatic lapse rate in different units, being  K (dbar) :sup:`-1`  and  K Pa :sup:`-1`  respectively.


Thanks
======

* Bjørn Ådlandsvik - Testing unit and several bug fixes
* Eric Firing - Support for masked arrays, re-write of _delta_SA
* Trevor J. McDougall (and all of SCOR/IAPSO WG127) for making available the Matlab and Fortran versions of this software

Acknowledgments
---------------

* SCOR/IAPSO WG127. Most of module is derived from the GSW Oceanographic Toolbox of TEOS-10.

The MAJOR.MINOR.MICRO will be used to represent:

MAJOR == The matlab version from the TEOS-10 Group

MINOR == Significant changes made in the python version

MICRO == Bug fixes only
