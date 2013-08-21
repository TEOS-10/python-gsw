python gsw
==========

[![Build](https://badge.fury.io/py/gsw.png)](http://badge.fury.io/py/gsw)
[![Build](https://api.travis-ci.org/ocefpaf/python-gsw.png?branch=master)](https://travis-ci.org/ocefpaf/python-gsw)
[![Downloads](https://pypip.in/d/gsw/badge.png)](https://crate.io/packages/gsw/)


Python implementation of the Thermodynamic Equation Of Seawater - 2010 (TEOS-10)
--------------------------------------------------------------------------------

For more information go to:
    http://www.teos-10.org/


gsw vs. csiro
-------------

This table shows some function names in the gibbs library and the corresponding function names in the csiro library.

| **Variable**                                        | **SeaWater (EOS 80)**                           | **Gibbs SeaWater (GSW TEOS 10)**                      |
|:---------------------------------------------------:|:-----------------------------------------------:|:-----------------------------------------------------:|
| Absolute Salinity                                   |  NA                                             | gsw.SA_from_SP(SP,p,long,lat)                         |
| Conservative Temperature                            |  NA                                             | gsw.CT_from_t(SA,t,p)                                 |
| density (i.e. in situ density)                      |  sw.dens(SP,t,p)                                | gsw.rho_CT(SA,CT,p), or gsw.rho(SA,t,p)               |
| potential density                                   |  sw.pden(SP,t,p,pr)                             | gsw.rho_CT(SA,CT,pr)                                  |
| potential temperature                               |  sw.ptmp(SP,t,p,pr)                             | gsw.pt_from_t(SA,t,p,pr)                              |
| $\sigma_0$, using $\theta_o$ = sw.ptmp(SP,t,p,0)    |  sw.dens(SP, $\theta_o$, 0) -1000 kg m$^{-3}$   | gsw.sigma0_CT(SA,CT)                                  |
| $\sigma_2$, using $\theta_2$ = sw.ptmp(SP,t,p,2000) |  sw.dens(SP,$\theta_2$, 2000) -1000 kg m$^{-3}$ | gsw.sigma2_CT(SA,CT)                                  |
| $\sigma_4$, using $\theta_4$ = sw.ptmp(SP,t,p,2000) |  sw.dens(SP,$\theta_4$, 4000) -1000 kg m$^{-3}$ | gsw.sigma2_CT(SA,CT)                                  |
| specific volume anomaly                             |  sw.svan(SP,t,p)                                | gsw.specvol_anom_CT(SA,CT,p)                          |
| dynamic height anomaly                              | -sw.gpan(SP,t,p)                                | gsw.geo_strf_dyn_height(SA,CT,p,delta_p,interp_style) |
| geostrophic velocity                                |  sw.gvel(ga,lat,long)                           | gsw.geostrophic_velocity(geo_str,long,lat,p)          |
| N$^2$                                               |  sw.bfrq(SP,t,p,lat)                            | gsw.Nsquared(SA,CT,p,lat)                             |
| pressure from height (SW uses depth, not height)    |  sw.pres(-z,lat)                                | gsw.p_from_z(z,lat)                                   |
| height from pressure (SW outputs depth, not height) |  z =  -sw.dpth(p,lat)                           | gsw.z_from_p(p,lat)                                   |
| in situ temperature from pt                         |  sw.temp(SP,pt,p,pr)                            | gsw.pt_from_t(SA,pt,pr,p)                             |
| sound speed                                         |  sw.svel(SP,t,p)                                | gsw.sound_speed(SA,t,p)                               |
| isobaric heat capacity                              |  sw.cp(SP,t,p)                                  | gsw.cp(SA,t,p)                                        |
| adiabatic lapse rate*                               |  sw.adtg(SP,t,p)                                | gsw.adiabatic_lapse_rate(SA,t,p)                      |
| SP from cndr,  (PSS 78)                             |  sw.salt(cndr,t,p)                              | gsw.SP_from_cndr(cndr,t,p)                            |
| cndr from SP,  (PSS 78)                             |  sw.cndr(SP,t,p)                                | gsw.cndr_from_SP(SP,t,p)                              |
| distance                                            |  sw.dist(lat,long,units)                        | gsw.distance(long,lat,p)                              |
| gravitational acceleration                          |  sw.g(lat,z)                                    | gsw.grav(lat,p)                                       |
| Coriolis parameter                                  |  sw.f(lat)                                      | gsw.f(lat)                                            |

Note that the SW and GSW functions output the adiabatic lapse rate in different units, being  K (dbar)$^{-1}$  and  K Pa$^{-1}$
respectively.


Authors
-------
* Bjørn Ådlandsvik
* Eric Firing
* Filipe Fernandes

Thanks
------

* Bjørn Ådlandsvik - Testing unit and several bug fixes.
* Eric Firing - Support for masked arrays, re-write of _delta_SA.
* Trevor J. McDougall (and all of SCOR/IAPSO WG127) for making available the Matlab version of this software.

Acknowledgments
---------------

* SCOR/IAPSO WG127.
