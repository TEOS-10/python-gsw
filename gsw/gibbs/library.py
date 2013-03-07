# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np

from constants import sfac, SSO, db2Pascal
from gsw.utilities import match_args_return, strip_mask, read_data

__all__ = [
           'gibbs',
           #'SAAR',  TODO
           #'Fdelta',  TODO
           #'delta_SA_ref',  TODO: delta_SA ?
           'SA_from_SP_Baltic',
           'SP_from_SA_Baltic',
           'infunnel',
           'entropy_part',
           'entropy_part_zerop',
           'interp_ref_cast',
           'interp_SA_CT',
           'gibbs_pt0_pt0',
           'specvol_SSO_0_p',
           'enthalpy_SSO_0_p',
           'Hill_ratio_at_SP2'
          ]


def gibbs(ns, nt, npr, SA, t, p):
    r"""Calculates specific Gibbs energy and its derivatives up to order 2 for
    seawater.

    The Gibbs function approach allows the calculation of internal energy,
    entropy, enthalpy, potential enthalpy and the chemical potentials of
    seawater as well as the freezing temperature, and the latent heats of
    freezing and of evaporation. These quantities were not available from
    EOS-80 but are essential for the accurate accounting of heat in the ocean
    and for the consistent and accurate treatment of air-sea and ice-sea heat
    fluxes.

    Parameters
    ----------
    ns : int
         order of SA derivative [0, 1 or 2 ]
    nt : int
         order of t derivative [0, 1 or 2 ]
    npr : int
          order of p derivative [0, 1 or 2 ]
    SA : array_like
         Absolute salinity [g kg :sup:`-1`]
    t : array_like
        in situ temperature [:math:`^\circ` C (ITS-90)]
    p : array_like
        pressure [dbar]

    Returns
    -------
    gibbs : array_like
            Specific Gibbs energy or its derivatives.

            Gibbs energy (ns=nt=npr=0) has units of:
            [J kg :sup:`-1`]

            Absolute Salinity derivatives are output in units of:
            [(J kg :sup:`-1`) (g kg :sup:`-1`) :sup:`-ns`]

            Temperature derivatives are output in units of:
            [(J kg :sup:`-1`) K :sup:`-nt`]

            Pressure derivatives are output in units of:
            [(J kg :sup:`-1`) Pa :sup:`-npr`]

            The mixed derivatives are output in units of:
            [(J kg :sup:`-1`) (g kg :sup:`-1`) :sup:`-ns` K :sup:`-nt`
            Pa :sup:`-npr`]

    Notes
    -----
    The Gibbs function for seawater is that of TEOS-10 (IOC et al., 2010),
    being the sum of IAPWS-08 for the saline part and IAPWS-09 for the pure
    water part. These IAPWS releases are the officially blessed IAPWS
    descriptions of Feistel (2008) and the pure water part of Feistel (2003).
    Absolute Salinity, SA, in all of the GSW routines is expressed on the
    Reference-Composition Salinity Scale of 2008 (RCSS-08) of Millero et al.
    (2008).

    The derivatives are taken with respect to pressure in Pa, not withstanding
    that the pressure input into this routine is in dbar.

    References
    ----------
    .. [1] Feistel, R., 2003: A new extended Gibbs thermodynamic potential of
    seawater Progr. Oceanogr., 58, 43-114.

    .. [2] Feistel, R., 2008: A Gibbs function for seawater thermodynamics
    for -6 to 80 :math:`^\circ` C and salinity up to 120 g kg :sup:`-1`,
    Deep-Sea Res. I, 55, 1639-1671.

    .. [3] IAPWS, 2008: Release on the IAPWS Formulation 2008 for the
    Thermodynamic Properties of Seawater. The International Association for the
    Properties of Water and Steam. Berlin, Germany, September 2008, available
    from http://www.iapws.org.  This Release is referred to as IAPWS-08.

    .. [4] IAPWS, 2009: Supplementary Release on a Computationally Efficient
    Thermodynamic Formulation for Liquid Water for Oceanographic Use. The
    International Association for the Properties of Water and Steam. Doorwerth,
    The Netherlands, September 2009, available from http://www.iapws.org.
    This Release is referred to as IAPWS-09.

    .. [5] IOC, SCOR and IAPSO, 2010: The international thermodynamic equation
    of seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp. See section 2.6 and appendices A.6,  G and H.

    .. [6] Millero, F. J., R. Feistel, D. G. Wright, and T. J. McDougall, 2008:
    The composition of Standard Seawater and the definition of the
    Reference-Composition Salinity Scale, Deep-Sea Res. I, 55, 50-72.

    Modifications:
    2010-09-24. David Jackett, Paul Barker and Trevor McDougall
    """

    SA, t, p = np.asanyarray(SA), np.asanyarray(t), np.asanyarray(p)

    SA = np.atleast_1d(SA)
    nonzero_SA = np.any(SA > 0)

    _SA = SA
    _t = t
    _p = p

    SA = np.ma.filled(SA, 0)
    t = np.ma.filled(t, 20)
    p = np.ma.filled(p, 10)

    SA, t, p = np.broadcast_arrays(SA, t, p)

    gibbs = np.zeros(SA.shape, dtype=np.float)  # Use if all_masked is True
    all_masked = False

    # Ensure a full mask, so we can set elements if necessary.
    mask = np.ma.mask_or(np.ma.getmaskarray(_SA), np.ma.getmask(_t))
    mask = np.ma.mask_or(mask, np.ma.getmask(_p))
    mask = np.ma.mask_or(mask, SA < 0)

    ipos = (SA > 0)
    # inpos = ~ipos  # FIXME: Assigned but never used.
    if np.all(ipos):
        ipos = slice(None)  # More efficient for usual case.

    x2 = sfac * SA
    x = np.sqrt(x2)

    y = t * 0.025
    z = p * 1e-4  # The input pressure (p) is sea pressure in units of dbar.

    if (ns == 0) & (nt == 0) & (npr == 0):
        g03 = (101.342743139674 + z * (100015.695367145 +
        z * (-2544.5765420363 + z * (284.517778446287 +
        z * (-33.3146754253611 + (4.20263108803084 - 0.546428511471039 * z)
        * z)))) +
        y * (5.90578347909402 + z * (-270.983805184062 +
        z * (776.153611613101 + z * (-196.51255088122 + (28.9796526294175 -
        2.13290083518327 * z) * z))) +
        y * (-12357.785933039 + z * (1455.0364540468 +
        z * (-756.558385769359 + z * (273.479662323528 + z *
        (-55.5604063817218 + 4.34420671917197 * z)))) +
        y * (736.741204151612 + z * (-672.50778314507 +
        z * (499.360390819152 + z * (-239.545330654412 + (48.8012518593872 -
        1.66307106208905 * z) * z))) +
        y * (-148.185936433658 + z * (397.968445406972 +
        z * (-301.815380621876 + (152.196371733841 - 26.3748377232802 * z) *
        z)) +
        y * (58.0259125842571 + z * (-194.618310617595 +
        z * (120.520654902025 + z * (-55.2723052340152 +
        6.48190668077221 * z))) +
        y * (-18.9843846514172 + y * (3.05081646487967 -
        9.63108119393062 * z) +
        z * (63.5113936641785 + z * (-22.2897317140459 +
        8.17060541818112 * z)))))))))

        if nonzero_SA:
            g08 = x2 * (1416.27648484197 + z * (-3310.49154044839 +
            z * (384.794152978599 + z * (-96.5324320107458 +
            (15.8408172766824 - 2.62480156590992 * z) * z))) +
            x * (-2432.14662381794 + x * (2025.80115603697 +
            y * (543.835333000098 + y * (-68.5572509204491 +
            y * (49.3667694856254 + y * (-17.1397577419788 +
            2.49697009569508 * y))) - 22.6683558512829 * z) +
            x * (-1091.66841042967 - 196.028306689776 * y +
            x * (374.60123787784 - 48.5891069025409 * x +
            36.7571622995805 * y) + 36.0284195611086 * z) +
            z * (-54.7919133532887 + (-4.08193978912261 -
            30.1755111971161 * z) * z)) +
            z * (199.459603073901 + z * (-52.2940909281335 +
            (68.0444942726459 - 3.41251932441282 * z) * z)) +
            y * (-493.407510141682 + z * (-175.292041186547 +
            (83.1923927801819 - 29.483064349429 * z) * z) +
            y * (-43.0664675978042 + z * (383.058066002476 + z *
            (-54.1917262517112 + 25.6398487389914 * z)) +
            y * (-10.0227370861875 - 460.319931801257 * z + y *
            (0.875600661808945 + 234.565187611355 * z))))) +
            y * (168.072408311545 + z * (729.116529735046 +
            z * (-343.956902961561 + z * (124.687671116248 + z *
            (-31.656964386073 + 7.04658803315449 * z)))) +
            y * (880.031352997204 + y * (-225.267649263401 +
            y * (91.4260447751259 + y * (-21.6603240875311 +
            2.13016970847183 * y) +
            z * (-297.728741987187 + (74.726141138756 -
            36.4872919001588 * z) * z)) +
            z * (694.244814133268 + z * (-204.889641964903 +
            (113.561697840594 - 11.1282734326413 * z) * z))) +
            z * (-860.764303783977 + z * (337.409530269367 +
            z * (-178.314556207638 + (44.2040358308 -
            7.92001547211682 * z) * z))))))

            g08[ipos] += x2[ipos] * (5812.81456626732 + 851.226734946706 *
            y[ipos]) * np.log(x[ipos])
        else:
            g08 = 0
        gibbs = g03 + g08

    elif (ns == 1) & (nt == 0) & (npr == 0):
        if nonzero_SA:
            g08 = (8645.36753595126 + z * (-6620.98308089678 +
            z * (769.588305957198 + z * (-193.0648640214916 +
            (31.6816345533648 - 5.24960313181984 * z) * z))) +
            x * (-7296.43987145382 + x * (8103.20462414788 +
            y * (2175.341332000392 + y * (-274.2290036817964 +
            y * (197.4670779425016 + y * (-68.5590309679152 +
            9.98788038278032 * y))) - 90.6734234051316 * z) +
            x * (-5458.34205214835 - 980.14153344888 * y +
            x * (2247.60742726704 - 340.1237483177863 * x +
            220.542973797483 * y) + 180.142097805543 * z) +
            z * (-219.1676534131548 + (-16.32775915649044 -
            120.7020447884644 * z) * z)) +
            z * (598.378809221703 + z * (-156.8822727844005 +
            (204.1334828179377 - 10.23755797323846 * z) * z)) +
            y * (-1480.222530425046 + z * (-525.876123559641 +
            (249.57717834054571 - 88.449193048287 * z) * z) +
            y * (-129.1994027934126 + z * (1149.174198007428 +
            z * (-162.5751787551336 + 76.9195462169742 * z)) +
            y * (-30.0682112585625 - 1380.9597954037708 * z + y *
            (2.626801985426835 + 703.695562834065 * z))))) +
            y * (1187.3715515697959 + z * (1458.233059470092 +
            z * (-687.913805923122 + z * (249.375342232496 + z *
            (-63.313928772146 + 14.09317606630898 * z)))) +
            y * (1760.062705994408 + y * (-450.535298526802 +
            y * (182.8520895502518 + y * (-43.3206481750622 +
            4.26033941694366 * y) +
            z * (-595.457483974374 + (149.452282277512 -
            72.9745838003176 * z) * z)) +
            z * (1388.489628266536 + z * (-409.779283929806 +
            (227.123395681188 - 22.2565468652826 * z) * z))) +
            z * (-1721.528607567954 + z * (674.819060538734 +
            z * (-356.629112415276 + (88.4080716616 -
            15.84003094423364 * z) * z))))))

            g08[ipos] = g08[ipos] + (11625.62913253464 + 1702.453469893412 *
            y[ipos]) * np.log(x[ipos])

            gibbs = 0.5 * sfac * g08
        else:
            all_masked = True

    elif (ns == 0) & (nt == 1) & (npr == 0):
        g03 = (5.90578347909402 + z * (-270.983805184062 +
        z * (776.153611613101 + z * (-196.51255088122 +
        (28.9796526294175 - 2.13290083518327 * z) * z))) +
        y * (-24715.571866078 + z * (2910.0729080936 +
        z * (-1513.116771538718 + z * (546.959324647056 + z *
        (-111.1208127634436 + 8.68841343834394 * z)))) +
        y * (2210.2236124548363 + z * (-2017.52334943521 +
        z * (1498.081172457456 + z * (-718.6359919632359 +
        (146.4037555781616 - 4.9892131862671505 * z) * z))) +
        y * (-592.743745734632 + z * (1591.873781627888 +
        z * (-1207.261522487504 + (608.785486935364 -
        105.4993508931208 * z) * z)) +
        y * (290.12956292128547 + z * (-973.091553087975 +
        z * (602.603274510125 + z * (-276.361526170076 +
        32.40953340386105 * z))) +
        y * (-113.90630790850321 + y * (21.35571525415769 -
        67.41756835751434 * z) +
        z * (381.06836198507096 + z * (-133.7383902842754 +
        49.023632509086724 * z))))))))

        if nonzero_SA:
            g08 = x2 * (168.072408311545 + z * (729.116529735046 +
            z * (-343.956902961561 + z * (124.687671116248 + z *
            (-31.656964386073 + 7.04658803315449 * z)))) +
            x * (-493.407510141682 + x * (543.835333000098 + x *
            (-196.028306689776 + 36.7571622995805 * x) +
            y * (-137.1145018408982 + y * (148.10030845687618 + y *
            (-68.5590309679152 + 12.4848504784754 * y))) -
            22.6683558512829 * z) + z * (-175.292041186547 +
            (83.1923927801819 - 29.483064349429 * z) * z) +
            y * (-86.1329351956084 + z * (766.116132004952 + z *
            (-108.3834525034224 + 51.2796974779828 * z)) +
            y * (-30.0682112585625 - 1380.9597954037708 * z + y *
            (3.50240264723578 + 938.26075044542 * z)))) +
            y * (1760.062705994408 + y * (-675.802947790203 +
            y * (365.7041791005036 + y * (-108.30162043765552 +
            12.78101825083098 * y) +
            z * (-1190.914967948748 + (298.904564555024 -
            145.9491676006352 * z) * z)) +
            z * (2082.7344423998043 + z * (-614.668925894709 +
            (340.685093521782 - 33.3848202979239 * z) * z))) +
            z * (-1721.528607567954 + z * (674.819060538734 +
            z * (-356.629112415276 + (88.4080716616 -
            15.84003094423364 * z) * z)))))

            g08[ipos] += 851.226734946706 * x2[ipos] * np.log(x[ipos])

            gibbs = (g03 + g08) * 0.025
        else:
            gibbs = g03

    elif (ns == 0) & (nt == 0) & (npr == 1):
        g03 = (100015.695367145 + z * (-5089.1530840726 +
        z * (853.5533353388611 + z * (-133.2587017014444 +
        (21.0131554401542 - 3.278571068826234 * z) * z))) +
        y * (-270.983805184062 + z * (1552.307223226202 +
        z * (-589.53765264366 + (115.91861051767 -
        10.664504175916349 * z) * z)) +
        y * (1455.0364540468 + z * (-1513.116771538718 +
        z * (820.438986970584 + z * (-222.2416255268872 +
        21.72103359585985 * z))) +
        y * (-672.50778314507 + z * (998.720781638304 +
        z * (-718.6359919632359 + (195.2050074375488 -
        8.31535531044525 * z) * z)) +
        y * (397.968445406972 + z * (-603.630761243752 +
        (456.589115201523 - 105.4993508931208 * z) * z) +
        y * (-194.618310617595 + y * (63.5113936641785 -
        9.63108119393062 * y +
        z * (-44.5794634280918 + 24.511816254543362 * z)) +
        z * (241.04130980405 + z * (-165.8169157020456 +
        25.92762672308884 * z))))))))

        if nonzero_SA:
            g08 = x2 * (-3310.49154044839 + z * (769.588305957198 +
            z * (-289.5972960322374 + (63.3632691067296 -
            13.1240078295496 * z) * z)) +
            x * (199.459603073901 + x * (-54.7919133532887 +
            36.0284195611086 * x - 22.6683558512829 * y +
            (-8.16387957824522 - 90.52653359134831 * z) * z) +
            z * (-104.588181856267 + (204.1334828179377 -
            13.65007729765128 * z) * z) +
            y * (-175.292041186547 + (166.3847855603638 -
            88.449193048287 * z) * z +
            y * (383.058066002476 + y * (-460.319931801257 +
            234.565187611355 * y) +
            z * (-108.3834525034224 + 76.9195462169742 * z)))) +
            y * (729.116529735046 + z * (-687.913805923122 +
            z * (374.063013348744 + z * (-126.627857544292 +
            35.23294016577245 * z))) +
            y * (-860.764303783977 + y * (694.244814133268 +
            y * (-297.728741987187 + (149.452282277512 -
            109.46187570047641 * z) * z) +
            z * (-409.779283929806 + (340.685093521782 -
            44.5130937305652 * z) * z)) +
            z * (674.819060538734 + z * (-534.943668622914 +
            (176.8161433232 - 39.600077360584095 * z) * z)))))
        else:
            g08 = 0
        # Pressure derivative of the Gibbs function
        # in units of (J kg :sup:`-1`) (Pa :sup:`-1`) = m :sup:`3` kg :sup:`-1`
        gibbs = (g03 + g08) * 1e-8

    elif (ns == 1) & (nt == 1) & (npr == 0):
        if nonzero_SA:
            g08 = (1187.3715515697959 + z * (1458.233059470092 +
            z * (-687.913805923122 + z * (249.375342232496 + z *
            (-63.313928772146 + 14.09317606630898 * z)))) +
            x * (-1480.222530425046 + x * (2175.341332000392 + x *
            (-980.14153344888 + 220.542973797483 * x) +
            y * (-548.4580073635929 + y * (592.4012338275047 + y *
            (-274.2361238716608 + 49.9394019139016 * y))) -
            90.6734234051316 * z) + z * (-525.876123559641 +
            (249.57717834054571 - 88.449193048287 * z) * z) +
            y * (-258.3988055868252 + z * (2298.348396014856 +
            z * (-325.1503575102672 + 153.8390924339484 * z)) +
            y * (-90.2046337756875 - 4142.8793862113125 * z + y *
            (10.50720794170734 + 2814.78225133626 * z)))) +
            y * (3520.125411988816 + y * (-1351.605895580406 +
            y * (731.4083582010072 + y * (-216.60324087531103 +
            25.56203650166196 * y) +
            z * (-2381.829935897496 + (597.809129110048 -
            291.8983352012704 * z) * z)) +
            z * (4165.4688847996085 + z * (-1229.337851789418 +
            (681.370187043564 - 66.7696405958478 * z) * z))) +
            z * (-3443.057215135908 + z * (1349.638121077468 +
            z * (-713.258224830552 + (176.8161433232 -
            31.68006188846728 * z) * z)))))

            g08[ipos] = g08[ipos] + 1702.453469893412 * np.log(x[ipos])
            gibbs = 0.5 * sfac * 0.025 * g08
            # FIXME: commented by FF, g110 without nan did not pass
            #mask[inpos] = True
        else:
            all_masked = True

    elif (ns == 1) & (nt == 0) & (npr == 1):
        g08 = (-6620.98308089678 + z * (1539.176611914396 +
        z * (-579.1945920644748 + (126.7265382134592 -
        26.2480156590992 * z) * z)) +
        x * (598.378809221703 + x * (-219.1676534131548 +
        180.142097805543 * x - 90.6734234051316 * y +
        (-32.65551831298088 - 362.10613436539325 * z) * z) +
        z * (-313.764545568801 + (612.4004484538132 -
        40.95023189295384 * z) * z) +
        y * (-525.876123559641 + (499.15435668109143 -
        265.347579144861 * z) * z +
        y * (1149.174198007428 + y * (-1380.9597954037708 +
        703.695562834065 * y) +
        z * (-325.1503575102672 + 230.7586386509226 * z)))) +
        y * (1458.233059470092 + z * (-1375.827611846244 +
        z * (748.126026697488 + z * (-253.255715088584 +
        70.4658803315449 * z))) +
        y * (-1721.528607567954 + y * (1388.489628266536 +
        y * (-595.457483974374 + (298.904564555024 -
        218.92375140095282 * z) * z) +
        z * (-819.558567859612 + (681.370187043564 -
        89.0261874611304 * z) * z)) +
        z * (1349.638121077468 + z * (-1069.887337245828 +
        (353.6322866464 - 79.20015472116819 * z) * z)))))

        # Derivative of the Gibbs function is in units of
        # (m :sup:`3` kg :sup:`-1`) / (g kg :sup:`-1`) = m :sup:`3` g :sup:`-1`
        # that is, it is the derivative of specific volume with respect to
        # Absolute Salinity measured in g kg :sup:`-1`

        gibbs = g08 * sfac * 0.5e-8

    elif (ns == 0) & (nt == 1) & (npr == 1):
        g03 = (-270.983805184062 + z * (1552.307223226202 + z *
        (-589.53765264366 +
        (115.91861051767 - 10.664504175916349 * z) * z)) +
        y * (2910.0729080936 + z * (-3026.233543077436 +
        z * (1640.877973941168 + z * (-444.4832510537744 +
        43.4420671917197 * z))) +
        y * (-2017.52334943521 + z * (2996.162344914912 +
        z * (-2155.907975889708 + (585.6150223126464 -
        24.946065931335752 * z) * z)) +
        y * (1591.873781627888 + z * (-2414.523044975008 +
        (1826.356460806092 - 421.9974035724832 * z) * z) +
        y * (-973.091553087975 + z * (1205.20654902025 + z *
        (-829.084578510228 + 129.6381336154442 * z)) +
        y * (381.06836198507096 - 67.41756835751434 * y + z *
        (-267.4767805685508 + 147.07089752726017 * z)))))))

        if nonzero_SA:
            g08 = x2 * (729.116529735046 + z * (-687.913805923122 +
            z * (374.063013348744 + z * (-126.627857544292 +
            35.23294016577245 * z))) +
            x * (-175.292041186547 - 22.6683558512829 * x +
            (166.3847855603638 - 88.449193048287 * z) * z +
            y * (766.116132004952 + y * (-1380.9597954037708 +
            938.26075044542 * y) +
            z * (-216.7669050068448 + 153.8390924339484 * z))) +
            y * (-1721.528607567954 + y * (2082.7344423998043 +
            y * (-1190.914967948748 + (597.809129110048 -
            437.84750280190565 * z) * z) +
            z * (-1229.337851789418 + (1022.055280565346 -
            133.5392811916956 * z) * z)) +
            z * (1349.638121077468 + z * (-1069.887337245828 +
            (353.6322866464 - 79.20015472116819 * z) * z))))
        else:
            g08 = 0
        # Derivative of the Gibbs function is in units of (m :sup:`3` (K kg))
        # that is, the pressure of the derivative in Pa.
        gibbs = (g03 + g08) * 2.5e-10

    elif (ns == 2) & (nt == 0) & (npr == 0):
        g08 = 2.0 * (8103.20462414788 +
        y * (2175.341332000392 + y * (-274.2290036817964 +
        y * (197.4670779425016 + y * (-68.5590309679152 +
        9.98788038278032 * y))) - 90.6734234051316 * z) +
        1.5 * x * (-5458.34205214835 - 980.14153344888 * y +
        (4.0 / 3.0) * x * (2247.60742726704 - 340.1237483177863 * 1.25 *
        x + 220.542973797483 * y) +
        180.142097805543 * z) +
        z * (-219.1676534131548 + (-16.32775915649044 -
        120.7020447884644 * z) * z))

        if nonzero_SA:
            tmp = ((-7296.43987145382 + z * (598.378809221703 +
            z * (-156.8822727844005 + (204.1334828179377 -
            10.23755797323846 * z) * z)) +
            y * (-1480.222530425046 + z * (-525.876123559641 +
            (249.57717834054571 - 88.449193048287 * z) * z) +
            y * (-129.1994027934126 + z * (1149.174198007428 +
            z * (-162.5751787551336 + 76.9195462169742 * z)) +
            y * (-30.0682112585625 - 1380.9597954037708 * z +
            y * (2.626801985426835 + 703.695562834065 * z))))) / x +
            (11625.62913253464 + 1702.453469893412 * y) / x2)
            g08[ipos] += tmp[ipos]

        gibbs = 0.25 * sfac ** 2 * g08

    elif (ns == 0) & (nt == 2) & (npr == 0):
        g03 = (-24715.571866078 + z * (2910.0729080936 + z *
        (-1513.116771538718 + z * (546.959324647056 + z *
        (-111.1208127634436 + 8.68841343834394 * z)))) +
        y * (4420.4472249096725 + z * (-4035.04669887042 +
        z * (2996.162344914912 + z * (-1437.2719839264719 +
        (292.8075111563232 - 9.978426372534301 * z) * z))) +
        y * (-1778.231237203896 + z * (4775.621344883664 +
        z * (-3621.784567462512 + (1826.356460806092 -
        316.49805267936244 * z) * z)) +
        y * (1160.5182516851419 + z * (-3892.3662123519 +
        z * (2410.4130980405 + z * (-1105.446104680304 +
        129.6381336154442 * z))) +
        y * (-569.531539542516 + y * (128.13429152494615 -
        404.50541014508605 * z) +
        z * (1905.341809925355 + z * (-668.691951421377 +
        245.11816254543362 * z)))))))

        if nonzero_SA:
            g08 = x2 * (1760.062705994408 + x * (-86.1329351956084 +
            x * (-137.1145018408982 + y * (296.20061691375236 +
            y * (-205.67709290374563 + 49.9394019139016 * y))) +
            z * (766.116132004952 + z * (-108.3834525034224 +
            51.2796974779828 * z)) +
            y * (-60.136422517125 - 2761.9195908075417 * z +
            y * (10.50720794170734 + 2814.78225133626 * z))) +
            y * (-1351.605895580406 + y * (1097.1125373015109 +
            y * (-433.20648175062206 + 63.905091254154904 * y) +
            z * (-3572.7449038462437 + (896.713693665072 -
            437.84750280190565 * z) * z)) +
            z * (4165.4688847996085 + z * (-1229.337851789418 +
            (681.370187043564 - 66.7696405958478 * z) * z))) +
            z * (-1721.528607567954 + z * (674.819060538734 +
            z * (-356.629112415276 + (88.4080716616 -
            15.84003094423364 * z) * z))))
        else:
            g08 = 0
        gibbs = (g03 + g08) * 0.000625

    elif (ns == 0) & (nt == 0) & (npr == 2):
        g03 = (-5089.1530840726 + z * (1707.1066706777221 +
        z * (-399.7761051043332 + (84.0526217606168 -
        16.39285534413117 * z) * z)) +
        y * (1552.307223226202 + z * (-1179.07530528732 +
        (347.75583155301 - 42.658016703665396 * z) * z) +
        y * (-1513.116771538718 + z * (1640.877973941168 +
        z * (-666.7248765806615 + 86.8841343834394 * z)) +
        y * (998.720781638304 + z * (-1437.2719839264719 +
        (585.6150223126464 - 33.261421241781 * z) * z) +
        y * (-603.630761243752 + (913.178230403046 -
        316.49805267936244 * z) * z +
        y * (241.04130980405 + y * (-44.5794634280918 +
        49.023632509086724 * z) +
        z * (-331.6338314040912 + 77.78288016926652 * z)))))))

        if nonzero_SA:
            g08 = x2 * (769.588305957198 + z * (-579.1945920644748 +
            (190.08980732018878 - 52.4960313181984 * z) * z) +
            x * (-104.588181856267 + x * (-8.16387957824522 -
            181.05306718269662 * z) +
            (408.2669656358754 - 40.95023189295384 * z) * z +
            y * (166.3847855603638 - 176.898386096574 * z + y *
            (-108.3834525034224 + 153.8390924339484 * z))) +
            y * (-687.913805923122 + z * (748.126026697488 +
            z * (-379.883572632876 + 140.9317606630898 * z)) +
            y * (674.819060538734 + z * (-1069.887337245828 +
            (530.4484299696 - 158.40030944233638 * z) * z) +
            y * (-409.779283929806 + y * (149.452282277512 -
            218.92375140095282 * z) +
            (681.370187043564 - 133.5392811916956 * z) * z))))
        else:
            g08 = 0
        # Second derivative of the Gibbs function with respect to pressure,
        # measured in Pa; units of (J kg :sup:`-1`) (Pa :sup:`-2`).
        gibbs = (g03 + g08) * 1e-16
    else:
        raise ValueError('Illegal derivative of the Gibbs function')

    gibbs = np.ma.array(gibbs, mask=mask, copy=False)

    # BÅ: Code below is not needed?
    #if all_masked:
    #    gibbs[:] = np.ma.masked

    # Do not allow zero salinity with salinity derivatives
    if ns > 0:
        gibbs = np.ma.masked_where(SA == 0, gibbs)

    return gibbs


def entropy_part(SA, t, p):
    r"""Calculates entropy, except that it does not evaluate any terms that are
    functions of Absolute Salinity alone.

    Parameters
    ----------
    SA : array_like
         Absolute salinity [g kg :sup:`-1`]
    t : array_like
        in situ temperature [:math:`^\circ` C (ITS-90)]
    p : array_like
        pressure [dbar]

    Returns
    -------
    entropy_part : array_like
                   entropy minus the terms that due to SA alone
                   [J kg :sup:`-1` K :sup:`-1`]

    Notes
    -----
    By not calculating these terms, which are a function only of Absolute
    Salinity, several unnecessary computations are avoided (including saving
    the computation of a natural logarithm). These terms are a necessary part
    of entropy, but are not needed when calculating potential temperature from
    in situ temperature.

    Modifications:
    """

    SA, t, p, mask = strip_mask(SA, t, p)

    x2 = sfac * SA
    x = np.sqrt(x2)
    y = t * 0.025
    z = p * 1e-4

    g03 = (z * (-270.983805184062 +
    z * (776.153611613101 + z * (-196.51255088122 + (28.9796526294175 -
    2.13290083518327 * z) * z))) +
    y * (-24715.571866078 + z * (2910.0729080936 +
    z * (-1513.116771538718 + z * (546.959324647056 + z *
    (-111.1208127634436 + 8.68841343834394 * z)))) +
    y * (2210.2236124548363 + z * (-2017.52334943521 +
    z * (1498.081172457456 + z * (-718.6359919632359 +
    (146.4037555781616 - 4.9892131862671505 * z) * z))) +
    y * (-592.743745734632 + z * (1591.873781627888 +
    z * (-1207.261522487504 + (608.785486935364 -
    105.4993508931208 * z) * z)) +
    y * (290.12956292128547 + z * (-973.091553087975 +
    z * (602.603274510125 + z * (-276.361526170076 +
    32.40953340386105 * z))) +
    y * (-113.90630790850321 + y *
    (21.35571525415769 - 67.41756835751434 * z) +
    z * (381.06836198507096 + z * (-133.7383902842754 +
    49.023632509086724 * z))))))))

    # TODO? short-circuit this if SA is zero
    g08 = x2 * (z * (729.116529735046 +
    z * (-343.956902961561 + z * (124.687671116248 + z * (-31.656964386073 +
    7.04658803315449 * z)))) +
    x * (x * (y * (-137.1145018408982 + y * (148.10030845687618 +
    y * (-68.5590309679152 + 12.4848504784754 * y))) -
    22.6683558512829 * z) + z * (-175.292041186547 +
    (83.1923927801819 - 29.483064349429 * z) * z) +
    y * (-86.1329351956084 + z * (766.116132004952 +
    z * (-108.3834525034224 + 51.2796974779828 * z)) +
    y * (-30.0682112585625 - 1380.9597954037708 * z +
    y * (3.50240264723578 + 938.26075044542 * z)))) +
    y * (1760.062705994408 + y * (-675.802947790203 +
    y * (365.7041791005036 + y * (-108.30162043765552 +
    12.78101825083098 * y) +
    z * (-1190.914967948748 + (298.904564555024 -
    145.9491676006352 * z) * z)) +
    z * (2082.7344423998043 + z * (-614.668925894709 +
    (340.685093521782 - 33.3848202979239 * z) * z))) +
    z * (-1721.528607567954 + z * (674.819060538734 +
    z * (-356.629112415276 + (88.4080716616 -
    15.84003094423364 * z) * z)))))

    entropy_part = -(g03 + g08) * 0.025

    return np.ma.array(entropy_part, mask=mask, copy=False)


def gibbs_pt0_pt0(SA, pt0):
    r"""Calculates the second derivative of the specific Gibbs function with
    respect to temperature at zero sea pressure or _gibbs(0,2,0,SA,t,0).

    Parameters
    ----------
    SA : array_like
         Absolute salinity [g kg :sup:`-1`]
    pt0 : array_like
          potential temperature relative to 0 dbar [:math:`^\circ` C (ITS-90)]

    Returns
    -------
    gibbs_pt0_pt0 : array_like
                    TODO: write the eq. for the second derivative of the
                    specific Gibbs function. FIXME: [units]

    Notes
    -----
    This library function is called by both "pt_from_CT(SA,CT)"
    and "pt0_from_t(SA,t,p)".

    Modifications:
    """

    SA, pt0, mask = strip_mask(SA, pt0)

    x2 = sfac * SA
    x = np.sqrt(x2)
    y = pt0 * 0.025

    g03 = (-24715.571866078 +
    y * (4420.4472249096725 +
    y * (-1778.231237203896 +
    y * (1160.5182516851419 +
    y * (-569.531539542516 + y * 128.13429152494615)))))

    g08 = x2 * (1760.062705994408 + x * (-86.1329351956084 +
    x * (-137.1145018408982 + y * (296.20061691375236 +
    y * (-205.67709290374563 + 49.9394019139016 * y))) +
    y * (-60.136422517125 + y * 10.50720794170734)) +
    y * (-1351.605895580406 + y * (1097.1125373015109 +
    y * (-433.20648175062206 + 63.905091254154904 * y))))

    gibbs_pt0_pt0 = (g03 + g08) * 0.000625

    return np.ma.array(gibbs_pt0_pt0, mask=mask, copy=False)


def entropy_part_zerop(SA, pt0):
    r"""Calculates entropy at a sea surface (p = 0 dbar), except that it does
    not evaluate any terms that are functions of Absolute Salinity alone.

    Parameters
    ----------
    SA : array_like
         Absolute salinity [g kg :sup:`-1`]
    pt0 : array_like
          potential temperature relative to 0 dbar [:math:`^\circ` C (ITS-90)]

    Returns
    -------
    entropy_part_zerop : array_like
                         [J kg :sup:`-1` K :sup:`-1`]

    Notes
    -----
    By not calculating these terms, which are a function only of Absolute
    Salinity, several unnecessary computations are avoided (including saving
    the computation of a natural logarithm). These terms are a necessary part
    of entropy, but are not needed when calculating potential temperature from
    in situ temperature.

    Modifications:
    """

    SA, pt0, mask = strip_mask(SA, pt0)

    x2 = sfac * SA
    x = np.sqrt(x2)
    y = pt0 * 0.025

    g03 = y * (-24715.571866078 + y * (2210.2236124548363 +
    y * (-592.743745734632 + y * (290.12956292128547 +
    y * (-113.90630790850321 + y * 21.35571525415769)))))

    g08 = x2 * (x * (x * (y * (-137.1145018408982 + y *
    (148.10030845687618 +
    y * (-68.5590309679152 + 12.4848504784754 * y)))) +
    y * (-86.1329351956084 + y * (-30.0682112585625 + y *
    3.50240264723578))) +
    y * (1760.062705994408 + y * (-675.802947790203 +
    y * (365.7041791005036 + y * (-108.30162043765552 +
    12.78101825083098 * y)))))

    entropy_part_zerop = -(g03 + g08) * 0.025

    return np.ma.array(entropy_part_zerop, mask=mask, copy=False)


# FIXME: Check if this is still used and remove it.
def enthalpy_SSO_0_CT25(p):
    r"""Calculates enthalpy at the Standard Ocean Salinity (SSO) and at a
    Conservative Temperature of zero degrees C (CT=0), as a function of
    pressure (p [dbar]) or enthalpy_CT25(35.16504,0,p).

    Parameters
    ----------
    p : array_like
        pressure [dbar]

    Returns
    -------
    enthalpy_CT25 : array_like
                    enthalpy_CT25 at (SSO, CT = 0, p), 25-term equation.
                    [J kg :sup:`-1`]

    Notes
    -----
    Uses a streamlined version of the 25-term CT version of the Gibbs function,
    that is, a streamlined version of the code "enthalpy_CT25(SA,CT,p)"

    Modifications:
    """

    p = np.asanyarray(p)
    mask = np.ma.getmask(p)
    p = np.ma.filled(p, 0)

    a0 = 1 + SSO * (2.0777716085618458e-3 + np.sqrt(SSO) *
    3.4688210757917340e-6)
    a1 = 6.8314629554123324e-6
    b0 = 9.9984380290708214e2 + SSO * (2.8925731541277653e0 + SSO *
    1.9457531751183059e-3)
    b1 = 0.5 * (1.1930681818531748e-2 + SSO * 5.9355685925035653e-6)
    b2 = -2.5943389807429039e-8
    A = b1 - np.sqrt(b1 ** 2 - b0 * b2)
    B = b1 + np.sqrt(b1 ** 2 - b0 * b2)

    part = (a0 * b2 - a1 * b1) / (b2 * (B - A))

    enthalpy_SSO_0_CT25 = db2Pascal * ((a1 / (2 * b2)) *
    np.log(1 + p * (2 * b1 + b2 * p) / b0) + part *
    np.log(1 + (b2 * p * (B - A)) / (A * (B + b2 * p))))

    return np.ma.array(enthalpy_SSO_0_CT25, mask=mask, copy=False)


# FIXME: Check if this is still used and remove it.
def specvol_SSO_0_CT25(p):
    r"""Calculates specific volume at the Standard Ocean Salinity (SSO) and
    Conservative Temperature of zero degrees C (CT=0), as a function of
    pressure (p [dbar]) or spec_vol_CT25(35.16504,0,p).

    Parameters
    ----------
    p : array_like
        pressure [dbar]

    Returns
    -------
    specvol_SSO_0_CT25 : array_like
                         Specific volume at (SSO, CT=0, p), 25-term equation.
                         [m :sup:`3` kg :sup:`-1`]

    Notes
    -----
    It uses a streamlined version of the 25-term CT version of specific volume
    that is, a streamlined version of the code "rho_alpha_beta_CT25(SA,CT,p)"

    Modifications
    """

    p = np.asanyarray(p)
    # No need to strip mask and replace it here; the calculation is simple.

    specvol_SSO_0_CT25 = ((1.00000000e+00 + SSO * (2.0777716085618458e-003 +
    np.sqrt(SSO) * 3.4688210757917340e-006) + p * 6.8314629554123324e-006) /
    (9.9984380290708214e+002 + SSO * (2.8925731541277653e+000 + SSO *
    1.9457531751183059e-003) + p * (1.1930681818531748e-002 + SSO *
    5.9355685925035653e-006 + p * -2.5943389807429039e-008)))

    return specvol_SSO_0_CT25


# Salinity lib functions
def in_Baltic(lon, lat):
    """Check if positions are in the Baltic Sea

    Parameters
    ----------
    lon, lat : array_like or masked arrays

    Returns
    -------
    in_Baltic : boolean array (at least 1D)
                True for points in the Baltic Sea
                False for points outside, masked or NaN

    """
    lon, lat = np.atleast_1d(lon, lat)

    # Polygon bounding the Baltic, (xb, yb)
    # Effective boundary is the intersection of this polygon
    # with rectangle defined by xmin, xmax, ymin, ymax
    #

    # start with southwestern point and go round cyclonically
    xb = np.array([12.6, 45.0, 26.0,  7.0, 12.6])
    yb = np.array([50.0, 50.0, 69.0, 59.0, 50.0])

    # Enclosing rectangle
    #xmin, xmax = xb.min(), xb.max()
    #ymin, ymax = yb.min(), yb.max()
    xmin, xmax = 7.0, 32.0
    ymin, ymax = 52.0, 67.0

    # First check if outside the rectangle
    in_rectangle = ((xmin < lon) & (lon < xmax) &
                     (ymin < lat) & (lat < ymax))

    # Masked values are also considered outside the rectangle
    if np.ma.is_masked(in_rectangle):
        in_rectangle = in_rectangle.data & ~in_rectangle.mask

    # Closer check for points in the rectangle
    if np.any(in_rectangle):
        lon, lat = np.broadcast_arrays(lon, lat)
        in_baltic = np.zeros(lon.shape, dtype='bool')
        lon1 = lon[in_rectangle]
        lat1 = lat[in_rectangle]

        # There are general ways of testing for point in polygon
        # This works for this special configuration of points
        xx_right = np.interp(lat1, yb[1:3], xb[1:3])
        xx_left = np.interp(lat1, yb[-1:1:-1], xb[-1:1:-1])

        in_baltic[in_rectangle] = (xx_left <= lon1) & (lon1 <= xx_right)

        return in_baltic

    else:  # Nothing inside the rectangle, return the False array.
        return in_rectangle


def SP_from_SA_Baltic(SA, lon, lat):
    r"""Calculates Practical Salinity (SP) for the Baltic Sea, from a value
    computed analytically from Absolute Salinity.

    Parameters
    ----------
    SA : array_like
         Absolute salinity [g kg :sup:`-1`]
    lon : array_like
          decimal degrees east [0..+360]
    lat : array_like
          decimal degrees (+ve N, -ve S) [-90..+90]

    Returns
    -------
    SP_baltic : array_like
                salinity [psu (PSS-78)], unitless

    See Also
    --------
    SP_from_SA, SP_from_Sstar

    Notes
    -----
    This program will only produce Practical Salinity values for the Baltic
    Sea.

    Examples
    --------
    >>> import gsw.library as lib
    >>> SA = [6.6699, 6.7738, 6.9130, 7.3661, 7.5862, 10.3895]
    >>> lon, lat = 20, 59
    >>> lat = 59
    >>> lib.SP_from_SA_Baltic(SA, lon, lat)
    masked_array(data = [6.56825466873 6.67192351682 6.8108138311 7.26290579519 7.4825161269
     10.2795794748],
                 mask = [False False False False False False],
           fill_value = 1e+20)
    <BLANKLINE>

    References
    ----------
    .. [1] Feistel, R., S. Weinreben, H. Wolf, S. Seitz, P. Spitzer, B. Adel,
    G. Nausch, B. Schneider and D. G. Wright, 2010c: Density and Absolute
    Salinity of the Baltic Sea 2006-2009.  Ocean Science, 6, 3-24.
    http://www.ocean-sci.net/6/3/2010/os-6-3-2010.pdf

    .. [2] IOC, SCOR and IAPSO, 2010: The international thermodynamic equation
    of seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.

    .. [3] McDougall, T.J., D.R. Jackett and F.J. Millero, 2010: An algorithm
    for estimating Absolute Salinity in the global ocean. Submitted to Ocean
    Science. A preliminary version is available at Ocean Sci. Discuss.,
    6, 215-242.
    http://www.ocean-sci-discuss.net/6/215/2009/osd-6-215-2009-print.pdf

    Modifications:
    2010-07-23. David Jackett, Trevor McDougall & Paul Barker
    """
    SA, lon, lat = map(np.ma.masked_invalid, (SA, lon, lat))
    lon, lat, SA = np.broadcast_arrays(lon, lat, SA)

    inds_baltic = in_Baltic(lon, lat)

    if not inds_baltic.sum():
        return None

    SP_baltic = np.ma.masked_all(SA.shape, dtype=np.float)

    SP_baltic[inds_baltic] = ((35 / (SSO - 0.087)) *
                              (SA[inds_baltic] - 0.087))

    return SP_baltic


# FIXME: Check if this is still used and remove it.
def SP_from_SA_Baltic_old(SA, lon, lat):
    r"""Calculates Practical Salinity (SP) for the Baltic Sea, from a value
    computed analytically from Absolute Salinity.

    Parameters
    ----------
    SA : array_like
         Absolute salinity [g kg :sup:`-1`]
    lon : array_like
          decimal degrees east [0..+360]
    lat : array_like
          decimal degrees (+ve N, -ve S) [-90..+90]

    Returns
    -------
    SP_baltic : array_like
                salinity [psu (PSS-78)], unitless

    See Also
    --------
    SP_from_SA, SP_from_Sstar

    Notes
    -----
    This program will only produce Practical Salinity values for the Baltic
    Sea.

    Examples
    --------
    >>> import gsw.library as lib
    >>> SA = [6.6699, 6.7738, 6.9130, 7.3661, 7.5862, 10.3895]
    >>> lon, lat = 20, 59
    >>> lat = 59
    >>> lib.SP_from_SA_Baltic(SA, lon, lat)
    masked_array(data = [6.56825466873 6.67192351682 6.8108138311 7.26290579519 7.4825161269
     10.2795794748],
                 mask = [False False False False False False],
           fill_value = 1e+20)
    <BLANKLINE>

    References
    ----------
    .. [1] Feistel, R., S. Weinreben, H. Wolf, S. Seitz, P. Spitzer, B. Adel,
    G. Nausch, B. Schneider and D. G. Wright, 2010c: Density and Absolute
    Salinity of the Baltic Sea 2006-2009.  Ocean Science, 6, 3-24.
    http://www.ocean-sci.net/6/3/2010/os-6-3-2010.pdf

    .. [2] IOC, SCOR and IAPSO, 2010: The international thermodynamic equation
    of seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.

    .. [3] McDougall, T.J., D.R. Jackett and F.J. Millero, 2010: An algorithm
    for estimating Absolute Salinity in the global ocean. Submitted to Ocean
    Science. A preliminary version is available at Ocean Sci. Discuss.,
    6, 215-242.
    http://www.ocean-sci-discuss.net/6/215/2009/osd-6-215-2009-print.pdf

    Modifications:
    2010-07-23. David Jackett, Trevor McDougall & Paul Barker
    """
    SA, lon, lat = map(np.ma.masked_invalid, (SA, lon, lat))
    lon, lat, SA = np.broadcast_arrays(lon, lat, SA)

    xb1, xb2, xb3 = 12.6, 7., 26.
    xb1a, xb3a = 45., 26.
    yb1, yb2, yb3 = 50., 59., 69.

    inds_baltic = (xb2 < lon) & (lon < xb1a) & (yb1 < lat) & (lat < yb3)
    if not inds_baltic.sum():
        return None

    SP_baltic = np.ma.masked_all(SA.shape, dtype=np.float)

    xx_left = np.interp(lat[inds_baltic], [yb1, yb2, yb3], [xb1, xb2, xb3])
    xx_right = np.interp(lat[inds_baltic], [yb1, yb3], [xb1a, xb3a])

    inds_baltic1 = ((xx_left <= lon[inds_baltic])
                     & (lon[inds_baltic] <= xx_right))

    if not inds_baltic1.sum():
        return None

    SP_baltic[inds_baltic[inds_baltic1]] = ((35 / (SSO - 0.087))
                             * (SA[inds_baltic[inds_baltic1]] - 0.087))

    return SP_baltic


def SA_from_SP_Baltic(SP, lon, lat):
    r"""Computes absolute salinity from practical in the Baltic Sea.

    Parameters
    ----------
    SP : array_like or masked array
        Practical salinity (PSS-78)
    lon, lat : array_like or masked arrays
               geographical position

    Returns
    -------
    SA : masked array, at least 1D
         Absolute salinity   [g/kg]
         masked where inputs are masked or position outside the Baltic

    """

    # Handle masked array input
    input_mask = False
    if np.ma.is_masked(SP):
        input_mask = input_mask | SP.mask
    if np.ma.is_masked(lon):
        input_mask = input_mask | lon.mask
    if np.ma.is_masked(lat):
        input_mask = input_mask | lat.mask

    SP, lon, lat = map(np.atleast_1d, (SP, lon, lat))
    SP, lon, lat = np.broadcast_arrays(SP, lon, lat)

    inds_baltic = in_Baltic(lon, lat)

    #SA_baltic = np.ma.masked_all(SP.shape, dtype=np.float)

    all_nans = np.nan + np.zeros_like(SP)
    SA_baltic = np.ma.MaskedArray(all_nans, mask=~inds_baltic)

    if np.any(inds_baltic):
        SA_baltic[inds_baltic] = (((SSO - 0.087) / 35) *
                                    SP[inds_baltic] + 0.087)

    SA_baltic.mask = SA_baltic.mask | input_mask | np.isnan(SP)

    return SA_baltic


class SA_table(object):
    """
    TODO: Write docstring.
    """

    # Central America barrier
    x_ca = np.array([260.0, 272.59, 276.5, 278.65, 280.73, 295.217])
    y_ca = np.array([19.55, 13.97, 9.6, 8.1, 9.33, 0.0])

    def __init__(self, fname="gsw_data_v3_0.npz",
                        max_p_fudge=10000,
                        min_frac=0):
        self.fname = fname
        self.max_p_fudge = max_p_fudge
        self.min_frac = min_frac
        data = read_data(fname)
        # Make the order x, y, z:
        temp = data.delta_SA_ref.transpose((2, 1, 0)).copy()
        self.dsa = np.ma.masked_invalid(temp)
        self.dsa.data[self.dsa.mask] = 0
        self.lon = data.longs_ref.astype(np.float)
        self.lat = data.lats_ref.astype(np.float)
        self.p = data.p_ref                # Depth levels
        # ndepth from the file disagrees with the unmasked count from
        # delta_SA_ref in a few places; this should be fixed in the
        # file, but for now we will simply calculate ndepth directly from
        # delta_SA_ref.
        #self.ndepth = np.ma.masked_invalid(data.ndepth_ref.T).astype(np.int8)
        ndepth = self.dsa.count(axis=-1)
        self.ndepth = np.ma.masked_equal(ndepth, 0)
        self.dlon = self.lon[1] - self.lon[0]
        self.dlat = self.lat[1] - self.lat[0]
        self.i_ca, self.j_ca = self.xy_to_ij(self.x_ca, self.y_ca)

    def xy_to_ij(self, x, y):
        """
        Convert from lat/lon to grid index coordinates,
        without truncation or rounding.
        """
        i = (x - self.lon[0]) / self.dlon
        j = (y - self.lat[0]) / self.dlat
        return i, j

    def _central_america(self, di, dj, ii, jj, gm):
        """
        Use a line running through Central America to zero
        the goodmask for grid points in the Pacific forming
        the grid box around input locations in the Atlantic,
        and vice-versa.
        """
        ix, jy = ii[0] + di, jj[0] + dj  # Reconstruction: minor inefficiency.

        inear = ((ix >= self.i_ca[0]) & (ix <= self.i_ca[-1])
                 & (jy >= self.j_ca[-1]) & (jy <= self.j_ca[0]))
        if not inear.any():
            return gm

        inear_ind = inear.nonzero()[0]
        ix = ix[inear]
        jy = jy[inear]
        ii = ii[:, inear]
        jj = jj[:, inear]

        jy_ca = np.interp(ix, self.i_ca, self.j_ca)
        above = jy - jy_ca  # > 0 if input point is above dividing line

        # Intersections of left and right grid lines with dividing line
        jleft_ca = np.interp(ii[0], self.i_ca, self.j_ca)
        jright_ca = np.interp(ii[1], self.i_ca, self.j_ca)

        jgrid_ca = [jleft_ca, jright_ca, jright_ca, jleft_ca]

        # Zero the goodmask for grid points on opposite side of divider
        for i in range(4):
            opposite = (above * (jj[i] - jgrid_ca[i])) < 0
            gm[i, inear_ind[opposite]] = 0

        return gm

    def xy_interp(self, di, dj, ii, jj, k):
        """
        2-D interpolation, bilinear if all 4 surrounding
        grid points are present, but treating missing points
        as having the average value of the remaining grid
        points. This matches the matlab V2 behavior.
        """
        # Array of weights, CCW around the grid box
        w = np.vstack(((1 - di) * (1 - dj),  # lower left
                      di * (1 - dj),         # lower right
                      di * dj,               # upper right
                      (1 - di) * dj))        # upper left

        gm = ~self.dsa.mask[ii, jj, k]   # gm is "goodmask"
        gm = self._central_america(di, dj, ii, jj, gm)

        # Save a measure of real interpolation quality.
        frac = (w * gm).sum(axis=0)

        # Now loosen the interpolation, allowing a value to
        # be calculated on a grid point that is masked.
        # This matches the matlab gsw version 2 behavior.

        jm_partial = gm.any(axis=0) & (~(gm.all(axis=0)))

        # The weights of the unmasked points will be increased
        # by the sum of the weights of the masked points divided
        # by the number of unmasked points in the grid square.
        # This is equivalent to setting the masked data values
        # to the average of the unmasked values, and then
        # unmasking, which is the matlab v2 implementation.

        if jm_partial.any():
            w_bad = w * (~gm)
            w[:, jm_partial] += (w_bad[:, jm_partial].sum(axis=0) /
                                             gm[:, jm_partial].sum(axis=0))
        w *= gm

        wsum = w.sum(axis=0)

        valid = wsum > 0  # Only need to prevent division by zero here.
        w[:, valid] /= wsum[:, valid]
        w[:, ~valid] = 0
        vv = self.dsa.data[ii, jj, k]
        vv *= w
        dsa = vv.sum(axis=0)
        return dsa, frac

    def delta_SA(self, p, lon, lat):
        r"""Table lookup of salinity anomaly, given pressure, lon, and lat."""

        p = np.ma.masked_less(p, 0)
        mask_in = np.ma.mask_or(np.ma.getmask(p), np.ma.getmask(lon))
        mask_in = np.ma.mask_or(mask_in, np.ma.getmask(lat))
        p, lon, lat = [np.ma.filled(a, 0).astype(float) for a in (p, lon, lat)]

        p, lon, lat = np.broadcast_arrays(p, lon, lat)
        if p.ndim > 1:
            shape_in = p.shape
            p, lon, lat = map(np.ravel, (p, lon, lat))
            reshaped = True
        else:
            reshaped = False

        p_orig = p.copy()  # Save for comparison to clipped p.

        ix0, iy0 = self.xy_to_ij(lon, lat)
        i0raw = np.floor(ix0).astype(int)
        i0 = np.clip(i0raw, 0, len(self.lon) - 2)
        di = ix0 - i0
        j0raw = np.floor(iy0).astype(int)
        j0 = np.clip(j0raw, 0, len(self.lat) - 2)
        dj = iy0 - j0

        # Start at lower left and go CCW; match order in _xy_interp.
        ii = np.vstack((i0, i0 + 1, i0 + 1, i0))
        jj = np.vstack((j0, j0, j0 + 1, j0 + 1))

        k1 = np.searchsorted(self.p, p, side='right')

        # Clip p and k1 at max p of grid cell.
        kmax = (self.ndepth[ii, jj].max(axis=0) - 1)
        mask_out = kmax.mask
        kmax = kmax.filled(1)
        clip_p = (p >= self.p[kmax])
        p[clip_p] = self.p[kmax[clip_p]]
        k1[clip_p] = kmax[clip_p]

        k0 = k1 - 1

        dsa0, frac0 = self.xy_interp(di, dj, ii, jj, k0)
        dsa1, frac1 = self.xy_interp(di, dj, ii, jj, k1)

        dp = np.diff(self.p)
        pfrac = (p - self.p[k0]) / dp[k0]
        delta_SA = dsa0 * (1 - pfrac) + dsa1 * pfrac

        # Save intermediate results in case we are curious about
        # them; the frac values are most likely to be useful.
        # We won't bother to reshape them, though, and we may
        # delete them later.
        self.dsa0 = dsa0
        self.frac0 = frac0
        self.dsa1 = dsa1
        self.frac1 = frac1
        self.pfrac = pfrac

        self.p_fudge = p_orig - p

        # Editing options, in case we don't want to use
        # values calculated from the wrong pressure, or from
        # an incomplete SA table grid square.
        mask_out |= self.p_fudge > self.max_p_fudge
        mask_out |= self.frac1 < self.min_frac

        delta_SA = np.ma.array(delta_SA, mask=mask_out, copy=False)
        if reshaped:
            delta_SA.shape = shape_in
            self.p_fudge.shape = shape_in
        if mask_in is not np.ma.nomask:
            delta_SA = np.ma.array(delta_SA, mask=mask_in, copy=False)
        return delta_SA


@match_args_return
def SAAR(p, lon, lat):
    r"""Absolute Salinity Anomaly Ratio (excluding the Baltic Sea).

    Calculates the Absolute Salinity Anomaly Ratio, SAAR, in the open ocean
    by spatially interpolating the global reference data set of SAAR to the
    location of the seawater sample.

    This function uses version 3.0 of the SAAR look up table.

    Parameters
    ----------
    p : array_like
        pressure [dbar]
    lon : array_like
          decimal degrees east (will be treated modulo 360)
    lat : array_like
          decimal degrees (+ve N, -ve S) [-90..+90]

    Returns
    -------
    SAAR : masked array; masked where no nearby ocean is found in data
           Absolute Salinity Anomaly Ratio [unitless] FIXME: [g kg :sup:`-1`]?

    Notes
    -----
    The Absolute Salinity Anomaly Ratio in the Baltic Sea is evaluated
    separately, since it is a function of Practical Salinity, not of space.
    The present function returns a SAAR of zero for data in the Baltic Sea.
    The correct way of calculating Absolute Salinity in the Baltic Sea is by
    calling SA_from_SP.

    The mask is only set when the observation is well and truly on dry
    land; often the warning flag is not set until one is several hundred
    kilometers inland from the coast.

    References
    ----------
    .. [1] IOC, SCOR and IAPSO, 2010: The international thermodynamic equation
    of seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.

    .. [2] McDougall, T.J., D.R. Jackett and F.J. Millero, 2010: An algorithm
    for estimating Absolute Salinity in the global ocean.  Submitted to Ocean
    Science. A preliminary version is available at Ocean Sci. Discuss.,
    6, 215-242.
    http://www.ocean-sci-discuss.net/6/215/2009/osd-6-215-2009-print.pdf

    The algorithm is taken from the matlab implementation of the references,
    but the numpy implementation here differs substantially from the
    matlab implementation.

    Modifications:
    """

    #FIXME: Compare old delta_SA with new SAAR.
    return SA_table().delta_SA(p, lon, lat)


def infunnel(SA, CT, p):
    r"""Oceanographic funnel check for the 25-term equation

    Parameters
    ----------
    SA : array_like
         Absolute Salinity            [g/kg]
    CT : array_like
         Conservative Temperature     [°C]
    p  : array_like
         sea pressure                 [dbar]
           (ie. absolute pressure - 10.1325 dbar)

    Returns
    -------
    in_funnel : boolean ndarray or scalar
        True,  if SA, CT and p are inside the "funnel"
        False, if SA, CT and p are outside the "funnel",
               or one of the values are NaN or masked

    Note. The term "funnel" describes the range of SA, CT and p over which
    the error in the fit of the computationally-efficient 25-term
    expression for density in terms of SA, CT and p was calculated
    (McDougall et al., 2010).

    author:
    Trevor McDougall and Paul Barker
    2011-02-27: Bjørn Ådlandsvik, python version
    """

    # Check variables and resize if necessary
    scalar = np.isscalar(SA) and np.isscalar(CT) and np.isscalar(p)
    SA, CT, p = np.broadcast_arrays(SA, CT, p)

    input_nan = np.isnan(SA) | np.isnan(CT) | np.isnan(p)

    infunnel = ((p <= 8000) &
                (SA >= 0) &
                (SA <= 42.2) &
                (CT >= (-0.3595467 - 0.0553734 * SA)) &
                ((p >= 5500) | (SA >= 0.006028 * (p - 500))) &
                ((p >= 5500) | (CT <= (33.0 - 0.003818181818182 * p))) &
                ((p <= 5500) | (SA >= 30.14)) &
                ((p <= 5500) | (CT <= 12.0)))

    infunnel = infunnel & np.logical_not(input_nan)

    if scalar:
        infunnel = bool(infunnel)

    return infunnel


@match_args_return
def Hill_ratio_at_SP2(t):
    r"""TODO: Write docstring
    Hill ratio at SP = 2
    """

    # USAGE:
    #  Hill_ratio = Hill_ratio_at_SP2(t)
    #
    # DESCRIPTION:
    #  Calculates the Hill ratio, which is the adjustment needed to apply for
    #  Practical Salinities smaller than 2.  This ratio is defined at a
    #  Practical Salinity = 2 and in-situ temperature, t using PSS-78. The Hill
    #  ratio is the ratio of 2 to the output of the Hill et al. (1986) formula
    #  for Practical Salinity at the conductivity ratio, Rt, at which Practical
    #  Salinity on the PSS-78 scale is exactly 2.
    #
    # INPUT:
    #  t  =  in-situ temperature (ITS-90)                  [ deg C ]
    #
    # OUTPUT:
    #  Hill_ratio  =  Hill ratio at SP of 2                [ unitless ]
    #
    # AUTHOR:
    #  Trevor McDougall and Paul Barker
    #
    # VERSION NUMBER: 3.0 (26th March, 2011)

    SP2 = 2 * np.ones_like(t)

    #------------------------------
    # Start of the calculation
    #------------------------------

    a0 = 0.0080
    a1 = -0.1692
    a2 = 25.3851
    a3 = 14.0941
    a4 = -7.0261
    a5 = 2.7081

    b0 = 0.0005
    b1 = -0.0056
    b2 = -0.0066
    b3 = -0.0375
    b4 = 0.0636
    b5 = -0.0144

    g0 = 2.641463563366498e-1
    g1 = 2.007883247811176e-4
    g2 = -4.107694432853053e-6
    g3 = 8.401670882091225e-8
    g4 = -1.711392021989210e-9
    g5 = 3.374193893377380e-11
    g6 = -5.923731174730784e-13
    g7 = 8.057771569962299e-15
    g8 = -7.054313817447962e-17
    g9 = 2.859992717347235e-19

    k = 0.0162

    t68 = t * 1.00024
    ft68 = (t68 - 15) / (1 + k * (t68 - 15))

    #--------------------------------------------------------------------------
    # Find the initial estimates of Rtx (Rtx0) and of the derivative dSP_dRtx
    # at SP = 2.
    #--------------------------------------------------------------------------

    Rtx0 = g0 + t68 * (g1 + t68 * (g2 + t68 * (g3 + t68 * (g4 + t68 * (g5
              + t68 * (g6 + t68 * (g7 + t68 * (g8 + t68 * g9))))))))

    dSP_dRtx = (a1 + (2 * a2 + (3 * a3 + (4 * a4 + 5 * a5 * Rtx0) * Rtx0) *
    Rtx0) * Rtx0 + ft68 * (b1 + (2 * b2 + (3 * b3 + (4 * b4 + 5 * b5 * Rtx0) *
    Rtx0) * Rtx0) * Rtx0))

    #--------------------------------------------------------------------------
    # Begin a single modified Newton-Raphson iteration to find Rt at SP = 2.
    #--------------------------------------------------------------------------

    SP_est = (a0 + (a1 + (a2 + (a3 + (a4 + a5 * Rtx0) * Rtx0) * Rtx0) * Rtx0) *
    Rtx0 + ft68 * (b0 + (b1 + (b2 + (b3 + (b4 + b5 * Rtx0) * Rtx0) * Rtx0) *
    Rtx0) * Rtx0))

    Rtx = Rtx0 - (SP_est - SP2) / dSP_dRtx
    Rtxm = 0.5 * (Rtx + Rtx0)
    dSP_dRtx = (a1 + (2 * a2 + (3 * a3 + (4 * a4 + 5 * a5 * Rtxm) * Rtxm) *
    Rtxm) * Rtxm + ft68 * (b1 + (2 * b2 + (3 * b3 + (4 * b4 + 5 * b5 * Rtxm) *
    Rtxm) * Rtxm) * Rtxm))

    Rtx = Rtx0 - (SP_est - SP2) / dSP_dRtx

    # This is the end of one full iteration of the modified Newton-Raphson
    # iterative equation solver.  The error in Rtx at this point is equivalent
    # to an error in SP of 9e-16 psu.

    x = 400 * Rtx * Rtx
    sqrty = 10 * Rtx
    part1 = 1 + x * (1.5 + x)
    part2 = 1 + sqrty * (1 + sqrty * (1 + sqrty))
    SP_Hill_raw_at_SP2 = SP2 - a0 / part1 - b0 * ft68 / part2

    return 2. / SP_Hill_raw_at_SP2


def interp_S_T(S, T, z, znew, P=None):
    r"""Linear interpolation of ndarrays *S* and *T* from *z* to *znew*.
    Optionally interpolate a third ndarray, *P*.

    *z* must be strictly increasing or strictly decreasing.  It must
    be a 1-D array, and its length must match the last dimension
    of *S* and *T*.

    *znew* may be a scalar or a sequence.

    It is assumed, but not checked, that *S*, *T*, and *z* are
    all plain ndarrays, not masked arrays or other sequences.

    Out-of-range values of *znew*, and *nan* in *S* and *T*,
    yield corresponding *nan* in the output.

    The basic algorithm is from scipy.interpolate.
    """

    isscalar = False
    if not np.iterable(znew):
        isscalar = True
        znew = [znew]
    znew = np.asarray(znew)

    inverted = False
    if z[1] - z[0] < 0:
        inverted = True
        z = z[::-1]
        S = S[..., ::-1]
        T = T[..., ::-1]
        if P is not None:
            P = P[..., ::-1]

    if (np.diff(z) <= 0).any():
        raise ValueError("z must be strictly increasing or decreasing")

    hi = np.searchsorted(z, znew)
    hi = hi.clip(1, len(z) - 1).astype(int)
    lo = hi - 1

    z_lo = z[lo]
    z_hi = z[hi]
    S_lo = S[lo]
    S_hi = S[hi]
    T_lo = T[lo]
    T_hi = T[hi]
    zratio = (znew - z_lo) / (z_hi - z_lo)

    Si = S_lo + (S_hi - S_lo) * zratio
    Ti = T_lo + (T_hi - T_lo) * zratio
    if P is not None:
        Pi = P[lo] + (P[hi] - P[lo]) * zratio

    if inverted:
        Si = Si[..., ::-1]
        Ti = Ti[..., ::-1]
        if P is not None:
            Pi = Pi[..., ::-1]

    outside = (znew < z.min()) | (znew > z.max())
    if np.any(outside):
        Si[..., outside] = np.nan
        Ti[..., outside] = np.nan
        if P is not None:
            Pi[..., outside] = np.nan

    if isscalar:
        Si = Si[0]
        Ti = Ti[0]
        if P is not None:
            Pi = Pi[0]

    if P is None:
        return Si, Ti
    return Si, Ti, Pi


def interp_SA_CT(SA, CT, p, p_i):
    r"""TODO: Write docstring.
    function [SA_i, CT_i] = interp_SA_CT(SA,CT,p,p_i)
    interp_SA_CT                    linear interpolation to p_i on a cast
    ==========================================================================
    This function interpolates the cast with respect to the interpolating
    variable p. This function finds the values of SA, CT at p_i on this cast.
    """
    return interp_S_T(SA, CT, p, p_i)


def interp_ref_cast(spycnl, A="gn"):
    r"""Translation of:

    [SA_iref_cast, CT_iref_cast, p_iref_cast] = interp_ref_cast(spycnl, A)

    interp_ref_cast            linear interpolation of the reference cast
    ==========================================================================
    This function interpolates the reference cast with respect to the
    interpolating variable "spycnl".  This reference cast is at the location
    188E,4N from the reference data set which underlies the Jackett &
    McDougall (1997) Neutral Density computer code.  This function finds the
    values of SA, CT and p on this reference cast which correspond to the
    value of isopycnal which is passed to this function from the function
    "geo_strf_isopycnal_CT".  The isopycnal could be either gamma_n or
    sigma_2. If A is set to any of the following 's2','S2','sigma2','sigma_2'
    the interpolation will take place in sigma 2 space, any other input
    will result in the programme working in gamma_n space.

    VERSION NUMBER: 3.0 (14th April, 2011)

    REFERENCE:
    Jackett, D. R. and T. J. McDougall, 1997: A neutral density variable
    for the world<92>s oceans. Journal of Physical Oceanography, 27, 237-263.

    FIXME? Do we need argument checking here to handle masked arrays,
    etc.?  I suspect not, since I don't think this is intended to be
    user-callable, but is instead used internally by user-callable
    functions.

    """

    if A.lower() in ["s2", "sigma2", "sigma_2"]:
        A = "s2"

    gsw_data = read_data("gsw_data_v3_0.npz")

    SA_ref = gsw_data.SA_ref_cast
    CT_ref = gsw_data.CT_ref_cast
    p_ref = gsw_data.p_ref_cast
    if A == "s2":
        zvar_ref = gsw_data.sigma_2_ref_cast
    else:
        zvar_ref = gsw_data.gamma_n_ref_cast

    # Not sure why this is needed, but it is in the Matlab version,
    # and presumably can't hurt.
    cond = (spycnl >= 21.805) & (spycnl <= 28.3614)
    zvar_new = spycnl[cond]

    Si, Ci, Pi = interp_S_T(SA_ref, CT_ref, zvar_ref, zvar_new, P=p_ref)

    return Si, Ci, Pi


def enthalpy_SSO_0_p(p):
    r"""This function calculates enthalpy at the Standard Ocean Salinty, SSO,
    and at a Conservative Temperature of zero degrees C, as a function of
    pressure, p, in dbar, using a streamlined version of the 48-term CT
    version of the Gibbs function, that is, a streamlined version of the
    code "enthalpy(SA,CT,p).

    Modifications:
    """

    v01 = 9.998420897506056e+2
    v05 = -6.698001071123802
    v08 = -3.988822378968490e-2
    v12 = -2.233269627352527e-2
    v15 = -1.806789763745328e-4
    v17 = -3.087032500374211e-7
    v20 = 1.550932729220080e-10
    v21 = 1.0
    v26 = -7.521448093615448e-3
    v31 = -3.303308871386421e-5
    v36 = 5.419326551148740e-6
    v37 = -2.742185394906099e-5
    v41 = -1.105097577149576e-7
    v43 = -1.119011592875110e-10
    v47 = -1.200507748551599e-15

    a0 = v21 + SSO * (v26 + v36 * SSO + v31 * np.sqrt(SSO))

    a1 = v37 + v41 * SSO

    a2 = v43

    a3 = v47

    b0 = v01 + SSO * (v05 + v08 * np.sqrt(SSO))

    b1 = 0.5 * (v12 + v15 * SSO)

    b2 = v17 + v20 * SSO

    b1sq = b1 ** 2

    sqrt_disc = np.sqrt(b1sq - b0 * b2)

    N = a0 + (2 * a3 * b0 * b1 / b2 - a2 * b0) / b2

    M = a1 + (4 * a3 * b1sq / b2 - a3 * b0 - 2 * a2 * b1) / b2

    A = b1 - sqrt_disc
    B = b1 + sqrt_disc

    part = (N * b2 - M * b1) / (b2 * (B - A))

    return db2Pascal * (p * (a2 - 2 * a3 * b1 / b2 + 0.5 * a3 * p) /
                        b2 + (M / (2 * b2)) * np.log(1 + p *
                        (2 * b1 + b2 * p) / b0) + part *
                        np.log(1 + (b2 * p * (B - A)) / (A * (B + b2 * p))))


def specvol_SSO_0_p(p):
    r"""This function calculates specific volume at the Standard Ocean
    Salinity, SSO, and at a Conservative Temperature of zero degrees C, as a
    function of pressure, p, in dbar, using a streamlined version of the
    48-term CT version of specific volume, that is, a streamlined version of
    the code "specvol(SA, CT, p)".

    Modifications:
    """

    v01 = 9.998420897506056e+2
    v05 = -6.698001071123802
    v08 = -3.988822378968490e-2
    v12 = -2.233269627352527e-2
    v15 = -1.806789763745328e-4
    v17 = -3.087032500374211e-7
    v20 = 1.550932729220080e-10
    v21 = 1.0
    v26 = -7.521448093615448e-3
    v31 = -3.303308871386421e-5
    v36 = 5.419326551148740e-6
    v37 = -2.742185394906099e-5
    v41 = -1.105097577149576e-7
    v43 = -1.119011592875110e-10
    v47 = -1.200507748551599e-15

    return ((v21 + SSO * (v26 + v36 * SSO + v31 * np.sqrt(SSO)) + p *
           (v37 + v41 * SSO + p * (v43 + v47 * p))) / (v01 + SSO * (v05 + v08 *
           np.sqrt(SSO)) + p * (v12 + v15 * SSO + p * (v17 + v20 * SSO))))
