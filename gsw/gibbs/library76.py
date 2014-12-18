
# coding: utf-8

# In[1]:

from __future__ import division

import numpy as np
import freezing as fr
from utilities import match_args_return


# In[2]:

# This module contains the following definitions (in alphabetical order)
#          ['alpha', 
#          'alpha_on_beta',
#          'beta',
#          'cabbeling',
#          'CT_freezing_poly'
#          'CT_from_ehthalpy'
#          'CT_from_rho'
#          'CT_maxdensity',
#          'dynamic_enthalpy',
#          'enthalpy',
#          'enthalpy_diff',
#          'enthalpy_first_derivatives',
#          'enthalpy_second_derivatives',
#          'enthalpy_SSO_0_p',
#          'kappa',
#          'internal_energy',
#          'rho',
#          'rho_alpha_beta',
#          'rho_first_derivatives',
#          'rho_second_derivatives',
#          'rho_first_derivatives_wrt_enthalpy',
#          'rho_second_derivatives_wrt_enthalpy',
#          'SA_from_rho',
#          'sigma0',
#          'sigma1',
#          'sigma2',
#          'sigma3',
#          'sigma4',
#          'sound_speed',
#          'specvol',
#          'specvol_alpha_beta',
#          'specvol_anom',
#          'specvol_first_derivatives',
#          'specvol_second_derivatives',
#          'specvol_first_derivatives_wrt_enthalpy',
#          'specvol_second_derivatives_wrt_enthalpy',
#          'specvol_SSO_0_p',
#          'spiciness0',
#          'spiciness1',
#          'spiciness2',
#          'thermobaric']


# In[3]:

# list of all coefficients belonging to all polynomials used throughout 
# this module which have been fitted using the 76-term expression 
# for specific volume

# alpha coefficients from gsw_rho_alpha_beta
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

# beta coefficients from gsw_rho_alpha_beta
b000 = -3.1038981976e-4 
b001 =  2.4262468747e-5 
b002 = -5.8484432984e-7 
b003 =  3.6310188515e-7 
b004 = -1.1147125423e-7 
b010 =  3.5009599764e-5 
b011 = -9.5677088156e-6 
b012 = -5.5699154557e-6 
b013 = -2.7295696237e-7 
b020 = -3.7435842344e-5 
b021 = -2.3678308361e-7 
b022 =  3.9137387080e-7 
b030 =  2.4141479483e-5 
b031 = -3.4558773655e-6 
b032 =  7.7618888092e-9 
b040 = -8.7595873154e-6 
b041 =  1.2956717783e-6 
b050 = -3.3052758900e-7 
b100 =  1.33856134076e-3 
b101 = -6.9584921948e-5 
b102 = -9.62445031940e-6 
b103 =  3.3492607560e-8 
b110 = -8.7185357122e-5 
b111 =  2.2201669530e-5 
b112 =  1.09241497668e-5 
b120 =  7.1815645520e-5 
b121 =  5.8566692590e-6 
b122 = -1.31462208134e-6 
b130 = -2.8707266096e-5 
b131 =  6.3310612156e-7 
b140 =  8.7407361196e-6 
b200 = -2.55143801811e-3 
b201 =  1.12412331915e-4 
b202 =  1.47789320994e-5 
b210 =  1.03597385484e-4 
b211 = -2.95341353532e-5 
b212 = -4.0632556881e-6 
b220 = -5.6095752561e-5 
b221 = -1.4647841760e-6 
b230 =  6.8589973668e-6 
b300 =  2.32344279772e-3 
b301 = -6.9288874448e-5 
b302 = -7.1247898908e-6 
b310 = -4.7837639152e-5 
b311 =  1.0363690104e-5 
b320 =  1.54381356976e-5 
b400 = -1.05461852535e-3 
b401 =  1.54637136265e-5 
b410 =  6.9322972905e-6 
b500 =  1.9159474383e-4 

# CT_freezing poly coefficients from gsw_CT_freezing_poly
c0  =  0.017947064327968736
c1 =  -6.076099099929818
c2 =   4.883198653547851
c3 =  -11.88081601230542
c4 =   13.34658511480257
c5 =  -8.722761043208607
c6 =   2.082038908808201
c7 =  -7.389420998107497
c8 =  -2.110913185058476
c9 =   0.2295491578006229   
c10 = -0.9891538123307282
c11 = -0.08987150128406496
c12 =  0.3831132432071728
c13 =  1.054318231187074
c14 =  1.065556599652796
c15 = -0.7997496801694032
c16 =  0.3850133554097069
c17 = -2.078616693017569
c18 =  0.8756340772729538
c19 = -2.079022768390933
c20 =  1.596435439942262
c21 =  0.1338002171109174
c22 =  1.242891021876471

# kappa coefficients from gsw_kappa
c000 = -6.0799143809e-5 
c001 =  1.99712338438e-5 
c002 = -3.3928084311e-6 
c003 =  4.2124612320e-7 
c004 = -6.3236306430e-8 
c005 =  1.1768102358e-8 
c010 =  1.8505765429e-5 
c011 = -2.3472773462e-6 
c012 = -1.09581019659e-6 
c013 =  1.25816399608e-6 
c020 = -1.1716606853e-5 
c021 =  4.2610057480e-6 
c022 =  8.6087715477e-7 
c030 =  7.9279656173e-6 
c031 = -9.2265080074e-7 
c040 = -3.4102187482e-6 
c041 = -1.26705833028e-7 
c050 =  5.0736766814e-7 
c100 =  2.4262468747e-5 
c101 = -1.16968865968e-6 
c102 =  1.08930565545e-6 
c103 = -4.4588501692e-7 
c110 = -9.5677088156e-6 
c111 = -1.11398309114e-5 
c112 = -8.1887088711e-7 
c120 = -2.3678308361e-7 
c121 =  7.8274774160e-7 
c130 = -3.4558773655e-6 
c131 =  1.55237776184e-8 
c140 =  1.2956717783e-6 
c200 = -3.4792460974e-5 
c201 = -9.6244503194e-6 
c202 =  5.0238911340e-8 
c210 =  1.1100834765e-5 
c211 =  1.09241497668e-5 
c220 =  2.9283346295e-6 
c221 = -1.31462208134e-6 
c230 =  3.1655306078e-7 
c300 =  3.7470777305e-5 
c301 =  9.8526213996e-6 
c310 = -9.8447117844e-6 
c311 = -2.7088371254e-6 
c320 = -4.8826139200e-7 
c400 = -1.7322218612e-5 
c401 = -3.5623949454e-6 
c410 =  2.5909225260e-6 
c500 =  3.0927427253e-6 

# enthalpy coefficients from gsw_enthalpy
h001 =  1.0769995862e-3 
h002 = -3.0399571905e-5 
h003 =  3.3285389740e-6 
h004 = -2.8273403593e-7 
h005 =  2.1062306160e-8 
h006 = -2.1078768810e-9 
h007 =  2.8019291329e-10 
h011 = -1.5649734675e-5 
h012 =  9.2528827145e-6 
h013 = -3.9121289103e-7 
h014 = -9.1317516383e-8 
h015 =  6.2908199804e-8 
h021 =  2.7762106484e-5 
h022 = -5.8583034265e-6 
h023 =  7.1016762467e-7 
h024 =  7.1739762898e-8 
h031 = -1.6521159259e-5 
h032 =  3.9639828087e-6 
h033 = -1.5377513346e-7 
h042 = -1.7051093741e-6 
h043 = -2.1117638838e-8 
h041 =  6.9111322702e-6 
h051 = -8.0539615540e-7 
h052 =  2.5368383407e-7 
h061 =  2.0543094268e-7 
h101 = -3.1038981976e-4 
h102 =  1.21312343735e-5 
h103 = -1.9494810995e-7 
h104 =  9.0775471288e-8 
h105 = -2.2294250846e-8 
h111 =  3.5009599764e-5 
h112 = -4.7838544078e-6 
h113 = -1.8566384852e-6 
h114 = -6.8239240593e-8 
h121 = -3.7435842344e-5 
h122 = -1.18391541805e-7 
h123 =  1.3045795693e-7 
h131 =  2.4141479483e-5 
h132 = -1.72793868275e-6 
h133 =  2.5872962697e-9 
h141 = -8.7595873154e-6 
h142 =  6.4783588915e-7 
h151 = -3.3052758900e-7
h201 =  6.6928067038e-4 
h202 = -1.7396230487e-5 
h203 = -1.6040750532e-6 
h204 =  4.1865759450e-9 
h211 = -4.3592678561e-5 
h212 =  5.5504173825e-6 
h213 =  1.8206916278e-6 
h221 =  3.5907822760e-5 
h222 =  1.46416731475e-6 
h223 = -2.1910368022e-7 
h231 = -1.4353633048e-5 
h232 =  1.5827653039e-7 
h241 =  4.3703680598e-6
h301 = -8.5047933937e-4 
h302 =  1.87353886525e-5 
h303 =  1.6421035666e-6 
h311 =  3.4532461828e-5 
h312 = -4.9223558922e-6 
h313 = -4.5147285423e-7 
h321 = -1.8698584187e-5 
h322 = -2.4413069600e-7 
h331 =  2.2863324556e-6
h401 =  5.8086069943e-4 
h402 = -8.6611093060e-6 
h403 = -5.9373249090e-7 
h411 = -1.1959409788e-5 
h421 =  3.8595339244e-6 
h412 =  1.2954612630e-6
h501 = -2.1092370507e-4 
h502 =  1.54637136265e-6 
h511 =  1.3864594581e-6 
h601 =  3.1932457305e-5

# spiciness0 coefficients from gsw_spiciness0. NOTE: all spiciness0 
# coefficients have an s0 prefix
s001 = -9.22982898371678e1
s002 = -1.35727873628866e1
s003 =  1.87353650994010e1
s004 = -1.61360047373455e1
s005 =  3.76112762286425e1
s006 = -4.27086671461257e1
s007 =  2.00820111041594e1
s008 =  2.87969717584045e2
s009 =  1.13747111959674e1
s010 =  6.07377192990680e1
s011 = -7.37514033570187e1
s012 = -7.51171878953574e1
s013 =  1.63310989721504e2
s014 = -8.83222751638095e1
s015 = -6.41725302237048e2
s016 =  2.79732530789261e1
s017 = -2.49466901993728e2
s018 =  3.26691295035416e2
s019 =  2.66389243708181e1
s020 = -2.93170905757579e2
s021 =  1.76053907144524e2
s022 =  8.27634318120224e2
s023 = -7.02156220126926e1
s024 =  3.82973336590803e2
s025 = -5.06206828083959e2
s026 =  6.69626565169529e1
s027 =  3.02851235050766e2
s028 = -1.96345285604621e2
s029 = -5.74040806713526e2
s030 =  7.03285905478333e1
s031 = -2.97870298879716e2
s032 =  3.88340373735118e2
s033 = -8.29188936089122e1
s034 = -1.87602137195354e2
s035 =  1.27096944425793e2
s036 =  2.11671167892147e2
s037 = -3.15140919876285e1
s038 =  1.16458864953602e2
s039 = -1.50029730802344e2
s040 =  3.76293848660589e1
s041 =  6.47247424373200e1
s042 = -4.47159994408867e1
s043 = -3.23533339449055e1
s044 =  5.30648562097667
s045 = -1.82051249177948e1
s046 =  2.33184351090495e1
s047 = -6.22909903460368
s048 = -9.55975464301446
s049 =  6.61877073960113

# spiciness1 coefficients from gsw_spiciness1. NOTE: all spiciness1 
# coefficients have an s1 prefix
s101 = -9.19874584868912e1
s102 = -1.33517268529408e1
s103 =  2.18352211648107e1
s104 = -2.01491744114173e1
s105 =  3.70004204355132e1
s106 = -3.78831543226261e1
s107 =  1.76337834294554e1
s108 =  2.87838842773396e2
s109 =  2.14531420554522e1
s110 =  3.14679705198796e1
s111 = -4.04398864750692e1
s112 = -7.70796428950487e1
s113 =  1.36783833820955e2
s114 = -7.36834317044850e1
s115 = -6.41753415180701e2
s116 =  1.33701981685590
s117 = -1.75289327948412e2
s118 =  2.42666160657536e2
s119 =  3.17062400799114e1
s120 = -2.28131490440865e2
s121 =  1.39564245068468e2
s122 =  8.27747934506435e2
s123 = -3.50901590694775e1
s124 =  2.87473907262029e2
s125 = -4.00227341144928e2
s126 =  6.48307189919433e1
s127 =  2.16433334701578e2
s128 = -1.48273032774305e2
s129 = -5.74545648799754e2
s130 =  4.50446431127421e1
s131 = -2.30714981343772e2
s132 =  3.15958389253065e2
s133 = -8.60635313930106e1
s134 = -1.22978455069097e2
s135 =  9.18287282626261e1
s136 =  2.12120473062203e2
s137 = -2.21528216973820e1
s138 =  9.19013417923270e1
s139 = -1.24400776026014e2
s140 =  4.08512871163839e1
s141 =  3.91127352213516e1
s142 = -3.10508021853093e1
s143 = -3.24790035899152e1
s144 =  3.91029016556786
s145 = -1.45362719385412e1
s146 =  1.96136194246355e1
s147 = -7.06035474689088
s148 = -5.36884688614009
s149 =  4.43247303092448

# spiciness2 coefficients from gsw_spiciness2. NOTE: all spiciness2 
# coefficients have an s2 prefix
s201 = -9.17327320732265e1
s202 = -1.31200235147912e1
s203 =  2.49574345782503e1
s204 = -2.41678075247398e1
s205 =  3.61654631402053e1
s206 = -3.22582164667710e1
s207 =  1.45092623982509e1
s208 =  2.87776645983195e2
s209 =  3.13902307672447e1
s210 =  1.69777467534459
s211 = -5.69630115740438
s212 = -7.97586359017987e1
s213 =  1.07507460387751e2
s214 = -5.58234404964787e1
s215 = -6.41708068766557e2
s216 = -2.53494801286161e1
s217 = -9.86755437385364e1
s218 =  1.52406930795842e2
s219 =  4.23888258264105e1
s220 = -1.60118811141438e2
s221 =  9.67497898053989e1
s222 =  8.27674355478637e2
s223 =  5.27561234412133e-1
s224 =  1.87440206992396e2
s225 = -2.83295392345171e2
s226 =  5.14485994597635e1
s227 =  1.29975755062696e2
s228 = -9.36526588377456e1
s229 = -5.74911728972948e2
s230 =  1.91175851862772e1
s231 = -1.59347231968841e2
s232 =  2.33884725744938e2
s233 = -7.87744010546157e1
s234 = -6.04757235443685e1
s235 =  5.27869695599657e1
s236 =  2.12517758478878e2
s237 = -1.24351794740528e1
s238 =  6.53904308937490e1
s239 = -9.44804080763788e1
s240 =  3.93874257887364e1
s241 =  1.49425448888996e1
s242 = -1.62350721656367e1
s243 = -3.25936844276669e1
s244 =  2.44035700301595
s245 = -1.05079633683795e1
s246 =  1.51515796259082e1
s247 = -7.06609886460683
s248 = -1.48043337052968
s249 =  2.10066653978515

# specvol coefficients from gsw_rho_alpha_beta
v000 =  1.0769995862e-3 
v001 = -6.0799143809e-5 
v002 =  9.9856169219e-6 
v003 = -1.1309361437e-6 
v004 =  1.0531153080e-7 
v005 = -1.2647261286e-8 
v006 =  1.9613503930e-9 
v010 = -3.1038981976e-4 
v011 =  2.4262468747e-5 
v012 = -5.8484432984e-7 
v013 =  3.6310188515e-7 
v014 = -1.1147125423e-7 
v020 =  6.6928067038e-4 
v021 = -3.4792460974e-5 
v022 = -4.8122251597e-6 
v023 =  1.6746303780e-8 
v030 = -8.5047933937e-4 
v031 =  3.7470777305e-5 
v032 =  4.9263106998e-6 
v040 =  5.8086069943e-4 
v041 = -1.7322218612e-5 
v042 = -1.7811974727e-6 
v050 = -2.1092370507e-4 
v051 =  3.0927427253e-6 
v060 =  3.1932457305e-5 
v100 = -1.5649734675e-5 
v101 =  1.8505765429e-5 
v102 = -1.1736386731e-6 
v103 = -3.6527006553e-7 
v104 =  3.1454099902e-7 
v110 =  3.5009599764e-5 
v111 = -9.5677088156e-6 
v112 = -5.5699154557e-6 
v113 = -2.7295696237e-7 
v120 = -4.3592678561e-5 
v121 =  1.1100834765e-5 
v122 =  5.4620748834e-6 
v130 =  3.4532461828e-5 
v131 = -9.8447117844e-6 
v132 = -1.3544185627e-6 
v140 = -1.1959409788e-5 
v141 =  2.5909225260e-6 
v150 =  1.3864594581e-6
v200 =  2.7762106484e-5 
v201 = -1.1716606853e-5 
v202 =  2.1305028740e-6 
v203 =  2.8695905159e-7 
v210 = -3.7435842344e-5 
v211 = -2.3678308361e-7 
v212 =  3.9137387080e-7 
v220 =  3.5907822760e-5 
v221 =  2.9283346295e-6 
v222 = -6.5731104067e-7 
v230 = -1.8698584187e-5 
v231 = -4.8826139200e-7 
v240 =  3.8595339244e-6 
v300 = -1.6521159259e-5 
v301 =  7.9279656173e-6 
v302 = -4.6132540037e-7 
v310 =  2.4141479483e-5 
v311 = -3.4558773655e-6 
v312 =  7.7618888092e-9 
v320 = -1.4353633048e-5 
v321 =  3.1655306078e-7 
v330 =  2.2863324556e-6
v400 =  6.9111322702e-6 
v401 = -3.4102187482e-6 
v402 = -6.3352916514e-8 
v410 = -8.7595873154e-6 
v411 =  1.2956717783e-6 
v420 =  4.3703680598e-6 
v500 = -8.0539615540e-7 
v501 =  5.0736766814e-7 
v510 = -3.3052758900e-7 
v600 =  2.0543094268e-7


# In[3]:




# In[4]:

@match_args_return
def alpha(SA,CT,p):
    
    """
    gsw.alpha              thermal expansion coefficient with respect to   
                             Conservative Temperature (76-term equation)  
    ====================================================================  

    USAGE:  
    alpha = gsw.alpha(SA,CT,p)

    DESCRIPTION:
    Calculates the thermal expansion coefficient of seawater with 
    respect to Conservative Temperature using the computationally-
    efficient expression for specific volume in terms of SA, CT and 
    p (Roquet et al., 2014).
  
    Note that this 76-term equation has been fitted in a restricted 
    range of parameter space, and is most accurate inside the 
    "oceanographic funnel" described in McDougall et al. (2010).  
    The GSW library function "gsw.infunnel(SA,CT,p)" is avaialble to be 
    used if one wants to test if some of one's data lies outside this 
    "funnel".  

    INPUT:
    SA  =  Absolute Salinity                                    [ g/kg ]  
    CT  =  Conservative Temperature (ITS-90)                   [ deg C ]
    p   =  sea pressure                                         [ dbar ]
           ( i.e. absolute pressure - 10.1325 dbar )

    SA & CT need to have the same dimensions.
    p may have dimensions 1x1 or Mx1 or 1xN or MxN, where SA & CT are 
    MxN.

    OUTPUT:
    alpha  =  thermal expansion coefficient                      [ 1/K ]  
              with respect to Conservative Temperature
   
    AUTHOR: 
    Paul Barker and Trevor McDougall                [ help@teos-10.org ]

    VERSION NUMBER: 3.05 (27th November, 2015)

    REFERENCES:
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation 
    of seawater - 2010: Calculation and use of thermodynamic properties.  
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 
    56, UNESCO (English), 196 pp.  Available from http://www.TEOS-10.org  
    See Eqn. (2.18.3) of this TEOS-10 manual.

    Roquet, F., G. Madec, T.J. McDougall, P.M. Barker, 2014: Accurate
    polynomial expressions for the density and specifc volume of seawater
    using the TEOS-10 standard. Ocean Modelling.
    """    
    # This line ensures that SA is non-negative.
    SA = np.maximum(SA, 0)

    #deltaS = 24
    sfac = 0.0248826675584615            # sfac = 1/(40*(35.16504/35)).
    offset = 5.971840214030754e-1               # offset = deltaS*sfac.

    x2 = sfac*SA
    xs = np.sqrt(x2 + offset)
    ys = CT*0.025
    z = p*1e-4
    
    v_CT_part = ( a000 + xs*(a100 + xs*(a200 + xs*(a300 + xs*(a400 
        + a500*xs)))) + ys*(a010 + xs*(a110 + xs*(a210 + xs*(a310 
        + a410*xs))) + ys*(a020 + xs*(a120 + xs*(a220 + a320*xs)) 
        + ys*(a030 + xs*(a130 + a230*xs) + ys*(a040 + a140*xs 
        + a050*ys )))) + z*(a001 + xs*(a101 + xs*(a201 + xs*(a301 
        + a401*xs))) + ys*(a011 + xs*(a111 + xs*(a211 + a311*xs)) 
        + ys*(a021 + xs*(a121 + a221*xs) + ys*(a031 + a131*xs 
        + a041*ys))) + z*(a002 + xs*(a102 + xs*(a202 + a302*xs)) 
        + ys*(a012 + xs*(a112 + a212*xs) + ys*(a022 + a122*xs 
        + a032*ys)) + z*(a003 + a103*xs + a013*ys + a004*z))) )

    v = ( v000 + xs*(v010 + xs*(v020 + xs*(v030 + xs*(v040 + xs*(v050 
        + v060*xs))))) + ys*(v100 + xs*(v110 + xs*(v120 + xs*(v130 
        + xs*(v140 + v150*xs)))) + ys*(v200 + xs*(v210 + xs*(v220 
        + xs*(v230 + v240*xs))) + ys*(v300 + xs*(v310 + xs*(v320 
        + v330*xs)) + ys*(v400 + xs*(v410 + v420*xs) + ys*(v500 
        + v510*xs + v600*ys))))) + z*(v001 + xs*(v011 + xs*(v021 
        + xs*(v031 + xs*(v041 + v051*xs)))) + ys*(v101 + xs*(v111 
        + xs*(v121 + xs*(v131 + v141*xs))) + ys*(v201 + xs*(v211 
        + xs*(v221 + v231*xs)) + ys*(v301 + xs*(v311 + v321*xs) 
        + ys*(v401 + v411*xs + v501*ys)))) + z*(v002 + xs*(v012 
        + xs*(v022 + xs*(v032 + v042*xs))) + ys*(v102 + xs*(v112 
        + xs*(v122 + v132*xs)) + ys*(v202 + xs*(v212 + v222*xs)
        + ys*(v302 + v312*xs + v402*ys))) + z*(v003 + xs*(v013 
        + v023*xs) + ys*(v103 + v113*xs + v203*ys) + z*(v004 
        + v014*xs + v104*ys + z*(v005 + v006*z))))) )

    alpha_return = 0.025*v_CT_part/v
    
    return(alpha_return)


# In[4]:




# In[5]:

@match_args_return
def alpha_on_beta(SA,CT,p):

    """
    alpha_on_beta                          alpha/beta (76-term equation)  
    ====================================================================
                                                                           
    USAGE:  
    alpha_on_beta = gsw.alpha_on_beta(SA,CT,p)

    DESCRIPTION:
    Calculates alpha divided by beta, where alpha is the thermal 
    expansion coefficient and beta is the saline contraction coefficient
    of seawater from Absolute Salinity and Conservative Temperature.  
    This function uses the computationally-efficient expression for
    specific volume in terms of SA, CT and p (Roquet et al., 2014).

    Note that the 76-term equation has been fitted in a restricted range
    of parameter space, and is most accurate inside the "oceanographic
    funnel" described in IOC et al. (2010).  The GSW library function 
    "gsw.infunnel(SA,CT,p)" is avaialble to be used if one wants to test
    if some of one's data lies outside this "funnel".  

    INPUT:
    SA  =  Absolute Salinity                                    [ g/kg ]    
    CT  =  Conservative Temperature (ITS-90)                   [ deg C ]
    p   =  sea pressure                                         [ dbar ]
           ( i.e. absolute pressure - 10.1325 dbar )

    SA & CT need to have the same dimensions.
    p may have dimensions 1x1 or Mx1 or 1xN or MxN, where SA & CT are
    MxN.

    OUTPUT:
    alpha_on_beta  =  thermal expansion coefficient with respect to                          
                      Conservative Temperature divided by the saline 
                      contraction coefficient at constant Conservative
                      Temperature                       [ kg g^-1 K^-1 ]  

    AUTHOR: 
    Paul Barker and Trevor McDougall                [ help@teos-10.org ]

    VERSION NUMBER: 3.04 (10th December, 2013)

    REFERENCES:
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation
    of seawater - 2010: Calculation and use of thermodynamic properties.  
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 
    56, UNESCO (English), 196 pp.  Available from http://www.TEOS-10.org  
    See appendix A.20 and appendix K of this TEOS-10 Manual. 

    Roquet, F., G. Madec, T.J. McDougall, P.M. Barker, 2014: Accurate
    polynomial expressions for the density and specifc volume of 
    seawater using the TEOS-10 standard. Ocean Modelling.

    The software is available from http://www.TEOS-10.org
    """
    # This line ensures that SA is non-negative.
    SA = np.maximum(SA, 0)

    # deltaS = 24;
    sfac = 0.0248826675584615             # sfac = 1/(40*(35.16504/35)).
    offset = 5.971840214030754e-1                # offset = deltaS*sfac.   

    x2 = sfac*SA
    xs = np.sqrt(x2 + offset) 
    ys = CT*0.025
    z = p*1e-4

    v_CT_part = ( a000 + xs*(a100 + xs*(a200 + xs*(a300 + xs*(a400 
        + a500*xs)))) + ys*(a010 + xs*(a110 + xs*(a210 + xs*(a310 
        + a410*xs))) + ys*(a020 + xs*(a120 + xs*(a220 + a320*xs)) 
        + ys*(a030 + xs*(a130 + a230*xs) + ys*(a040 + a140*xs 
        + a050*ys )))) + z*(a001 + xs*(a101 + xs*(a201 + xs*(a301 
        + a401*xs))) + ys*(a011 + xs*(a111 + xs*(a211 + a311*xs)) 
        + ys*(a021 + xs*(a121 + a221*xs) + ys*(a031 + a131*xs 
        + a041*ys))) + z*(a002 + xs*(a102 + xs*(a202 + a302*xs)) 
        + ys*(a012 + xs*(a112 + a212*xs) + ys*(a022 + a122*xs 
        + a032*ys)) + z*(a003 + a103*xs + a013*ys + a004*z))) )

    v_SA_part = ( b000 + xs*(b100 + xs*(b200 + xs*(b300 + xs*(b400 
        + b500*xs)))) + ys*(b010 + xs*(b110 + xs*(b210 + xs*(b310 
        + b410*xs))) + ys*(b020 + xs*(b120 + xs*(b220 + b320*xs)) 
        + ys*(b030 + xs*(b130 + b230*xs) + ys*(b040 + b140*xs 
        + b050*ys)))) + z*(b001 + xs*(b101 + xs*(b201 + xs*(b301 
        + b401*xs))) + ys*(b011 + xs*(b111 + xs*(b211 + b311*xs)) 
        + ys*(b021 + xs*(b121 + b221*xs) + ys*(b031 + b131*xs 
        + b041*ys))) + z*(b002 + xs*(b102 + xs*(b202 + b302*xs)) 
        + ys*(b012 + xs*(b112 + b212*xs) + ys*(b022 + b122*xs 
        + b032*ys)) + z*(b003 +  b103*xs + b013*ys + b004*z))) )
 
    alpha_on_beta_return = -(v_CT_part*xs)/(20*sfac*v_SA_part)
         
    return(alpha_on_beta_return)


# In[5]:




# In[6]:

@match_args_return
def beta(SA,CT,p):
   
    """
    gsw.beta                   saline contraction coefficient at constant  
                              Conservative Temperature (76-term equation)  
    ======================================================================

    USAGE:  
    beta = gsw.beta(SA,CT,p)

    DESCRIPTION:
    Calculates the saline (i.e. haline) contraction coefficient of 
    seawater at constant Conservative Temperature using the 
    computationally-efficient 76-term expression for specific volume in
    terms of SA, CT and p (Roquet et al., 2014).
  
    Note that the 76-term equation has been fitted in a restricted range
    of parameter space, and is most accurate inside the "oceanographic 
    funnel" described in McDougall et al. (2010).  The GSW library 
    function "gsw.infunnel(SA,CT,p)" is avaialble to be used if one 
    wants to test if some of one's data lies outside this "funnel".  
    
    INPUT:
    SA  =  Absolute Salinity                                    [ g/kg ]
    CT  =  Conservative Temperature (ITS-90)                   [ deg C ]
    p   =  sea pressure                                         [ dbar ]
        ( i.e. absolute pressure - 10.1325 dbar )

    SA & CT need to have the same dimensions.
    p may have dimensions 1x1 or Mx1 or 1xN or MxN, where SA & CT are 
    MxN.

    OUTPUT:
    beta  =  saline contraction coefficient                     [ kg/g ]
          at constant Conservative Temperature
   
    AUTHOR: 
    Paul Barker and Trevor McDougall                [ help@teos-10.org ]

    VERSION NUMBER: 3.05 (27th November, 2015)

    REFERENCES:
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation
    of seawater - 2010: Calculation and use of thermodynamic properties.    
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 
    56, UNESCO (English), 196 pp.  Available from http://www.TEOS-10.org  
    See Eqn. (2.19.3) of this TEOS-10 manual.

    Roquet, F., G. Madec, T.J. McDougall, P.M. Barker, 2014: Accurate
    polynomial expressions for the density and specifc volume of 
    seawater using the TEOS-10 standard. Ocean Modelling.
    """
    # This line ensures that SA is non-negative.
    SA = np.maximum(SA, 0)

    #deltaS = 24;
    sfac = 0.0248826675584615             # sfac = 1/(40*(35.16504/35)).  
    offset = 5.971840214030754e-1                # offset = deltaS*sfac.

    x2 = sfac*SA
    xs = np.sqrt(x2 + offset)
    ys = CT*0.025
    z = p*1e-4
    
    v_SA_part = ( b000 + xs*(b100 + xs*(b200 + xs*(b300 + xs*(b400 
        + b500*xs)))) + ys*(b010 + xs*(b110 + xs*(b210 + xs*(b310 
        + b410*xs))) + ys*(b020 + xs*(b120 + xs*(b220 + b320*xs)) 
        + ys*(b030 + xs*(b130 + b230*xs) + ys*(b040 + b140*xs 
        + b050*ys)))) + z*(b001 + xs*(b101 + xs*(b201 + xs*(b301 
        + b401*xs))) + ys*(b011 + xs*(b111 + xs*(b211 + b311*xs)) 
        + ys*(b021 + xs*(b121 + b221*xs) + ys*(b031 + b131*xs 
        + b041*ys))) + z*(b002 + xs*(b102 + xs*(b202 + b302*xs)) 
        + ys*(b012 + xs*(b112 + b212*xs) + ys*(b022 + b122*xs 
        + b032*ys)) + z*(b003 +  b103*xs + b013*ys + b004*z))) )
 
    v = ( v000 + xs*(v010 + xs*(v020 + xs*(v030 + xs*(v040 + xs*(v050 
        + v060*xs))))) + ys*(v100 + xs*(v110 + xs*(v120 + xs*(v130 
        + xs*(v140 + v150*xs)))) + ys*(v200 + xs*(v210 + xs*(v220 
        + xs*(v230 + v240*xs)))  + ys*(v300 + xs*(v310 + xs*(v320 
        + v330*xs)) + ys*(v400 + xs*(v410 + v420*xs) + ys*(v500 
        + v510*xs + v600*ys))))) + z*(v001 + xs*(v011 + xs*(v021 
        + xs*(v031 + xs*(v041 + v051*xs)))) + ys*(v101 + xs*(v111 
        + xs*(v121 + xs*(v131 + v141*xs))) + ys*(v201 + xs*(v211 
        + xs*(v221 + v231*xs)) + ys*(v301 + xs*(v311 + v321*xs) 
        + ys*(v401 + v411*xs + v501*ys)))) + z*(v002 + xs*(v012 
        + xs*(v022 + xs*(v032 + v042*xs))) + ys*(v102 + xs*(v112 
        + xs*(v122 + v132*xs)) + ys*(v202 + xs*(v212 + v222*xs) 
        + ys*(v302 + v312*xs + v402*ys))) + z*(v003 + xs*(v013 
        + v023*xs) + ys*(v103 + v113*xs + v203*ys) + z*(v004 
        + v014*xs + v104*ys + z*(v005 + v006*z))))) )

    beta_return = -v_SA_part*0.5*sfac/(v*xs)
    
    return(beta_return)


# In[6]:




# In[7]:

@match_args_return
def cabbeling(SA,CT,p):

    """
    gsw.cabbeling                                  cabbeling coefficient   
                                                      (76-term equation)
    ====================================================================

    USAGE:  
    cabbeling = gsw.cabbeling(SA,CT,p)

    DESCRIPTION:
    Calculates the cabbeling coefficient of seawater with respect to  
    Conservative Temperature.  This function uses the computationally-
    efficient expression for specific volume in terms of SA, CT and p
    (Roquet et al., 2014).
  
    Note that the 76-term equation has been fitted in a restricted range
    of parameter space, and is most accurate inside the "oceanographic 
    funnel" described in IOC et al. (2010).  The GSW library function 
    "gsw.infunnel(SA,CT,p)" is avaialble to be used if one wants to test 
    if some of one's data lies outside this "funnel".  

    INPUT:
    SA  =  Absolute Salinity                                    [ g/kg ]
    CT  =  Conservative Temperature (ITS-90)                   [ deg C ]
    p   =  sea pressure                                         [ dbar ]
           ( i.e. absolute pressure - 10.1325 dbar )

    SA & CT need to have the same dimensions.
    p may have dimensions 1x1 or Mx1 or 1xN or MxN, where SA & CT are MxN.

    OUTPUT:
    cabbeling  =  cabbeling coefficient with respect to        [ 1/K^2 ]
                  Conservative Temperature.                    

    AUTHOR: 
    Trevor McDougall and Paul Barker                [ help@teos-10.org ]   

    VERSION NUMBER: 3.04 (10th December, 2013)

    REFERENCES:
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation 
    of seawater - 2010: Calculation and use of thermodynamic properties.  
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 
    56,UNESCO (English), 196 pp.  Available from http://www.TEOS-10.org
    See Eqns. (3.9.2) and (P.4) of this TEOS-10 manual.

    Roquet, F., G. Madec, T.J. McDougall, P.M. Barker, 2014: Accurate
    polynomial expressions for the density and specifc volume of seawater
    using the TEOS-10 standard. Ocean Modelling.
    """
    # This line ensures that SA is non-negative.
    SA = np.maximum(SA, 0)

    [v_SA, v_CT, dummy] = specvol_first_derivatives(SA,CT,p)

    [v_SA_SA, v_SA_CT, v_CT_CT, dummy, dummy] = specvol_second_derivatives(SA,CT,p)
    
    rho_temp = rho(SA,CT,p)

    alpha_CT = rho_temp*(v_CT_CT - rho_temp*v_CT**2)

    alpha_SA = rho_temp*(v_SA_CT - rho_temp*v_SA*v_CT)

    beta_SA = -rho_temp*(v_SA_SA - rho_temp*v_SA**2)

    alpha_on_beta_temp = alpha_on_beta(SA,CT,p)

    cabbeling_return = ( alpha_CT + alpha_on_beta_temp*(2*alpha_SA 
                        - alpha_on_beta_temp*beta_SA) )

    return(cabbeling_return)


# In[7]:




# In[8]:

@match_args_return
def CT_freezing_poly(SA,p,saturation_fraction = 1):

    """
    gsw.CT_freezing_poly               Conservative Temperature at which  
                                                 seawater freezes (poly)
    ====================================================================

    USAGE:
    CT_freezing = gsw.CT_freezing_poly(SA,p,saturation_fraction)

    DESCRIPTION:
    Calculates the Conservative Temperature at which seawater freezes.
    The error of this fit ranges between -5e-4 K and 6e-4 K when 
    compared with the Conservative Temperature calculated from the exact
    in-situ freezing temperature which is found by a Newton-Raphson 
    iteration of the equality of the chemical potentials of water in 
    seawater and in ice. Note that the Conservative Temperature freezing
    temperature can be found by this exact method using the function 
    gsw.CT_freezing.

    INPUT:
    SA  =  Absolute Salinity                                    [ g/kg ]
    p   =  sea pressure                                         [ dbar ]
           ( i.e. absolute pressure - 10.1325 dbar ) 

    OPTIONAL:
    saturation_fraction = the saturation fraction of dissolved air in 
                          seawater
                          (i.e., saturation_fraction must be between 0 
                          and 1, and the default is 1, completely 
                          saturated) 

    p & saturation_fraction (if provided) may have dimensions 1x1 or Mx1
    or 1xN or MxN, where SA is MxN.

    OUTPUT:
    CT_freezing = Conservative Temperature at freezing of seawater 
                                                               [ deg C ] 
                  That is, the freezing temperature expressed in
                  terms of Conservative Temperature (ITS-90).                

    AUTHOR: 
    Trevor McDougall, Paul Barker and Rainer Feistal[ help@teos-10.org ]

    VERSION NUMBER: 3.04 (10th December, 2013)

    REFERENCES:
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation 
    of seawater - 2010: Calculation and use of thermodynamic properties.  
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 
    56,UNESCO (English), 196 pp.  Available from http://www.TEOS-10.org.
    See sections 3.33 and 3.34 of this TEOS-10 Manual.      
    """    
    # These few lines ensure that SA is non-negative.
    if (SA < 0).any():
        raise ValueError('SA must be non-negative!')  # This doesn't seem to work?

    SA_r = SA*1e-2
    x = np.sqrt(SA_r)
    p_r = p*1e-4

    CT_freezing_return = ( c0 + SA_r*(c1 + x*(c2 + x*(c3 + x*(c4 
                        + x*(c5 + c6*x))))) + p_r*(c7 + p_r*(c8 
                        + c9*p_r)) + SA_r*p_r*(c10 + p_r*(c12 
                        + p_r*(c15 + c21*SA_r)) + SA_r*(c13 + c17*p_r 
                        + c19*SA_r) + x*(c11 + p_r*(c14 + c18*p_r) 
                        + SA_r*(c16 + c20*p_r + c22*SA_r))) )

    CT_freezing_return = np.asanyarray(CT_freezing_return)
    
    # Adjust for the effects of dissolved air 
    a = 0.014289763856964       # Note that a = 0.502500117621/35.16504.
    b = 0.057000649899720
    CT_freezing_return = ( CT_freezing_return - 
                          saturation_fraction*(1e-3)*(2.4 
                         - a*SA)*(1 + b*(1 - SA/35.16504)) )

    # set any values that are out of range to be nan. 
    temp1 = np.logical_or(p > 10000, SA > 120)
    temp2 = np.logical_or(temp1, p + SA*71.428571428571402 > 13571.42857142857)
    
    CT_freezing_return = np.asanyarray(CT_freezing_return)              
    CT_freezing_return[temp2] = np.nan                 
    
    return(CT_freezing_return)


# In[8]:




# In[9]:

@match_args_return
def CT_from_enthalpy(SA,h,p):

    """
    gsw.CT_from_enthalpy     Conservative Temperature from specific
                            enthalpy of seawater (76-term equation)  
    ===============================================================

    USAGE:
    CT = gsw.CT_from_enthalpy(SA,h,p)

    DESCRIPTION:
    Calculates the Conservative Temperature of seawater, given the 
    Absolute Salinity, specific enthalpy, h, and pressure p.  The 
    specific enthalpy input is the one calculated from the 
    computationally-efficient expression for specific volume in 
    terms of SA, CT and p (Roquet et al.,2014).

    Note that the 76-term equation has been fitted in a restricted 
    range of parameter space, and is most accurate inside the 
    "oceanographic funnel" described in IOC et al. (2010).  The GSW
    library function "gsw.infunnel(SA,CT,p)" is avaialble to be 
    used if one wants to test if some of one's data lies outside 
    this "funnel".  

    INPUT:
    SA  =  Absolute Salinity                               [ g/kg ]  
    h   =  specific enthalpy                               [ J/kg ]
    p   =  sea pressure                                    [ dbar ]
           ( i.e. absolute pressure - 10.1325 dbar ) 

    SA & h need to have the same dimensions.
    p may have dimensions 1x1 or Mx1 or 1xN or MxN, where SA & h 
    are MxN.

    OUTPUT:
    CT  =  Conservative Temperature ( ITS-90)             [ deg C ]

    AUTHOR: 
    Trevor McDougall and Paul Barker.          [ help@teos-10.org ]  

    VERSION NUMBER: 3.04 (10th December, 2013)

    REFERENCES:
    IOC, SCOR and IAPSO, 2010: The international thermodynamic 
    equation of seawater - 2010: Calculation and use of 
    thermodynamic properties. Intergovernmental Oceanographic 
    Commission, Manuals and Guides No. 56,UNESCO (English), 196 pp. 
    Available from http://www.TEOS-10.org

    McDougall, T.J., 2003: Potential enthalpy: A conservative 
    oceanic variable for evaluating heat content and heat fluxes.
    Journal of Physical Oceanography, 33, 945-963.  

    McDougall, T.J., and S.J. Wotherspoon, 2014: A simple 
    modification of Newton's method to achieve convergence of order
    1 + sqrt(2).  Applied Mathematics Letters, 29, 20-25.  

    Roquet, F., G. Madec, T.J. McDougall, P.M. Barker, 2014: 
    Accurate polynomial expressions for the density and specifc 
    volume of seawater using the TEOS-10 standard. Ocean Modelling.
    """
    # This line ensures that SA is non-negative
    SA = np.maximum(SA, 0)  
    
    CT_freezing_var = CT_freezing_poly(SA,p,0) # This is the CT freezing 
    # temperature
    h_below_freeze = 798.37 # 798.37 = gsw_cp0*0.2, This allows for 
    # water to be 0.2C below the freezing temperature
    h_freezing = enthalpy(SA,CT_freezing_var,p)
    
    Icold = (h < (h_freezing - h_below_freeze))
    if Icold.any():
        h[Icold] = np.nan

    h_40 = enthalpy(SA,40*np.ones_like(SA),p)
    Ihot = h > h_40
    if Ihot.any():
        h[Ihot] = np.nan
        
    CT = ( CT_freezing_var + (40 - CT_freezing_var)*(h - h_freezing)/(h_40 
        - h_freezing) ) # First guess of CT
    [dummy, h_CT] = enthalpy_first_derivatives(SA,CT,p)

    # Begin the modified Newton-Raphson iterative procedure 
    CT_old = CT
    f = enthalpy(SA,CT_old,p) - h
    CT = CT_old - f/h_CT  # this is half way through the modified 
    # Newton's method (McDougall and Wotherspoon, 2014)
    CT_mean = 0.5*(CT + CT_old)
    [dummy, h_CT] = enthalpy_first_derivatives(SA,CT_mean,p)
    CT = CT_old - f/h_CT  # this is the end of one full iteration of the
    # modified Newton's method

    CT_old = CT
    f = enthalpy(SA,CT_old,p) - h
    CT = CT_old - f/h_CT # this is half way through the modified 
    # Newton's method (McDougall and Wotherspoon, 2013)

    # After 1.5 iterations of this modified Newton-Raphson iteration,
    # the error in CT is no larger than 2.5x10^-14 degrees C, which 
    # is machine precision for this calculation. 

    return(CT)


# In[9]:




# In[10]:

@match_args_return
def CT_from_rho(rho_input,SA,p, second_out = False):

    """
    gsw.CT_from_rho                Conservative Temperature from density  
                                                      (76-term equation)
    ====================================================================

    USAGE:
    [CT,CT_multiple] = gsw.CT_from_rho(rho,SA,p,True)
    OR
    [CT] = gsw.CT_from_rho(rho,SA,p)

    DESCRIPTION:
    Calculates the Conservative Temperature of a seawater sample, for
    given values of its density, Absolute Salinity and sea pressure 
    (in dbar), using the computationally-efficient expression for 
    specific volume in terms of SA, CT and p (Roquet et al., 2014).

    Note that the 76-term equation has been fitted in a restricted range
    of parameter space, and is most accurate inside the "oceanographic 
    funnel" described in IOC et al. (2010).  The GSW library function 
    "gsw.infunnel(SA,CT,p)" is avaialble to be used if one wants to test
    if some of one's data lies outside this "funnel".  

    INPUT:
    rho  =  density of a seawater sample (e.g. 1026 kg/m^3)   [ kg/m^3 ]
            Note. This input has not had 1000 kg/m^3 subtracted from it.
            That is, it is 'density', not 'density anomaly'.
    SA   =  Absolute Salinity                                   [ g/kg ]
    p    =  sea pressure                                        [ dbar ]
            ( i.e. absolute pressure - 10.1325 dbar )

    rho & SA need to have the same dimensions.
    p may have dimensions 1x1 or Mx1 or 1xN or MxN, where rho & SA are 
    MxN.

    OUTPUT:
    CT  =  Conservative Temperature  (ITS-90)                  [ deg C ]
    CT_multiple  =  Conservative Temperature  (ITS-90)         [ deg C ]
                    Note that at low salinities, in brackish water, 
                    there are two possible Conservative Temperatures for
                    a single density.  This programme will output both 
                    valid solutions.  To see this second solution the 
                    user must call the programme with the optional 4th 
                    parameter as "True", if there is only one possible 
                    solution and the programme has been run with the 
                    intention of returning two outputs the second 
                    variable will be set to NaN.

    AUTHOR:
    Trevor McDougall & Paul Barker                  [ help@teos-10.org ]

    VERSION NUMBER: 3.04 (10th December, 2013)

    REFERENCES:
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation
    of seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 
    56, UNESCO (English), 196 pp.  Available from http://www.TEOS-10.org  

    Roquet, F., G. Madec, T.J. McDougall, P.M. Barker, 2014: Accurate
    polynomial expressions for the density and specifc volume of 
    seawater using the TEOS-10 standard. Ocean Modelling.
    """
    # dummy variables to see if CT_a and CT_b have been used at the end
    # of the code.
    CT_a_exists = False
    CT_b_exists = False
    
    # alpha_limit is the positive value of the thermal expansion 
    # coefficient which is used at the freezing temperature to 
    # distinguish between I_salty and I_fresh.
    alpha_limit = 1e-5

    # rec_half_rho_TT is a constant representing the reciprocal of half 
    # the second derivative of density with respect to temperature near
    # the temperature of maximum density.
    rec_half_rho_TT = -110.0

    CT = np.nan*np.ones_like(SA)
    CT_multiple = np.nan*np.ones_like(SA)

    # SA out of range, set to NaN.
    temp1 = np.logical_or(SA < 0, SA > 42)
    temp2 = np.logical_or(p < -1.5, p > 12000)
    temp3 = np.logical_or(temp1, temp2)
    SA[temp3] = np.nan                
    
    rho_40 = rho(SA,40*np.ones_like(SA),p)
    # rho too light, set to NaN.
    SA[(rho_input - rho_40) < 0] = np.nan
            
    CT_max_rho = CT_maxdensity(SA,p)
    rho_max = rho(SA,CT_max_rho,p)
    rho_extreme = rho_max.copy()
    # name CT_freezing as CT_freezing_var so it doesn't overlap with
    # CT_freezing definition.
    CT_freezing_var = fr.CT_freezing(SA,p) # this assumes that the 
    # seawater is always saturated with air
    rho_freezing = rho(SA,CT_freezing_var,p)
    # reset the extreme values
    rho_extreme[(CT_freezing_var - CT_max_rho) > 0] = rho_freezing[(CT_freezing_var - CT_max_rho) > 0]

    # set SA values to NaN for the rho's that are too dense.
    SA[rho_input > rho_extreme] = np.nan 
    
    SA[np.isnan(SA + p + rho_input)] = np.nan
     
    alpha_freezing = alpha(SA,CT_freezing_var,p)
    
    I_salty = alpha_freezing > alpha_limit 
    if I_salty.any():
        CT_diff = 40*np.ones_like(I_salty) - CT_freezing_var[I_salty]

        top = ( rho_40[I_salty] - rho_freezing[I_salty] 
            + rho_freezing[I_salty]*alpha_freezing[I_salty]*CT_diff )
             
        a = top/(CT_diff**2)
        b = - rho_freezing[I_salty]*alpha_freezing[I_salty]
        c = rho_freezing[I_salty] - rho_input[I_salty]
        sqrt_disc = np.sqrt(b**2 - 4*a*c)
        # the value of t(I_salty) here is the initial guess at CT in the 
        # range of I_salty.
        CT[I_salty] = CT_freezing_var[I_salty] + 0.5*(-b - sqrt_disc)/a
           
    I_fresh = alpha_freezing <= alpha_limit  
    if I_fresh.any(): 
        CT_diff = 40*np.ones_like(I_fresh) - CT_max_rho[I_fresh]
        factor = ( (rho_max[I_fresh] 
                - rho_input[I_fresh])/(rho_max[I_fresh] 
                - rho_40[I_fresh]) )
        delta_CT = CT_diff*np.sqrt(factor)
        
        I_fresh_NR = delta_CT > 5
        if I_fresh_NR.any():
            CT[I_fresh[I_fresh_NR]] = ( CT_max_rho[I_fresh[I_fresh_NR]] 
                                        + delta_CT[I_fresh_NR] )
           
        I_quad = delta_CT <= 5   
        if I_quad.any():
            CT_a_exists = True # CT_a is created here, although it only 
            # contains NaNs at this stage
            CT_a = np.nan*np.ones_like(SA) 
            # set the initial value of the quadratic solution routes.
            CT_a[I_fresh[I_quad]] = ( CT_max_rho[I_fresh[I_quad]] 
                + np.sqrt(rec_half_rho_TT*(rho_input[I_fresh[I_quad]] 
                - rho_max[I_fresh[I_quad]])) )      
            Number_of_iterations = 7
            for j in range(0,Number_of_iterations):
                CT_old = CT_a.copy()
                rho_old = rho(SA,CT_old,p)
                factorqa = (rho_max - rho_input)/(rho_max - rho_old)
                CT_a = CT_max_rho + (CT_old - CT_max_rho)*np.sqrt(factorqa)
            
            CT_a[CT_freezing_var - CT_a < 0] = np.nan
            
            CT_b = np.nan*np.ones_like(SA)
            CT_b_exists = True # this is where CT_b is created, 
            # although it only contains NaNS at this stage.
            
            # set the initial value of the quadratic solution roots.  
            CT_b[I_fresh[I_quad]] = ( CT_max_rho[I_fresh[I_quad]]
                - np.sqrt(rec_half_rho_TT*(rho_input[I_fresh[I_quad]] 
                - rho_max[I_fresh[I_quad]])) )    
             
            for j in range(0,Number_of_iterations):
                CT_old = CT_b.copy()
                rho_old = rho(SA,CT_old,p)
                factorqb = (rho_max - rho_input)/(rho_max - rho_old)
                CT_b = CT_max_rho + (CT_old - CT_max_rho)*np.sqrt(factorqb)
            
             # After seven iterations of this quadratic iterative 
             # procedure, the error in rho is no larger than 4.6x10^-13 
             # kg/m^3.
            CT_b[CT_freezing_var - CT_b < 0] = np.nan
           
    # begin the modified Newton-Raphson iterative method, which will only
    # operate on non-NaN CT data.

    v_lab = np.ones_like(rho_input)/rho_input
    v_CT = specvol(SA,CT,p)*alpha(SA,CT,p)

    Number_of_iterations = 3
    for k in range(0,Number_of_iterations):
        CT_old = CT.copy()
        delta_v = specvol(SA,CT_old,p) - v_lab
        CT = CT_old - delta_v/v_CT  # this is half way through the 
        # modified N-R method
        CT_mean = 0.5*(CT + CT_old)
        v_CT = specvol(SA,CT_mean,p)*alpha(SA,CT_mean,p)
        CT = CT_old - delta_v/v_CT 
    
    # After three iterations of this modified Newton-Raphson iteration,
    # the error in rho is no larger than 1.6x10^-12 kg/m^3.

    if CT_a_exists == True:
        # redefine CT
        temp_1a = np.isnan(CT_a)
        temp_2a = np.where(temp_1a == False)
        CT[temp_2a] = CT_a[temp_2a]
    
    if CT_b_exists == True:
        # redefine CT_multiple
        temp_1b = np.isnan(CT_b)
        temp_2b = np.where(temp_1b == False)
        CT_multiple[temp_2b] = CT_b[temp_2b]
            
    if second_out == False: # if user only wants one output. 
        # NOTE: this is the default
        return(CT)
    
    elif second_out == True: # if user assigns the optional 4th 
        # parameter in the definition call to "True"
        return(CT,CT_multiple)


# In[10]:




# In[11]:

@match_args_return
def CT_maxdensity(SA,p):

    """
    gsw.CT_maxdensity                Conservative Temperature of maximum 
                                  density of seawater (76-term equation)
    ====================================================================

    USAGE:
    CT_maxdensity = gsw.CT_maxdensity(SA,p)

    DESCRIPTION:
    Calculates the Conservative Temperature of maximum density of 
    seawater. This function returns the Conservative temperature at 
    which the density of seawater is a maximum, at given Absolute 
    Salinity, SA, and sea pressure, p (in dbar).  This function uses the
    computationally-efficient expression for specific volume in terms of  
    SA, CT and p (Roquet et al., 2014).

    Note that the 76-term equation has been fitted in a restricted range
    of parameter space, and is most accurate inside the "oceanographic 
    funnel" described in IOC et al. (2010).  The GSW library function 
    "gsw.infunnel(SA,CT,p)" is avaialble to be used if one wants to test 
    if some of one's data lies outside this "funnel".  

    INPUT:
    SA =  Absolute Salinity                                     [ g/kg ]
    p  =  sea pressure                                          [ dbar ]
          ( i.e. absolute pressure - 10.1325 dbar ) 

    p may have dimensions 1x1 or Mx1 or 1xN or MxN, where SA is MxN.

    OUTPUT:
    CT_maxdensity  =  Conservative Temperature at which         [ deg C ]
                      the density of seawater is a maximum for
                      given Absolute Salinity and pressure.

    AUTHOR: 
    Trevor McDougall & Paul Barker                  [ help@teos-10.org ]

    VERSION NUMBER: 3.04 (10th December, 2013)

    REFERENCES:
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation 
    of seawater - 2010: Calculation and use of thermodynamic properties.  
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 
    56, UNESCO (English), 196 pp.  Available from http://www.TEOS-10.org  
    See section 3.42 of this TEOS-10 Manual.  

    McDougall, T.J., and S.J. Wotherspoon, 2012: A simple modification 
    of Newtons method to achieve convergence of order "1 + sqrt(2)".
    Submitted to Applied Mathematics and Computation. 

    Roquet, F., G. Madec, T.J. McDougall, P.M. Barker, 2014: Accurate
    polynomial expressions for the density and specifc volume of 
    seawater using the TEOS-10 standard. Ocean Modelling.
    """
    dCT = 0.001                # the Conservative Temperature increment.

    CT = 3.978 - 0.22072*SA                   # the initial guess of CT.  

    dalpha_dCT = 1.1e-5            # the initial guess for d(alpha)_dCT.

    Number_of_iterations = 3
    for j in range(0, Number_of_iterations):
        CT_old = CT
        alpha_temp = alpha(SA,CT_old,p)
        CT = CT_old - alpha_temp/dalpha_dCT # this is half way through the 
        # modified method
        CT_mean = 0.5*(CT + CT_old)
        dalpha_dCT = ( (alpha(SA,CT_mean + dCT,p) 
                        - alpha(SA,CT_mean - dCT,p))/(dCT + dCT) )
        CT = CT_old - alpha_temp/dalpha_dCT

    # After three iterations of this modified Newton-Raphson (McDougall
    # and Wotherspoon, 2012) iteration, the error in CT_maxdensity is 
    # typically no larger than 1x10^-15 degress C.  

    CT_maxdensity_return = CT

    return(CT_maxdensity_return)


# In[11]:




# In[12]:

@match_args_return
def dynamic_enthalpy(SA,CT,p):

    """
    gsw.dynamic_enthalpy                    dynamic enthalpy of seawater  
                                                      (76-term equation)
    ====================================================================  

    USAGE:
    dynamic_enthalpy = gsw.dynamic_enthalpy(SA,CT,p)

    DESCRIPTION:
    Calculates dynamic enthalpy of seawater using the computationally-
    efficient expression for specific volume in terms of SA, CT and p
    (Roquet et al., 2014).  Dynamic enthalpy is defined as enthalpy 
    minus potential enthalpy (Young, 2010). 

    Note that the 76-term equation has been fitted in a restricted range 
    of parameter space, and is most accurate inside the "oceanographic 
    funnel" described in IOC et al. (2010).  The GSW library function 
    "gsw.infunnel(SA,CT,p)" is avaialble to be used if one wants to test 
    if some of one's data lies outside this "funnel".  

    INPUT:
    SA  =  Absolute Salinity                                    [ g/kg ]  
    CT  =  Conservative Temperature (ITS-90)                   [ deg C ]
    p   =  sea pressure                                         [ dbar ]
           ( i.e. absolute pressure - 10.1325 dbar )

    SA & CT need to have the same dimensions.
    p may have dimensions 1x1 or Mx1 or 1xN or MxN, where SA & CT are 
    MxN.

    OUTPUT:
    dynamic_enthalpy  =  dynamic enthalpy                       [ J/kg ]

    AUTHOR: 
    Trevor McDougall and Paul Barker                [ help@teos-10.org ]

    VERSION NUMBER: 3.04 (10th December, 2013)

    REFERENCES:
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation 
    of seawater - 2010: Calculation and use of thermodynamic properties.  
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 
    56, UNESCO (English), 196 pp.  Available from http://www.TEOS-10.org
    See section 3.2 of this TEOS-10 Manual. 

    McDougall, T. J., 2003: Potential enthalpy: A conservative oceanic 
    variable for evaluating heat content and heat fluxes. Journal of 
    Physical Oceanography, 33, 945-963.  
    See Eqns. (18) and (22)

    Roquet, F., G. Madec, T.J. McDougall, P.M. Barker, 2014: Accurate
    polynomial expressions for the density and specifc volume of seawater
    using the TEOS-10 standard. Ocean Modelling.

    Young, W.R., 2010: Dynamic enthalpy, Conservative Temperature, and 
    the seawater Boussinesq approximation. Journal of Physical 
    Oceanography, 40, 394-400.
    """    
    # This line ensures that SA is non-negative.
    SA = np.maximum(SA, 0)

    # Set lower temperature limit for water that is much colder than the
    # freezing termperature.
    CT_frozen = - 1.79e-02 - 6.7157e-2*SA - 9.2708e-04*p

    Icold = CT < (CT_frozen - 0.2)
    if Icold.any():
        CT[Icold] = np.nan

    db2Pa = 1e4                      # factor to convert from dbar to Pa  
    cp0 = 3991.86795711963     # from Eqn. (3.3.3) of IOC et al. (2010).

    sfac = 0.0248826675584615             # sfac = 1/(40*(35.16504/35)).
    offset = 5.971840214030754e-1  
    
    x2 = sfac*SA
    xs = np.sqrt(x2 + offset)
    ys = CT*0.025
    z = p*1e-4
    
    dynamic_enthalpy_part = ( z*(h001 + xs*(h101 + xs*(h201 + xs*(h301 
        + xs*(h401 + xs*(h501 + h601*xs))))) + ys*(h011 + xs*(h111 
        + xs*(h211 + xs*(h311 + xs*(h411 + h511*xs)))) + ys*(h021 
        + xs*(h121 + xs*(h221 + xs*(h321 + h421*xs))) + ys*(h031 
        + xs*(h131 + xs*(h231 + h331*xs)) + ys*(h041 + xs*(h141 
        + h241*xs) + ys*(h051 + h151*xs + h061*ys))))) + z*(h002 
        + xs*(h102 + xs*(h202 + xs*(h302 + xs*(h402 + h502*xs)))) 
        + ys*(h012 + xs*(h112 + xs*(h212 + xs*(h312 + h412*xs))) 
        + ys*(h022 + xs*(h122 + xs*(h222 + h322*xs)) + ys*(h032 
        + xs*(h132 + h232*xs) + ys*(h042 + h142*xs + h052*ys)))) 
        + z*(h003 + xs*(h103 + xs*(h203 + xs*(h303 + h403*xs))) 
        + ys*(h013 + xs*(h113 + xs*(h213 + h313*xs)) + ys*(h023 
        + xs*(h123 + h223*xs) + ys*(h033 + h133*xs + h043*ys))) 
        + z*(h004 + xs*(h104 + h204*xs) + ys*(h014 + h114*xs + h024*ys) 
        + z*(h005 + h105*xs + h015*ys + z*(h006 + h007*z)))))) )

    dynamic_enthalpy_return = dynamic_enthalpy_part*1e8 
    # Note. 1e8 = db2Pa*1e4  
    
    return(dynamic_enthalpy_return)


# In[12]:




# In[13]:

@match_args_return
def enthalpy(SA,CT,p):

    """
    gsw.enthalpy                           specific enthalpy of seawater
                                                      (76-term equation)
    ====================================================================

    USAGE:
    enthalpy = gsw.enthalpy(SA,CT,p)

    DESCRIPTION:
    Calculates specific enthalpy of seawater using the computationally-
    efficient expression for specific volume in terms of SA, CT and p
    (Roquet et al., 2014).

    Note that the 76-term equation has been fitted in a restricted range
    of parameter space, and is most accurate inside the "oceanographic 
    funnel" described in IOC et al. (2010).  The GSW library function 
    "gsw.infunnel(SA,CT,p)" is avaialble to be used if one wants to test
    if some of one's data lies outside this "funnel".  

    INPUT:
    SA  =  Absolute Salinity                                    [ g/kg ]
    CT  =  Conservative Temperature (ITS-90)                   [ deg C ]
    p   =  sea pressure                                         [ dbar ]
           ( i.e. absolute pressure - 10.1325 dbar ) 

    SA & CT need to have the same dimensions.
    p may have dimensions 1x1 or Mx1 or 1xN or MxN, where SA & CT are 
    MxN.

    OUTPUT:
    enthalpy  =  specific enthalpy                              [ J/kg ]

    AUTHOR: 
    Trevor McDougall, David Jackett, Claire Roberts-Thomson and Paul 
    Barker.    
                                                    [ help@teos-10.org ]

    VERSION NUMBER: 3.04 (10th December, 2013)

    REFERENCES:
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation 
    of seawater - 2010: Calculation and use of thermodynamic properties.  
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 
    56, UNESCO (English), 196 pp.  Available from http://www.TEOS-10.org
    See Eqn. (A.30.6) of this TEOS-10 Manual. 

    McDougall, T.J., 2003: Potential enthalpy: A conservative oceanic 
    variable for evaluating heat content and heat fluxes. Journal of 
    Physical Oceanography, 33, 945-963.  
    See Eqns. (18) and (22)

    Roquet, F., G. Madec, T.J. McDougall, P.M. Barker, 2014: Accurate
    polynomial expressions for the density and specifc volume of 
    seawater using the TEOS-10 standard. Ocean Modelling.
    """
    # This line ensures that SA is non-negative.
    SA = np.maximum(SA, 0)

    # Set lower temperature limit for water that is much colder than the
    # freezing termperature.
    CT_frozen = - 1.79e-02 - 6.7157e-2*SA - 9.2708e-04*p

    Icold = CT < CT_frozen - 0.2
    if Icold.any():
        CT[Icold] = np.nan

    #db2Pa = 1e4                     # factor to convert from dbar to Pa  
    cp0 = 3991.86795711963     # from Eqn. (3.3.3) of IOC et al. (2010). 

    sfac = 0.0248826675584615             # sfac = 1/(40*(35.16504/35)).
    offset = 5.971840214030754e-1                # offset = deltaS*sfac.

    x2 = sfac*SA
    xs = np.sqrt(x2 + offset)
    ys = CT*0.025
    z = p*1e-4
    
    dynamic_enthalpy_dummy = ( z*(h001 + xs*(h101 + xs*(h201 + xs*(h301 
        + xs*(h401 + xs*(h501 + h601*xs))))) + ys*(h011 + xs*(h111 
        + xs*(h211 + xs*(h311 + xs*(h411 + h511*xs)))) + ys*(h021 
        + xs*(h121 + xs*(h221 + xs*(h321 + h421*xs))) + ys*(h031 
        + xs*(h131 + xs*(h231 + h331*xs)) + ys*(h041 + xs*(h141 
        + h241*xs) + ys*(h051 + h151*xs + h061*ys))))) + z*(h002 
        + xs*(h102 + xs*(h202 + xs*(h302 + xs*(h402 + h502*xs)))) 
        + ys*(h012 + xs*(h112 + xs*(h212 + xs*(h312 + h412*xs))) 
        + ys*(h022 + xs*(h122 + xs*(h222 + h322*xs)) + ys*(h032 
        + xs*(h132 + h232*xs) + ys*(h042 + h142*xs + h052*ys)))) 
        + z*(h003 + xs*(h103 + xs*(h203 + xs*(h303 + h403*xs))) 
        + ys*(h013 + xs*(h113 + xs*(h213 + h313*xs)) + ys*(h023 
        + xs*(h123 + h223*xs) + ys*(h033 + h133*xs + h043*ys))) 
        + z*(h004 + xs*(h104 + h204*xs) + ys*(h014 + h114*xs + h024*ys) 
        + z*(h005 + h105*xs + h015*ys + z*(h006 + h007*z)))))) )

    enthalpy_return = cp0*CT + dynamic_enthalpy_dummy*1e8     
    # Note. 1e8 = db2Pa*1e4  

    return(enthalpy_return)


# In[13]:




# In[14]:

@match_args_return
def enthalpy_diff(SA,CT,p_shallow,p_deep):

    """
    gsw.enthalpy_diff            difference of enthalpy at two pressures  
                                                      (76-term equation)
    ====================================================================

    USAGE:
    enthalpy_diff = gsw.enthalpy_diff(SA,CT,p_shallow,p_deep)

    DESCRIPTION:
    Calculates the difference of the specific enthalpy of seawater 
    between two different pressures, p_deep (the deeper pressure) and 
    p_shallow (the shallower pressure), at the same values of SA and CT.  
    This function uses the computationally-efficient expression for
    density in terms of SA, CT and p (Roquet et al., 2014).  The output
    (enthalpy_diff_CT) is the specific enthalpy evaluated at 
    (SA,CT,p_deep) minus the specific enthalpy at (SA,CT,p_shallow). 

    Note that the 76-term equation has been fitted in a restricted range
    of parameter space, and is most accurate inside the "oceanographic 
    funnel" described in IOC et al. (2010).  The GSW library function 
    "gsw.infunnel(SA,CT,p)" is avaialble to be used if one wants to test
    if some of one's data lies outside this "funnel".  

    INPUT:
    SA         =  Absolute Salinity                             [ g/kg ]  
    CT         =  Conservative Temperature (ITS-90)            [ deg C ]
    p_shallow  =  upper sea pressure                            [ dbar ]
                  ( i.e. shallower absolute pressure - 10.1325 dbar ) 
    p_deep     =  lower sea pressure                            [ dbar ]
                  ( i.e. deeper absolute pressure - 10.1325 dbar )

    p_shallow and p_deep may have dimensions Mx1 or 1xN or MxN, 
    where SA and CT are MxN.

    OUTPUT:
    enthalpy_diff_CT  =  difference of specific enthalpy        [ J/kg ]
                         (deep minus shallow)

    AUTHOR: 
    Trevor McDougall & Paul Barker.                 [ help@teos-10.org ]

    VERSION NUMBER: 3.04 (10th December, 2013)

    REFERENCES:
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation 
    of seawater - 2010: Calculation and use of thermodynamic properties.  
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 
    56,UNESCO (English), 196 pp.  Available from http://www.TEOS-10.org
    See Eqns. (3.32.2) and (A.30.6) of this TEOS-10 Manual. 

    McDougall, T. J., 2003: Potential enthalpy: A conservative oceanic 
    variable for evaluating heat content and heat fluxes. Journal of 
    Physical Oceanography, 33, 945-963.  
    See Eqns. (18) and (22)

    Roquet, F., G. Madec, T.J. McDougall, P.M. Barker, 2014: Accurate
    polynomial expressions for the density and specifc volume of 
    seawater using the TEOS-10 standard. Ocean Modelling.
    """
    # This line ensures that SA is non-negative.
    SA = np.maximum(SA, 0)

    # Set lower temperature limit for water that is much colder than the
    # freezing termperature.
    CT_frozen = - 1.79e-02 - 6.7157e-2*SA - 9.2708e-04*p_deep;

    Icold = CT < CT_frozen - 0.2
    if Icold.any():
        CT[Icold] = np.nan

    db2Pa = 1e4                      # factor to convert from dbar to Pa  
    cp0 = 3991.86795711963     # from Eqn. (3.3.3) of IOC et al. (2010).

    sfac = 0.0248826675584615             # sfac = 1/(40*(35.16504/35)).
    offset = 5.971840214030754e-1                # offset = deltaS*sfac.

    x2 = sfac*SA
    xs = np.sqrt(x2 + offset)
    ys = CT*0.025
    z_shallow = p_shallow*1e-4
    z_deep = p_deep*1e-4
    
    part_1 = ( h001 + xs*(h101 + xs*(h201 + xs*(h301 + xs*(h401 + xs*(h501 
            + h601*xs))))) + ys*(h011 + xs*(h111 + xs*(h211 + xs*(h311 
            + xs*(h411 + h511*xs)))) + ys*(h021 + xs*(h121 + xs*(h221 
            + xs*(h321 + h421*xs))) + ys*(h031 + xs*(h131 + xs*(h231 
            + h331*xs)) + ys*(h041 + xs*(h141 + h241*xs) + ys*(h051 
            + h151*xs + h061*ys))))) )

    part_2 = ( h002 + xs*(h102 + xs*(h202 + xs*(h302 + xs*(h402 
            + h502*xs)))) + ys*(h012 + xs*(h112 + xs*(h212 + xs*(h312 
            + h412*xs))) + ys*(h022 + xs*(h122 + xs*(h222 + h322*xs)) 
            + ys*(h032 + xs*(h132 + h232*xs) + ys*(h042 + h142*xs 
            + h052*ys)))) )

    part_3 = ( h003 + xs*(h103 + xs*(h203 + xs*(h303 + h403*xs))) 
            + ys*(h013 + xs*(h113 + xs*(h213 + h313*xs)) + ys*(h023 
            + xs*(h123 + h223*xs) + ys*(h033 + h133*xs + h043*ys))) )

    part_4 = h004 + xs*(h104 + h204*xs) + ys*(h014 + h114*xs + h024*ys)

    part_5 = h005 + h105*xs + h015*ys

    dynamic_enthalpy_part_shallow = ( z_shallow*(part_1 
                                    + z_shallow*(part_2
                                    + z_shallow*(part_3 
                                    + z_shallow*(part_4 
                                    + z_shallow*(part_5 
                                    + z_shallow*(h006 
                                    + h007*z_shallow)))))) )

    dynamic_enthalpy_part_deep = ( z_deep*(part_1 + z_deep*(part_2 
                                + z_deep*(part_3 + z_deep*(part_4 
                                + z_deep*(part_5 + z_deep*(h006 
                                + h007*z_deep)))))) )

    enthalpy_diff_return = (dynamic_enthalpy_part_deep
                            - dynamic_enthalpy_part_shallow)*1e8   
                                        # Note. 1e8 = db2Pa*1e4;

    
    return(enthalpy_diff_return)


# In[14]:




# In[15]:

@match_args_return
def enthalpy_first_derivatives(SA,CT,p):

    """
    gsw.enthalpy_first_derivatives         first derivatives of enthalpy  
    ====================================================================

    USAGE:
    [h_SA, h_CT] = gsw.enthalpy_first_derivatives(SA,CT,p)

    DESCRIPTION:
    Calculates the following two derivatives of specific enthalpy (h) of
    seawater using the computationally-efficient expression for 
    specific volume in terms of SA, CT and p (Roquet et al., 2014).  
    (1) h_SA, the derivative with respect to Absolute Salinity at 
        constant CT and p, and
    (2) h_CT, derivative with respect to CT at constant SA and p. 
        Note that h_P is specific volume (1/rho) it can be caclulated by 
        calling gsw_specvol(SA,CT,p).

    Note that the 76-term equation has been fitted in a restricted range
    of parameter space, and is most accurate inside the "oceanographic 
    funnel" described in IOC et al. (2010).  The GSW library function 
    "gsw.infunnel(SA,CT,p)" is avaialble to be used if one wants to test
    if some of one's data lies outside this "funnel".  

    INPUT:
    SA  =  Absolute Salinity                                    [ g/kg ]
    CT  =  Conservative Temperature (ITS-90)                   [ deg C ]
    p   =  sea pressure                                         [ dbar ]
           ( i.e. absolute pressure - 10.1325 dbar )

    SA & CT need to have the same dimensions.
    p may have dimensions 1x1 or Mx1 or 1xN or MxN, where SA & CT are 
    MxN.

    OUTPUT:
     h_SA  =  The first derivative of specific enthalpy with respect to 
              Absolute Salinity at constant CT and p.     
                                          [ J/(kg (g/kg))]  i.e. [ J/g ] 
    h_CT  =  The first derivative of specific enthalpy with respect to 
             CT at constant SA and p.                       [ J/(kg K) ]

    AUTHOR: 
    Trevor McDougall and Paul Barker.               [ help@teos-10.org ]
     
    VERSION NUMBER: 3.04 (10th December, 2013)

    REFERENCES:
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation 
    of seawater - 2010: Calculation and use of thermodynamic properties.  
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 
    56,UNESCO (English), 196 pp.  Available from http://www.TEOS-10.org.  
    See Eqns. (A.11.18), (A.11.15) and (A.11.12) of this TEOS-10 Manual.   

    McDougall, T. J., 2003: Potential enthalpy: A conservative oceanic 
    variable for evaluating heat content and heat fluxes. Journal of 
    Physical Oceanography, 33, 945-963.  
    See Eqns. (18) and (22)

    Roquet, F., G. Madec, T.J. McDougall, P.M. Barker, 2014: Accurate
    polynomial expressions for the density and specifc volume of
    seawater using the TEOS-10 standard. Ocean Modelling.

    This software is available from http://www.TEOS-10.org
    """
    # This line ensures that SA is non-negative.
    SA = np.maximum(SA, 0)
    
    # db2Pa = 1e4                      # factor to convert from dbar to Pa  
    cp0 = 3991.86795711963     # from Eqn. (3.3.3) of IOC et al. (2010).

    sfac = 0.0248826675584615             # sfac = 1/(40*(35.16504/35)).
    offset = 5.971840214030754e-1  
    
    x2 = sfac*SA
    xs = np.sqrt(x2 + offset)
    ys = CT*0.025
    z = p*1e-4
    
    dynamic_h_SA_part = ( z*(h101 + xs*(2*h201 + xs*(3*h301 + xs*(4*h401 
        + xs*(5*h501 + 6*h601*xs)))) + ys*(h111 + xs*(2*h211 + xs*(3*h311 
        + xs*(4*h411 + 5*h511*xs))) + ys*(h121 + xs*(2*h221 + xs*(3*h321 
        + 4*h421*xs)) + ys*(h131 + xs*(2*h231 + 3*h331*xs) + ys*(h141 
        + 2*h241*xs + h151*ys**5 )))) + z*(h102 + xs*(2*h202 + xs*(3*h302 
        + xs*(4*h402 + 5.*h502*xs))) + ys*(h112 + xs*(2*h212 + xs*(3*h312 
        + 4*h412*xs)) + ys*(h122 + xs*(2*h222 + 3*h322*xs) + ys*(h132 
        + 2*h232*xs + h142*ys ))) + z*(h103 + xs*(2*h203 + xs*(3*h303 
        + 4*h403*xs)) + ys*(h113 + xs*(2*h213 + 3*h313*xs) + ys*(h123 
        + 2*h223*xs + h133*ys)) + z*(h104 + 2*h204*xs + h114*ys 
        + h105*z)))) )

    h_SA_return = 1e8*0.5*sfac*dynamic_h_SA_part/xs

    dynamic_h_CT_part = ( z*(h011 + xs*(h111 + xs*(h211 + xs*(h311 
        + xs*(h411 + h511*xs)))) + ys*(2*(h021 + xs*(h121 + xs*(h221 
        + xs*(h321 + h421*xs)))) + ys*(3*(h031 + xs*(h131 + xs*(h231 
        + h331*xs))) + ys*(4*(h041 + xs*(h141 + h241*xs)) + ys*(5*(h051 
        + h151*xs) + 6*h061*ys)))) + z*(h012 + xs*(h112 + xs*(h212 
        + xs*(h312 + h412*xs))) + ys*(2*(h022 + xs*(h122 + xs*(h222 
        + h322*xs))) + ys*(3*(h032 + xs*(h132 + h232*xs)) + ys*(4*(h042 
        + h142*xs) + 5*h052*ys))) + z*(h013 + xs*(h113 + xs*(h213 
        + h313*xs)) + ys*(2*(h023 + xs*(h123 + h223*xs)) + ys*(3*(h033 
        + h133*xs) + 4*h043*ys)) + z*(h014 + h114*xs + 2*h024*ys 
        + h015*z )))) )

    h_CT_return = cp0 + 1e8*0.025*dynamic_h_CT_part

    return(h_SA_return, h_CT_return)


# In[15]:




# In[16]:

@match_args_return
def enthalpy_second_derivatives(SA,CT,p):

    """
    gsw.enthalpy_second_derivatives       second derivatives of enthalpy
                                                      (76-term equation)
    ====================================================================  

    USAGE:
    [h_SA_SA, h_SA_CT, h_CT_CT]=gsw.enthalpy_second_derivatives(SA,CT,p) 

    DESCRIPTION:
    Calculates the following three second-order derivatives of specific
    enthalpy (h),using the computationally-efficient expression for 
    specific volume in terms of SA, CT and p (Roquet et al., 2014).
    (1) h_SA_SA, second-order derivative with respect to Absolute 
        Salinity at constant CT & p.
    (2) h_SA_CT, second-order derivative with respect to SA & CT at 
        constant p. 
    (3) h_CT_CT, second-order derivative with respect to CT at constant 
        SA and p. 

    Note that the 76-term equation has been fitted in a restricted range
    of parameter space, and is most accurate inside the "oceanographic 
    funnel" described in IOC et al. (2010).  The GSW library function 
    "gsw.infunnel(SA,CT,p)" is avaialble to be used if one wants to test 
    if some of one's data lies outside this "funnel".  

    INPUT:
    SA  =  Absolute Salinity                                    [ g/kg ]
    CT  =  Conservative Temperature (ITS-90)                   [ deg C ]
    p   =  sea pressure                                         [ dbar ]
           ( i.e. absolute pressure - 10.1325 dbar )

    SA & CT need to have the same dimensions.
    p may have dimensions 1x1 or Mx1 or 1xN or MxN, where SA & CT are 
    MxN.

    OUTPUT:
    h_SA_SA  =  The second derivative of specific enthalpy with respect 
                to Absolute Salinity at constant CT & p.
                                                     [ J/(kg (g/kg)^2) ]  
    h_SA_CT  =  The second derivative of specific enthalpy with respect
                to SA and CT at constant p.           [ J/(kg K(g/kg)) ]
    h_CT_CT  =  The second derivative of specific enthalpy with respect 
                to CT at constant SA and p.               [ J/(kg K^2) ]

    AUTHOR:   
    Trevor McDougall and Paul Barker.               [ help@teos-10.org ]

    VERSION NUMBER: 3.04 (10th December, 2013)

    REFERENCES:
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation 
    of seawater - 2010: Calculation and use of thermodynamic properties.  
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 
    56,UNESCO (English), 196 pp.  Available from http://www.TEOS-10.org.  
  
    McDougall, T.J., 2003: Potential enthalpy: A conservative oceanic 
    variable for evaluating heat content and heat fluxes. Journal of 
    Physical Oceanography, 33, 945-963.  
    See Eqns. (18) and (22)

    Roquet, F., G. Madec, T.J. McDougall, P.M. Barker, 2014: Accurate
    polynomial expressions for the density and specifc volume of seawater
    using the TEOS-10 standard. Ocean Modelling.
    """
    # This line ensures that SA is non-negative.
    SA = np.maximum(SA, 0)

    # db2Pa = 1e4                   # factor to convert from dbar to Pa
    # cp0 = 3991.86795711963   # from Eqn. (3.3.3) of IOC et al. (2010).

    sfac = 0.0248826675584615             # sfac = 1/(40*(35.16504/35)).
    offset = 5.971840214030754e-1                # offset = deltaS*sfac.

    x2 = sfac*SA
    xs = np.sqrt(x2 + offset)
    ys = CT*0.025
    z = p*1e-4

    xs2 = xs**2
    
    dynamic_h_SA_SA_part = ( z*(-h101 + xs2*(3*h301 + xs*(8*h401 
        + xs*(15*h501 + 24*h601*xs))) + ys*(- h111 + xs2*(3*h311 
        + xs*(8*h411 + 15*h511*xs)) + ys*(-h121 + xs2*(3*h321 
        + 8*h421*xs) + ys*(-h131 + 3*h331*xs2 + ys*(-h141 -h151*ys)))) 
        + z*(-h102 + xs2*(3*h302 + xs*(8*h402 + 15*h502*xs)) + ys*(-h112
        + xs2*(3.*h312 + 8*h412*xs) + ys*(-h122 + 3*h322*xs2 + ys*(-h132 
        - h142*ys ))) + z*(xs2*(8*h403*xs + 3*h313*ys) + z*(-h103 
        + 3*h303*xs2 + ys*(-h113 + ys*(-h123 - h133*ys)) + z*(-h104 
        - h114*ys - h105*z))))) )

    h_SA_SA_return = 1e8*0.25*sfac*sfac*dynamic_h_SA_SA_part/xs**3

    dynamic_h_SA_CT_part = ( z*(h111 + xs*(2*h211 + xs*(3*h311 
        + xs*(4*h411 + 5*h511*xs))) + ys*(2*h121 + xs*(4*h221 
        + xs*(6*h321 + 8*h421*xs)) + ys*(3*h131 + xs*(6*h231 
        + 9*h331*xs) + ys*(4*h141 + 8*h241*xs + 5*h151*ys ))) + z*(h112 
        + xs*(2*h212 + xs*(3*h312 + 4*h412*xs)) + ys*(2*h122 
        + xs*(4*h222 + 6*h322*xs) + ys*(3*h132 + 6*h232*xs + 4*h142*ys)) 
        + z*(h113 + xs*(2*h213 + 3*h313*xs) + ys*(2*h123 + 4*h223*xs 
        + 3*h133*ys) + h114*z))) )

    h_SA_CT_return = 1e8*0.025*0.5*sfac*dynamic_h_SA_CT_part/xs

    dynamic_h_CT_CT_part = ( z*(2*h021 + xs*(2*h121 + xs*(2*h221 
        + xs*(2*h321 + 2*h421*xs))) + ys*(6*h031 + xs*(6*h131 
        + xs*(6*h231 + 6*h331*xs)) + ys*(12*h041 + xs*(12*h141 
        + 12*h241*xs) + ys*(20*h051 + 20*h151*xs + 30*h061*ys))) 
        + z*(2*h022 + xs*(2*h122 + xs*(2*h222 + 2*h322*xs)) + ys*(6*h032  
        + xs*(6*h132 + 6*h232*xs) + ys*(12*h042 + 12*h142*xs 
        + 20*h052*ys)) + z*(2*h023 + xs*(2*h123 + 2*h223*xs) 
        + ys*(6*h133*xs + 6*h033 + 12*h043*ys) + 2*h024*z))) )

    h_CT_CT_return = 1e8*6.25e-4*dynamic_h_CT_CT_part
    
    return(h_SA_SA_return, h_SA_CT_return, h_CT_CT_return)


# In[16]:




# In[17]:

@match_args_return
def enthalpy_SSO_0_p(p):

    """
    gsw.enthalpy_SSO_0_p                        enthalpy at (SSO,CT=0,p)
                                                          (76-term eqn.)
    ====================================================================
    This function calculates enthalpy at the Standard Ocean Salinity, 
    SSO, and at a Conservative Temperature of zero degrees C, as a 
    function of pressure, p, in dbar, using a streamlined version of the 
    76-term computationally-efficient expression for specific volume, 
    that is, a streamlined version of the code "gsw_enthalpy(SA,CT,p)".

    VERSION NUMBER: 3.04 (10th December, 2013)

    REFERENCES:
    Roquet, F., G. Madec, T.J. McDougall, P.M. Barker, 2014: Accurate
    polynomial expressions for the density and specifc volume of 
    seawater using the TEOS-10 standard. Ocean Modelling.
    """
    z = p*1e-4

    h006 = -2.1078768810e-9 
    h007 =  2.8019291329e-10 

    dynamic_enthalpy_SSO_0_p_test = ( z*(9.726613854843870e-4 
        + z*(-2.252956605630465e-5 + z*(2.376909655387404e-6 
        + z*(-1.664294869986011e-7 + z*(-5.988108894465758e-9 
        + z*(h006 + h007*z)))))) )

    enthalpy_SSO_0_p_return = dynamic_enthalpy_SSO_0_p_test*1e8         
    # Note. 1e8 = db2Pa*1e4;

    return(enthalpy_SSO_0_p_return)


# In[17]:




# In[18]:

@match_args_return
def kappa(SA,CT,p):

    """
    gsw.kappa              isentropic compressibility (76-term equation)       
    ====================================================================

    USAGE:  
    kappa = gsw.kappa(SA,CT,p)

    DESCRIPTION:
    Calculates the isentropic compressibility of seawater. This function  
    has inputs of Absolute Salinity and Conservative Temperature. This 
    function uses the computationally-efficient expression for 
    specific volume in terms of SA, CT and p (Roquet et al., 2014).

    Note that the 76-term equation has been fitted in a restricted range  
    of parameter space, and is most accurate inside the "oceanographic
    funnel" described in IOC et al. (2010).  The GSW library function 
    "gsw.infunnel(SA,CT,p)" is avaialble to be used if one wants to test
    if some of one's data lies outside this "funnel".  
 
    INPUT:
    SA  =  Absolute Salinity                                    [ g/kg ]  
    CT  =  Conservative Temperature (ITS-90)                   [ deg C ]
    p   =  sea pressure                                         [ dbar ]
           ( i.e. absolute pressure - 10.1325 dbar )

    SA & CT need to have the same dimensions.
    p may have dimensions 1x1 or Mx1 or 1xN or MxN, where SA & CT are 
    MxN.

    OUTPUT:
    kappa  =  isentropic compressibility of seawater            [ 1/Pa ]

    AUTHOR: 
    Paul Barker and Trevor McDougall                [ help@teos-10.org ]   

    VERSION NUMBER: 3.04 (10th December, 2013)

    REFERENCES:
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation 
    of seawater - 2010: Calculation and use of thermodynamic properties.  
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 
    56, UNESCO (English), 196 pp.  Available from http://www.TEOS-10.org  
    See Eqn. (2.17.1) of this TEOS-10 Manual. 

    F. Roquet, G. Madec, T.J. McDougall, P.M. Barker, 2014: Accurate
    polynomial expressions for the density and specifc volume of 
    seawater using the TEOS-10 standard.
    """
    # This line ensures that SA is non-negative.
    SA = np.maximum(SA, 0)

    # deltaS = 24;
    sfac = 0.0248826675584615             # sfac = 1/(40*(35.16504/35)).
    offset = 5.971840214030754e-1                # offset = deltaS*sfac.  

    x2 = sfac*SA
    xs = np.sqrt(x2 + offset)
    ys = CT*0.025
    z = p*1e-4

    v = ( v000 + xs*(v010 + xs*(v020 + xs*(v030 + xs*(v040 + xs*(v050 
        + v060*xs))))) + ys*(v100 + xs*(v110 + xs*(v120 + xs*(v130 
        + xs*(v140 + v150*xs)))) + ys*(v200 + xs*(v210 + xs*(v220 
        + xs*(v230 + v240*xs))) + ys*(v300 + xs*(v310 + xs*(v320 
        + v330*xs)) + ys*(v400 + xs*(v410 + v420*xs) + ys*(v500 
        + v510*xs + v600*ys))))) + z*(v001 + xs*(v011 + xs*(v021 
        + xs*(v031 + xs*(v041 + v051*xs)))) + ys*(v101 + xs*(v111 
        + xs*(v121 + xs*(v131 + v141*xs))) + ys*(v201 + xs*(v211 
        + xs*(v221 + v231*xs)) + ys*(v301 + xs*(v311 + v321*xs) 
        + ys*(v401 + v411*xs + v501*ys)))) + z*(v002 + xs*(v012 
        + xs*(v022 + xs*(v032 + v042*xs))) + ys*(v102 + xs*(v112 
        + xs*(v122 + v132*xs)) + ys*(v202 + xs*(v212 + v222*xs) 
        + ys*(v302 + v312*xs + v402*ys))) + z*(v003 + xs*(v013 
        + v023*xs) + ys*(v103 + v113*xs + v203*ys) + z*(v004 + v014*xs 
        + v104*ys + z*(v005 + v006*z))))) )
    
    v_p = ( c000 + xs*(c100 + xs*(c200 + xs*(c300 + xs*(c400 
        + c500*xs)))) + ys*(c010 + xs*(c110 + xs*(c210 + xs*(c310 
        + c410*xs))) + ys*(c020 + xs*(c120 + xs*(c220 + c320*xs)) 
        + ys*(c030 + xs*(c130 + c230*xs) + ys*(c040 + c140*xs 
        + c050*ys)))) + z*(c001 + xs*(c101 + xs*(c201 + xs*(c301 
        + c401*xs))) + ys*(c011 + xs*(c111 + xs*(c211 + c311*xs)) 
        + ys*(c021 + xs*(c121 + c221*xs) + ys*(c031 + c131*xs 
        + c041*ys))) + z*( c002 + xs*(c102 + c202*xs) + ys*(c012 
        + c112*xs + c022*ys) + z*(c003 + c103*xs + c013*ys + z*(c004 
        + c005*z)))) )
    
    kappa_return = -1e-8*v_p/v
    
    return(kappa_return)


# In[18]:




# In[19]:

@match_args_return
def internal_energy(SA,CT,p):

    """
    gsw.internal_energy              specific interal energy of seawater    
                                                      (76-term equation)
    ====================================================================

    USAGE:
    internal_energy = gsw.internal_energy(SA,CT,p)

    DESCRIPTION:
    Calculates specific internal energy of seawater using the 
    computationally-efficient expression for density in terms of SA, 
    CT and p (Roquet et al., 2014).

    Note that the 76-term equation has been fitted in a restricted range
    of parameter space, and is most accurate inside the "oceanographic 
    funnel" described in IOC et al. (2010).  The GSW library function 
    "gsw.infunnel(SA,CT,p)" is avaialble to be used if one wants to test
    if some of one's data lies outside this "funnel".  

    INPUT:
    SA  =  Absolute Salinity                                    [ g/kg ]
    CT  =  Conservative Temperature                            [ deg C ]
    p   =  sea pressure                                         [ dbar ]
           ( i.e. absolute pressure - 10.1325 dbar ) 

    SA & CT need to have the same dimensions.
    p may have dimensions 1x1 or Mx1 or 1xN or MxN, where SA & CT are 
    MxN.

    OUTPUT:
    internal_energy  =  specific internal energy                [ J/kg ]

    AUTHOR: 
    Trevor McDougall and Paul Barker.               [ help@teos-10.org ]

    VERSION NUMBER: 3.04 (10th December, 2013)

    REFERENCES:
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation 
    of seawater - 2010: Calculation and use of thermodynamic properties.  
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 
    56, UNESCO (English), 196 pp.  Available from http://www.TEOS-10.org

    Roquet, F., G. Madec, T.J. McDougall, P.M. Barker, 2014: Accurate
    polynomial expressions for the density and specifc volume of 
    seawater using the TEOS-10 standard. Ocean Modelling.
    """
    # This line ensures that SA is non-negative.
    SA = np.maximum(SA, 0)

    
    P0 = 101325            # Pressure (in Pa) of one standard atmosphere  
    db2Pa = 1e4                           # dbar to Pa conversion factor 

    enthalpy_temp = enthalpy(SA,CT,p)
    internal_energy_return = ( enthalpy_temp 
                            - (P0 + db2Pa*p)*specvol(SA,CT,p) )
    
    return(internal_energy_return)


# In[19]:




# In[20]:

@match_args_return
def rho(SA,CT,p):
    
    """
    gsw.rho                           in-situ density (76-term equation)
    ====================================================================

    USAGE:  
    rho = gsw.rho(SA,CT,p)

    DESCRIPTION:
    Calculates in-situ density from Absolute Salinity and Conservative 
    Temperature, using the computationally-efficient expression for
    specific volume in terms of SA, CT and p  (Roquet et al., 2014).

    Note that potential density with respect to reference pressure, pr, 
    is obtained by calling this function with the pressure argument 
    being pr (i.e. "gsw.rho(SA,CT,pr)").

    Note that the computationally-efficient expression has been fitted 
    in a restricted range of parameter space, and is most accurate 
    inside the "oceanographic funnel" described in IOC et al. (2010).  
    The GSW library function "gsw.infunnel(SA,CT,p)" is avaialble to be 
    used if one wants to test if some of one's data lies outside this
    "funnel".  

    INPUT:
    SA  =  Absolute Salinity                                    [ g/kg ]
    CT  =  Conservative Temperature (ITS-90)                   [ deg C ]
    p   =  sea pressure                                         [ dbar ]
           ( i.e. absolute pressure - 10.1325 dbar )

    SA & CT need to have the same dimensions.
    p may have dimensions 1x1 or Mx1 or 1xN or MxN, where SA & CT are 
    MxN.

    OUTPUT:
    rho  =  in-situ density                                     [ kg/m ]

    AUTHOR: 
    Paul Barker and Trevor McDougall                [ help@teos-10.org ]

    VERSION NUMBER: 3.05 (27th November, 2015)

    REFERENCES:
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation 
    of seawater - 2010: Calculation and use of thermodynamic properties.  
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 
    56,UNESCO (English), 196 pp.  Available from http://www.TEOS-10.org
    See appendix A.20 and appendix K of this TEOS-10 Manual. 

    Roquet, F., G. Madec, T.J. McDougall, P.M. Barker, 2014: Accurate
    polynomial expressions for the density and specifc volume of 
    seawater using the TEOS-10 standard. Ocean Modelling.
    """
    # This line ensures that SA is non-negative.
    SA = np.maximum(SA, 0)

    # deltaS = 24
    sfac = 0.0248826675584615             # sfac = 1/(40*(35.16504/35)).
    offset = 5.971840214030754e-1                # offset = deltaS*sfac.     

    x2 = sfac*SA
    xs = np.sqrt(x2 + offset)
    ys = CT*0.025
    z = p*1e-4
    
    v = ( v000 + xs*(v010 + xs*(v020 + xs*(v030 + xs*(v040 
        + xs*(v050 + v060*xs))))) + ys*(v100 + xs*(v110 + xs*(v120 
        + xs*(v130 + xs*(v140 + v150*xs)))) + ys*(v200 + xs*(v210 
        + xs*(v220 + xs*(v230 + v240*xs)))  + ys*(v300 + xs*(v310 
        + xs*(v320 + v330*xs)) + ys*(v400 + xs*(v410 + v420*xs) 
        + ys*(v500 + v510*xs + v600*ys))))) + z*(v001 + xs*(v011 
        + xs*(v021 + xs*(v031 + xs*(v041 + v051*xs)))) + ys*(v101 
        + xs*(v111 + xs*(v121 + xs*(v131 + v141*xs))) + ys*(v201 
        + xs*(v211 + xs*(v221 + v231*xs)) + ys*(v301 + xs*(v311 
        + v321*xs) + ys*(v401 + v411*xs + v501*ys)))) + z*(v002
        + xs*(v012 + xs*(v022 + xs*(v032 + v042*xs))) + ys*(v102 
        + xs*(v112 + xs*(v122 + v132*xs)) + ys*(v202 + xs*(v212 
        + v222*xs) + ys*(v302 + v312*xs + v402*ys))) + z*(v003 
        + xs*(v013 + v023*xs) + ys*(v103 + v113*xs + v203*ys) 
        + z*(v004 + v014*xs + v104*ys + z*(v005 + v006*z))))) )
    
    rho_return = 1/v
    
    return(rho_return)


# In[20]:




# In[21]:

@match_args_return
def rho_alpha_beta(SA,CT,p):
    
    """       
    rho_alpha_beta              in-situ density, thermal expansion &    
                       saline contraction coefficient (76-term equation)    
    ====================================================================

    USAGE:  
    [rho, alpha, beta] = gsw.rho_alpha_beta(SA,CT,p)

    DESCRIPTION:
    Calculates in-situ density, the appropiate thermal expansion 
    coefficient and the appropriate saline contraction coefficient of 
    seawater from Absolute Salinity and Conservative Temperature.  This
    function uses the computationally-efficient expression for specific
    volume in terms of SA, CT and p (Roquet et al., 2014).

    Note that potential density (pot_rho) with respect to reference
    pressure p_ref is obtained by calling this function with the 
    pressure argument being p_ref as in [pot_rho, ~, ~] = 
    gsw.rho_alpha_beta(SA,CT,p_ref).

    Note that the 76-term equation has been fitted in a restricted range
    of parameter space, and is most accurate inside the "oceanographic
    funnel" described in IOC et al. (2010).  The GSW library function 
    "gsw.infunnel(SA,CT,p)" is avaialble to be used if one wants to test
    if some of one's data lies outside this "funnel".  

    INPUT:
    SA  =  Absolute Salinity                                    [ g/kg ]
    CT  =  Conservative Temperature (ITS-90)                   [ deg C ]
    p   =  sea pressure                                         [ dbar ]
           ( i.e. absolute pressure - 10.1325 dbar )

    SA & CT need to have the same dimensions.
    p may have dimensions 1x1 or Mx1 or 1xN or MxN, where SA & CT are 
    MxN.

    OUTPUT:
    rho    =  in-situ density                                   [ kg/m ]
    alpha  =  thermal expansion coefficient                      [ 1/K ]
              with respect to Conservative Temperature
    beta   =  saline (i.e. haline) contraction                  [ kg/g ]
              coefficient at constant Conservative Temperature

    AUTHOR: 
    Paul Barker and Trevor McDougall                [ help@teos-10.org ]

    VERSION NUMBER: 3.05 (27th November, 2015)

    REFERENCES:
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation
    of seawater - 2010: Calculation and use of thermodynamic properties.  
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 
    56,UNESCO (English), 196 pp.  Available from http://www.TEOS-10.org
    See appendix A.20 and appendix K of this TEOS-10 Manual. 

    Roquet, F., G. Madec, T.J. McDougall, P.M. Barker, 2014: Accurate
    polynomial expressions for the density and specifc volume of seawater
    using the TEOS-10 standard. Ocean Modelling.
    """
    # This line ensures that SA is non-negative.
    SA = np.maximum(SA, 0)
                                                                          
    #deltaS = 24 
    sfac = 0.0248826675584615           # sfac = 1/(40*(35.16504/35)).    
    offset = 5.971840214030754e-1              # offset = deltaS*sfac.

    x2 = sfac*SA
    xs = np.sqrt(x2 + offset)
    ys = CT*0.025
    z = p*1e-4
                                                                        
    v = ( v000 + xs*(v010 + xs*(v020 + xs*(v030 + xs*(v040 + xs*(v050 
        + v060*xs))))) + ys*(v100 + xs*(v110 + xs*(v120 + xs*(v130 
        + xs*(v140 + v150*xs)))) + ys*(v200 + xs*(v210 + xs*(v220 
        + xs*(v230 + v240*xs))) + ys*(v300 + xs*(v310 + xs*(v320 
        + v330*xs)) + ys*(v400 + xs*(v410 + v420*xs) + ys*(v500 
        + v510*xs + v600*ys))))) + z*(v001 + xs*(v011 + xs*(v021 
        + xs*(v031 + xs*(v041 + v051*xs)))) + ys*(v101 + xs*(v111
        + xs*(v121 + xs*(v131 + v141*xs))) + ys*(v201 + xs*(v211 
        + xs*(v221 + v231*xs)) + ys*(v301 + xs*(v311 + v321*xs) 
        + ys*(v401 + v411*xs + v501*ys)))) + z*(v002 + xs*(v012 
        + xs*(v022 + xs*(v032 + v042*xs))) + ys*(v102 + xs*(v112 
        + xs*(v122 + v132*xs)) + ys*(v202 + xs*(v212 + v222*xs) 
        + ys*(v302 + v312*xs + v402*ys))) + z*(v003 + xs*(v013 
        + v023*xs) + ys*(v103 + v113*xs + v203*ys) + z*(v004 
        + v014*xs + v104*ys + z*(v005 + v006*z))))) )

    rho_return = 1/v

    v_CT = ( a000 + xs*(a100 + xs*(a200 + xs*(a300 + xs*(a400 + a500*xs)))) 
        + ys*(a010 + xs*(a110 + xs*(a210 + xs*(a310 + a410*xs)))
        + ys*(a020 + xs*(a120 + xs*(a220 + a320*xs)) + ys*(a030 
        + xs*(a130 + a230*xs) + ys*(a040 + a140*xs + a050*ys )))) 
        + z*(a001 + xs*(a101 + xs*(a201 + xs*(a301 + a401*xs))) 
        + ys*(a011 + xs*(a111 + xs*(a211 + a311*xs)) + ys*(a021 
        + xs*(a121 + a221*xs) + ys*(a031 + a131*xs + a041*ys))) 
        + z*(a002 + xs*(a102 + xs*(a202 + a302*xs)) + ys*(a012 
        + xs*(a112 + a212*xs) + ys*(a022 + a122*xs + a032*ys)) 
        + z*(a003 + a103*xs + a013*ys + a004*z))) ) 

    alpha_return = 0.025*v_CT/v
        
    v_SA_part = ( b000 + xs*(b100 + xs*(b200 + xs*(b300 + xs*(b400 
        + b500*xs)))) + ys*(b010 + xs*(b110 + xs*(b210 + xs*(b310 
        + b410*xs))) + ys*(b020 + xs*(b120 + xs*(b220 + b320*xs)) 
        + ys*(b030 + xs*(b130 + b230*xs) + ys*(b040 + b140*xs 
        + b050*ys)))) + z*(b001 + xs*(b101 + xs*(b201 + xs*(b301 
        + b401*xs))) + ys*(b011 + xs*(b111 + xs*(b211 + b311*xs)) 
        + ys*(b021 + xs*(b121 + b221*xs) + ys*(b031 + b131*xs 
        + b041*ys))) + z*(b002 + xs*(b102 + xs*(b202 + b302*xs)) 
        + ys*(b012 + xs*(b112 + b212*xs) + ys*(b022 + b122*xs 
        + b032*ys)) + z*(b003 +  b103*xs + b013*ys + b004*z))) )
 
    beta_return = -v_SA_part*0.5*sfac/(v*xs)
    
    return(rho_return, alpha_return, beta_return)


# In[21]:




# In[22]:

@match_args_return
def rho_first_derivatives(SA,CT,p):

    """
    gsw.rho_first_derivatives           SA, CT and p partial derivatives
                                           of density (76-term equation)
    ====================================================================

    USAGE:  
    [drho_dSA, drho_dCT, drho_dP] = gsw.rho_first_derivatives(SA,CT,p)

    DESCRIPTION:
    Calculates the three (3) partial derivatives of in-situ density with 
    respect to Absolute Salinity, Conservative Temperature and pressure.  
    Note that the pressure derivative is done with respect to pressure 
    in Pa, not dbar.  This function uses the computationally-efficient 
    expression for specific volume in terms of SA, CT and p (Roquet et 
    al., 2014).  

    Note that the 76-term equation has been fitted in a restricted range
    of parameter space, and is most accurate inside the "oceanographic 
    funnel" described in IOC et al. (2010).  The GSW library function 
    "gsw.infunnel(SA,CT,p)" is avaialble to be used if one wants to test
    if some of one's data lies outside this "funnel".  

    INPUT:
    SA  =  Absolute Salinity                                    [ g/kg ]  
    CT  =  Conservative Temperature (ITS-90)                   [ deg C ]
    p   =  sea pressure                                         [ dbar ]
           ( i.e. absolute pressure - 10.1325 dbar )

    SA & CT need to have the same dimensions.
    p may have dimensions 1x1 or Mx1 or 1xN or MxN, where SA & CT are 
    MxN.

    OUTPUT:
    drho_dSA  =  partial derivatives of density         [ kg^2/(g m^3) ]
                 with respect to Absolute Salinity
    drho_dCT  =  partial derivatives of density           [ kg/(K m^3) ]
                 with respect to Conservative Temperature
    drho_dP   =  partial derivatives of density          [ kg/(Pa m^3) ]   
                 with respect to pressure in Pa

    AUTHOR: 
    Paul Barker and Trevor McDougall                [ help@teos-10.org ]

    VERSION NUMBER: 3.04 (10th December, 2013)

    REFERENCES:
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation 
    of seawater - 2010: Calculation and use of thermodynamic properties.  
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 
    56, UNESCO (English), 196 pp.  Available from http://www.TEOS-10.org
    See appendix A.20 and appendix K of this TEOS-10 Manual. 

    Roquet, F., G. Madec, T.J. McDougall, P.M. Barker, 2014: Accurate
    polynomial expressions for the density and specifc volume of  
    seawater using the TEOS-10 standard. Ocean Modelling.
    """
    # This line ensures that SA is non-negative.
    SA = np.maximum(SA, 0)

    # deltaS = 24
    sfac = 0.0248826675584615             # sfac = 1/(40*(35.16504/35)).
    offset = 5.971840214030754e-1                # offset = deltaS*sfac.     

    x2 = sfac*SA
    xs = np.sqrt(x2 + offset)
    ys = CT*0.025
    z = p*1e-4
    
    v_SA = ( b000 + xs*(b100 + xs*(b200 + xs*(b300 + xs*(b400 
        + b500*xs)))) + ys*(b010 + xs*(b110 + xs*(b210 + xs*(b310 
        + b410*xs))) + ys*(b020 + xs*(b120 + xs*(b220 + b320*xs)) 
        + ys*(b030 + xs*(b130 + b230*xs) + ys*(b040 + b140*xs 
        + b050*ys)))) + z*(b001 + xs*(b101 + xs*(b201 + xs*(b301 
        + b401*xs))) + ys*(b011 + xs*(b111 + xs*(b211 + b311*xs)) 
        + ys*(b021 + xs*(b121 + b221*xs) + ys*(b031 + b131*xs 
        + b041*ys))) + z*(b002 + xs*(b102 + xs*(b202 + b302*xs))
        + ys*(b012 + xs*(b112 + b212*xs) + ys*(b022 + b122*xs 
        + b032*ys)) + z*(b003 +  b103*xs + b013*ys + b004*z))) )

    v_CT = ( a000 + xs*(a100 + xs*(a200 + xs*(a300 + xs*(a400 
        + a500*xs)))) + ys*(a010 + xs*(a110 + xs*(a210 + xs*(a310 
        + a410*xs))) + ys*(a020 + xs*(a120 + xs*(a220 + a320*xs)) 
        + ys*(a030 + xs*(a130 + a230*xs) + ys*(a040 + a140*xs 
        + a050*ys )))) + z*(a001 + xs*(a101 + xs*(a201 + xs*(a301 
        + a401*xs))) + ys*(a011 + xs*(a111 + xs*(a211 + a311*xs)) 
        + ys*(a021 + xs*(a121 + a221*xs) + ys*(a031 + a131*xs 
        + a041*ys))) + z*(a002 + xs*(a102 + xs*(a202 + a302*xs)) 
        + ys*(a012 + xs*(a112 + a212*xs) + ys*(a022 + a122*xs 
        + a032*ys)) + z*(a003 + a103*xs + a013*ys + a004*z))) )

    v_p = ( c000 + xs*(c100 + xs*(c200 + xs*(c300 + xs*(c400 
        + c500*xs)))) + ys*(c010 + xs*(c110 + xs*(c210 + xs*(c310 
        + c410*xs))) + ys*(c020 + xs*(c120 + xs*(c220 + c320*xs)) 
        + ys*(c030 + xs*(c130 + c230*xs) + ys*(c040 + c140*xs 
        + c050*ys)))) + z*(c001 + xs*(c101 + xs*(c201 + xs*(c301 
        + c401*xs))) + ys*(c011 + xs*(c111 + xs*(c211 + c311*xs)) 
        + ys*(c021 + xs*(c121 + c221*xs) + ys*(c031 + c131*xs 
        + c041*ys))) + z*(c002 + xs*(c102 + c202*xs) + ys*(c012 
        + c112*xs + c022*ys) + z*(c003 + c103*xs + c013*ys + z*(c004 
        + c005*z)))) )
  
    v = ( v000 + xs*(v010 + xs*(v020 + xs*(v030 + xs*(v040 + xs*(v050 
        + v060*xs))))) + ys*(v100 + xs*(v110 + xs*(v120 + xs*(v130 
        + xs*(v140 + v150*xs)))) + ys*(v200 + xs*(v210 + xs*(v220 
        + xs*(v230 + v240*xs))) + ys*(v300 + xs*(v310 + xs*(v320 
        + v330*xs)) + ys*(v400 + xs*(v410 + v420*xs) + ys*(v500 
        + v510*xs + v600*ys))))) + z*(v001 + xs*(v011 + xs*(v021 
        + xs*(v031 + xs*(v041 + v051*xs)))) + ys*(v101 + xs*(v111 
        + xs*(v121 + xs*(v131 + v141*xs))) + ys*(v201 + xs*(v211 
        + xs*(v221 + v231*xs)) + ys*(v301 + xs*(v311 + v321*xs) 
        + ys*(v401 + v411*xs + v501*ys)))) + z*(v002 + xs*(v012 
        + xs*(v022 + xs*(v032 + v042*xs))) + ys*(v102 + xs*(v112 
        + xs*(v122 + v132*xs)) + ys*(v202 + xs*(v212 + v222*xs) 
        + ys*(v302 + v312*xs + v402*ys))) + z*(v003 + xs*(v013 
        + v023*xs) + ys*(v103 + v113*xs + v203*ys) + z*(v004 + v014*xs 
        + v104*ys + z*(v005 + v006*z))))) )

    rho2 = (1/v)**2

    drho_dSA_return = -rho2*0.5*sfac*v_SA/xs

    drho_dCT_return = -rho2*0.025*v_CT

    Pa2db = 1e-4                    # factor to convert from Pa to dbar

    drho_dP_return = 1e-4*Pa2db*-rho2*v_p

    return(drho_dSA_return, drho_dCT_return, drho_dP_return)


# In[22]:




# In[23]:

@match_args_return
def rho_second_derivatives(SA,CT,p):
    
    """
    gsw.rho_second_derivatives                        second derivatives  
                                               of rho (76-term equation) 
    ====================================================================

    USAGE:
    [rho_SA_SA, rho_SA_CT, rho_CT_CT, rho_SA_P, rho_CT_P] = ...
                                     gsw.rho_second_derivatives(SA,CT,p)    

    DESCRIPTION:
    Calculates the following three second-order derivatives of rho 
    (1) rho_SA_SA, second-order derivative with respect to Absolute  
        Salinity at constant CT & p.
    (2) rho_SA_CT, second-order derivative with respect to SA & CT at 
        constant p. 
    (3) rho_CT_CT, second-order derivative with respect to CT at 
        constant SA & p. 
    (4) rho_SA_P, second-order derivative with respect to SA & P at 
        constant CT. 
    (5) rho_CT_P, second-order derivative with respect to CT & P at 
        constant SA. 


    Note that this function uses the using the computationally-efficient
    expression for specific volume (Roquet et al., 2014).  There is an 
    alternative to calling this function, namely 
    gsw_rho_second_derivatives(SA,CT,p) which uses the full Gibbs function 
    (IOC et al., 2010).   

    This 76-term equation has been fitted in a restricted range of parameter
    space, and is most accurate inside the "oceanographic funnel" described 
    in McDougall et al. (2010).  The GSW library function 
    "gsw.infunnel(SA,CT,p)" is avaialble to be used if one wants to test if 
    some of one's data lies outside this "funnel".  

    INPUT:
    SA  =  Absolute Salinity                                    [ g/kg ]
    CT  =  Conservative Temperature (ITS-90)                   [ deg C ]
    p   =  sea pressure                                         [ dbar ]
           ( i.e. absolute pressure - 10.1325 dbar )

    SA & CT need to have the same dimensions.
    p may have dimensions 1x1 or Mx1 or 1xN or MxN, where SA & CT are 
    MxN.

    OUTPUT:
    rho_SA_SA = The second-order derivative of rho with respect to 
                Absolute Salinity at constant CT & p.[ J/(kg (g/kg)^2) ]
    rho_SA_CT = The second-order derivative of rho with respect to 
                SA and CT at constant p.              [ J/(kg K(g/kg)) ] 
    rho_CT_CT = The second-order derivative of rho with respect to CT at 
                constant SA & p
    rho_SA_P  = The second-order derivative with respect to SA & P at 
                constant CT. 
    rho_CT_P  = The second-order derivative with respect to CT & P at 
                constant SA. 

    AUTHOR:   
    Trevor McDougall and Paul Barker.               [ help@teos-10.org ]

    VERSION NUMBER: 3.04 (10th December, 2013)

    REFERENCES:
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation 
    of seawater - 2010: Calculation and use of thermodynamic properties.  
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 
    56,UNESCO (English), 196 pp.  Available from http://www.TEOS-10.org.  

    Roquet, F., G. Madec, T.J. McDougall, P.M. Barker, 2014: Accurate
    polynomial expressions for the density and specifc volume of 
    seawater using the TEOS-10 standard. Ocean Modelling.
    """
    # This line ensures that SA is non-negative.
    SA = np.maximum(SA, 0)

    rec_v = 1/specvol(SA,CT,p)
    [v_SA, v_CT, v_P] = specvol_first_derivatives(SA,CT,p)
    [v_SA_SA, v_SA_CT, v_CT_CT, v_SA_P, v_CT_P] = specvol_second_derivatives(SA,CT,p) 

    rec_v2 = rec_v**2
    rec_v3 = rec_v2*rec_v

    rho_CT_CT_return = -v_CT_CT*rec_v2 + 2*v_CT**2*rec_v3
    rho_SA_CT_return = -v_SA_CT*rec_v2 + 2*v_SA*v_CT*rec_v3
    rho_SA_SA_return = -v_SA_SA*rec_v2 + 2*v_SA**2*rec_v3
    rho_SA_P_return = -v_SA_P*rec_v2 + 2*v_SA*v_P*rec_v3
    rho_CT_P_return = -v_CT_P*rec_v2 + 2*v_CT*v_P*rec_v3
    
    return(rho_SA_SA_return, rho_SA_CT_return, rho_CT_CT_return, rho_SA_P_return, rho_CT_P_return)


# In[23]:




# In[24]:

@match_args_return
def rho_first_derivatives_wrt_enthalpy(SA,CT,p):

    """
    gsw.rho_first_derivatives_wrt_enthalpy             first derivatives  
                                specific volume with respect to enthalpy
    ====================================================================

    USAGE:
    [rho_SA, rho_h] = gsw.rho_first_derivatives_wrt_enthalpy(SA,CT,p)

    DESCRIPTION:
    Calculates the following two first-order derivatives of specific
    volume (v),
    (1) rho_SA, first-order derivative with respect to Absolute Salinity 
        at constant CT & p.
    (2) rho_h, first-order derivative with respect to SA & CT at 
        constant p. 

    Note that this function uses the using the computationally-efficient
    expression for specific volume (Roquet et al., 2014).  There is an 
    alternative to calling this function, namely 
    gsw_specvol_first_derivatives_wrt_enthalpy_CT_exact(SA,CT,p) which 
    uses the full Gibbs function (IOC et al., 2010).   

    This 76-term equation has been fitted in a restricted range of
    parameter space, and is most accurate inside the "oceanographic 
    funnel" described in McDougall et al. (2010).  The GSW library 
    function "gsw.infunnel(SA,CT,p)" is avaialble to be used if one 
    wants to test if some of one's data lies outside this "funnel".  

    INPUT:
    SA  =  Absolute Salinity                                    [ g/kg ]
    CT  =  Conservative Temperature (ITS-90)                   [ deg C ]
    p   =  sea pressure                                         [ dbar ]
           ( i.e. absolute pressure - 10.1325 dbar )

    SA & CT need to have the same dimensions.
    p may have dimensions 1x1 or Mx1 or 1xN or MxN, where SA & CT are 
    MxN.

    OUTPUT:
    rho_SA =  The first derivative of rho with respect to 
              Absolute Salinity at constant CT & p.  [ J/(kg (g/kg)^2) ] 
    rho_h  =  The first derivative of rho with respect to 
              SA and CT at constant p.                [ J/(kg K(g/kg)) ]

    AUTHOR:   
    Trevor McDougall and Paul Barker.               [ help@teos-10.org ]

    VERSION NUMBER: 3.04 (10th December, 2013)

    REFERENCES:
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation 
    of seawater - 2010: Calculation and use of thermodynamic properties.  
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 
    56,UNESCO (English), 196 pp.  Available from http://www.TEOS-10.org.  

    F. Roquet, G. Madec, T.J. McDougall, P.M. Barker, 2014: Accurate
    polynomial expressions for the density and specifc volume of 
    seawater using the TEOS-10 standard. Ocean Modelling.
    """
    # This line ensures that SA is non-negative.
    SA = np.maximum(SA, 0)

    rec_v = 1/specvol(SA,CT,p)
    [v_SA, v_h] = specvol_first_derivatives_wrt_enthalpy(SA,CT,p)

    rec_v2 = rec_v**2

    rho_h_return = -v_h*rec_v2
    rho_SA_return = -v_SA*rec_v2
    
    return(rho_SA_return, rho_h_return)


# In[24]:




# In[25]:

@match_args_return
def rho_second_derivatives_wrt_enthalpy(SA,CT,p):

    """
    gsw.rho_second_derivatives_wrt_enthalpy           second derivatives  
                                         of rho with respect to enthalpy
    ====================================================================

    USAGE:
    [rho_SA_SA, rho_SA_h, rho_h_h] = ...
                        gsw.rho_second_derivatives_wrt_enthalpy(SA,CT,p)

    DESCRIPTION:
    Calculates the following three second-order derivatives of rho with 
    respect to enthalpy,
    (1) rho_SA_SA, second-order derivative with respect to Absolute 
        Salinity at constant h & p.
    (2) rho_SA_h, second-order derivative with respect to SA & h at 
        constant p. 
    (3) rho_h_h, second-order derivative with respect to h at 
        constant SA & p. 

    Note that this function uses the using the computationally-efficient
    expression for specific volume (Roquet et al., 2014).  There is an 
    alternative to calling this function, namely 
    gsw_rho_second_derivatives_wrt_enthalpy_CT_exact(SA,CT,p) which uses   
    the full Gibbs function (IOC et al., 2010).   

    This 76-term equation has been fitted in a restricted range of 
    parameter space, and is most accurate inside the "oceanographic
    funnel" described in McDougall et al. (2010).  The GSW library 
    function "gsw.infunnel(SA,CT,p)" is avaialble to be used if one 
    wants to test if some of one's data lies outside this "funnel".  

    INPUT:
    SA  =  Absolute Salinity                                    [ g/kg ]
    CT  =  Conservative Temperature (ITS-90)                   [ deg C ]
    p   =  sea pressure                                         [ dbar ]
           ( i.e. absolute pressure - 10.1325 dbar )

    SA & CT need to have the same dimensions.
    p may have dimensions 1x1 or Mx1 or 1xN or MxN, where SA & CT are 
    MxN.

    OUTPUT:
    rho_SA_SA = The second-order derivative of rho with respect to 
                Absolute Salinity at constant h & p. [ J/(kg (g/kg)^2) ]
    rho_SA_h  = The second-order derivative of rho with respect to 
                SA and h at constant p.               [ J/(kg K(g/kg)) ]
    rho_h_h   = The second-order derivative of rho with respect to h at 
                constant SA & p

    AUTHOR:   
    Trevor McDougall and Paul Barker.               [ help@teos-10.org ]

    VERSION NUMBER: 3.04 (10th December, 2013)

    REFERENCES:
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of 
    seawater - 2010: Calculation and use of thermodynamic properties.  
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Available from http://www.TEOS-10.org.  

    F. Roquet, G. Madec, T.J. McDougall, P.M. Barker, 2014: Accurate
    polynomial expressions for the density and specifc volume of 
    seawater using the TEOS-10 standard. Ocean Modelling.
    """
    # This line ensures that SA is non-negative.
    SA = np.maximum(SA, 0)

    rec_v = 1/specvol(SA,CT,p)
    [v_SA, v_h] = specvol_first_derivatives_wrt_enthalpy(SA,CT,p)
    [v_SA_SA, v_SA_h, v_h_h] = specvol_second_derivatives_wrt_enthalpy(SA,CT,p)

    rec_v2 = rec_v**2
    rec_v3 = rec_v2*rec_v

    rho_h_h_return = -v_h_h*rec_v2 + 2*v_h**2*rec_v3
    rho_SA_h_return = -v_SA_h*rec_v2 + 2*v_SA*v_h*rec_v3
    rho_SA_SA_return = -v_SA_SA*rec_v2 + 2.*v_SA**2*rec_v3
    
    return(rho_SA_SA_return, rho_SA_h_return, rho_h_h_return)


# In[25]:




# In[26]:

@match_args_return
def SA_from_rho(rho, CT, p):

    """
    gsw.SA_from_rho                       Absolute Salinity from density  
                                                      (76-term equation) 
    ====================================================================

    USAGE:
    SA = gsw.SA_from_rho(rho,CT,p)

    DESCRIPTION:
    Calculates the Absolute Salinity of a seawater sample, for given 
    values of its density, Conservative Temperature and sea pressure 
    (in dbar). This function uses the computationally-efficient 76-term
    expression for density in terms of SA, CT and p (Roquet et al., 
    2014).

    Note that the 76-term equation has been fitted in a restricted range
    of parameter space, and is most accurate inside the "oceanographic 
    funnel" described in IOC et al. (2010).  The GSW library function 
    "gsw.infunnel(SA,CT,p)" is avaialble to be used if one wants to test 
    if some of one's data lies outside this "funnel".  

    INPUT:
    rho =  density of a seawater sample (e.g. 1026 kg/m^3).   [ kg/m^3 ]  
           Note. This input has not had 1000 kg/m^3 subtracted from it. 
           That is, it is 'density', not 'density anomaly'.
    CT  =  Conservative Temperature (ITS-90)                   [ deg C ]   
    p   =  sea pressure                                         [ dbar ]
           ( i.e. absolute pressure - 10.1325 dbar )

    rho & CT need to have the same dimensions.
    p may have dimensions 1x1 or Mx1 or 1xN or MxN, where rho & CT are
    MxN.

    OUTPUT:
    SA  =  Absolute Salinity.                                   [ g/kg ]
    Note. This is expressed on the Reference-Composition Salinity
    Scale of Millero et al. (2008). 

    AUTHOR: 
    Trevor McDougall & Paul Barker                  [ help@teos-10.org ]
     
    VERSION NUMBER: 3.05 (27th November, 2015)

    REFERENCES:
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation 
    of seawater - 2010: Calculation and use of thermodynamic properties.  
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 
    56,UNESCO (English), 196 pp.  Available from http://www.TEOS-10.org
    See section 2.5 of this TEOS-10 Manual. 

    Millero, F.J., R. Feistel, D.G. Wright, and T.J. McDougall, 2008: 
    The composition of Standard Seawater and the definition of the 
    Reference-Composition Salinity Scale. Deep-Sea Res. I, 55, 50-72. 

    F. Roquet, G. Madec, T.J. McDougall, P.M. Barker, 2014: Accurate
    polynomial expressions for the density and specifc volume of seawater
    using the TEOS-10 standard. Ocean Modelling.
    """
    v_lab = np.ones_like(rho)/rho
    v_0 = specvol(np.zeros_like(rho),CT,p)
    v_50 = specvol(50*np.ones_like(rho),CT,p)
 
    SA_return = 50*(v_lab - v_0)/(v_50 - v_0)  # initial estimate of SA. 
    SA_return[np.logical_or(SA_return < 0, SA_return > 50)] = np.nan

    v_SA = (v_50 - v_0)/50 #initial estimate of v_SA, the SA derivative 
    # of v

    # Begin the modified Newton-Raphson iterative procedure 
    number_of_iterations = 2
    for j in range(0, number_of_iterations):
        SA_old = SA_return
        delta_v = specvol(SA_old,CT,p) - v_lab
        SA_return = SA_old - delta_v/v_SA     # this is half way through   
        # the modified N-R method (McDougall and Wotherspoon, 2012)
        SA_mean = 0.5*(SA_return + SA_old)
        [rho_temp,alpha_dummy,beta_temp] = rho_alpha_beta(SA_mean,CT,p)
        v_SA = - beta_temp / rho_temp 
        SA_return = SA_old - delta_v/v_SA
        # incase input were single numbers
        SA_return = np.asanyarray(SA_return)
        SA_return[np.logical_or(SA_return < 0, SA_return > 50)] = np.nan 
    # After two iterations of this modified Newton-Raphson iteration,
    # the error in SA is no larger than 8x10^-13 g/kg, which 
    # is machine precision for this calculation. 
 
    return(SA_return)


# In[26]:




# In[27]:

@match_args_return
def sigma0(SA,CT):
    
    """
    gsw.sigma0                  potential density anomaly with reference  
                               sea pressure of 0 dbar (76-term equation)
    ====================================================================

    USAGE:  
    sigma0 = gsw.sigma0(SA,CT)

    DESCRIPTION:
    Calculates potential density anomaly with reference pressure of 0 
    dbar, this being this particular potential density minus 1000 
    kg/m^3. This function has inputs of Absolute Salinity and 
    Conservative Temperature.  This function uses the computationally
    -efficient expression for specific volume in terms of SA, CT and 
    p (Roquet et al., 2014).

    Note that the 76-term equation has been fitted in a restricted range
    of parameter space, and is most accurate inside the "oceanographic 
    funnel" described in IOC et al. (2010).  The GSW library function 
    "gsw.infunnel(SA,CT,p)" is avaialble to be used if one wants to test
    if some of one's data lies outside this "funnel".  

    INPUT:
    SA  =  Absolute Salinity                                    [ g/kg ]
    CT  =  Conservative Temperature (ITS-90)                   [ deg C ]

    SA & CT need to have the same dimensions.

    OUTPUT:
    sigma1  =  potential density anomaly with                 [ kg/m^3 ]  
               respect to a reference pressure of 0 dbar,   
               that is, this potential density - 1000 kg/m^3.

    AUTHOR: 
    Paul Barker and Trevor McDougall                [ help@teos-10.org ]

    VERSION NUMBER: 3.04 (10th December, 2013)

    REFERENCES:
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation 
    of seawater - 2010: Calculation and use of thermodynamic properties.  
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 
    56, UNESCO (English), 196 pp.  Available from http://www.TEOS-10.org
    See Eqn. (A.30.1) of this TEOS-10 Manual. 

    F. Roquet, G. Madec, T.J. McDougall, P.M. Barker, 2014: Accurate
    polynomial expressions for the density and specifc volume of 
    seawater using the TEOS-10 standard. Ocean Modelling.
    """
    # This line ensures SA is non-negative
    SA = np.maximum(SA,0)

    # deltaS = 24
    sfac = 0.0248826675584615             # sfac = 1/(40*(35.16504/35)).  
    offset = 5.971840214030754e-1                # offset = deltaS*sfac.

    x2 = sfac*SA
    xs = np.sqrt(x2 + offset)
    ys = CT*0.025
    
    vp0 = ( v000 + xs*(v010 + xs*(v020 + xs*(v030 + xs*(v040 + xs*(v050 
        + v060*xs))))) + ys*(v100 + xs*(v110 + xs*(v120 + xs*(v130 
        + xs*(v140 + v150*xs)))) + ys*(v200 + xs*(v210 + xs*(v220
        + xs*(v230 + v240*xs))) + ys*(v300 + xs*(v310 + xs*(v320 
        + v330*xs)) + ys*(v400 + xs*(v410 + v420*xs) + ys*(v500 
        + v510*xs + v600*ys))))) )

    sigma0_return = 1/vp0 - 1000
    
    return(sigma0_return)


# In[27]:




# In[28]:

@match_args_return
def sigma1(SA,CT):
    
    """
    gsw.sigma1                  potential density anomaly with reference  
                            sea pressure of 1000 dbar (76-term equation)
    ====================================================================

    USAGE:  
    sigma1 = gsw.sigma1(SA,CT)

    DESCRIPTION:
    Calculates potential density anomaly with reference pressure of 1000 
    dbar, this being this particular potential density minus 1000 
    kg/m^3. This function has inputs of Absolute Salinity and 
    Conservative Temperature.  This function uses the computationally
    -efficient expression for specific volume in terms of SA, CT and 
    p (Roquet et al., 2014).

    Note that the 76-term equation has been fitted in a restricted range
    of parameter space, and is most accurate inside the "oceanographic 
    funnel" described in IOC et al. (2010).  The GSW library function 
    "gsw.infunnel(SA,CT,p)" is avaialble to be used if one wants to test
    if some of one's data lies outside this "funnel".  

    INPUT:
    SA  =  Absolute Salinity                                    [ g/kg ]
    CT  =  Conservative Temperature (ITS-90)                   [ deg C ]

    SA & CT need to have the same dimensions.

    OUTPUT:
    sigma1  =  potential density anomaly with                 [ kg/m^3 ]  
               respect to a reference pressure of 1000 dbar,   
               that is, this potential density - 1000 kg/m^3.

    AUTHOR: 
    Paul Barker and Trevor McDougall                [ help@teos-10.org ]

    VERSION NUMBER: 3.04 (10th December, 2013)

    REFERENCES:
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation 
    of seawater - 2010: Calculation and use of thermodynamic properties.  
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 
    56, UNESCO (English), 196 pp.  Available from http://www.TEOS-10.org
    See Eqn. (A.30.1) of this TEOS-10 Manual. 

    F. Roquet, G. Madec, T.J. McDougall, P.M. Barker, 2014: Accurate
    polynomial expressions for the density and specifc volume of 
    seawater using the TEOS-10 standard. Ocean Modelling.
    """
    # This line ensures that SA is non-negative.
    SA = np.maximum(SA, 0)
    
    pr1000 = 1000*np.ones_like(SA)

    rho1 = rho(SA,CT,pr1000)

    sigma1_return = rho1 - 1000
    
    return(sigma1_return)


# In[28]:




# In[29]:

@match_args_return
def sigma2(SA,CT):

    """
    gsw.sigma2                  potential density anomaly with reference
                            sea pressure of 2000 dbar (76-term equation)
    ====================================================================

    USAGE:  
    sigma2 = gsw.sigma2(SA,CT)

    DESCRIPTION:
    Calculates potential density anomaly with reference pressure of 2000   
    dbar, this being this particular potential density minus 1000 
    kg/m^3. Temperature.  This function uses the computationally-
    efficient expression for specific volume in terms of SA, CT and p 
    (Roquet et al., 2014).

    Note that the 76-term equation has been fitted in a restricted range
    of parameter space, and is most accurate inside the "oceanographic 
    funnel" described in IOC et al. (2010).  The GSW library function 
    "gsw.infunnel(SA,CT,p)" is avaialble to be used if one wants to test
    if some of one's data lies outside this "funnel".  

    INPUT:
    SA  =  Absolute Salinity                                    [ g/kg ]  
    CT  =  Conservative Temperature (ITS-90)                   [ deg C ]

    SA & CT need to have the same dimensions.

    OUTPUT:
    sigma2  =  potential density anomaly with                 [ kg/m^3 ]
               respect to a reference pressure of 2000 dbar,   
               that is, this potential density - 1000 kg/m^3.

    AUTHOR: 
    Paul Barker and Trevor McDougall                [ help@teos-10.org ]

    VERSION NUMBER: 3.04 (10th December, 2013)

    REFERENCES:
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation 
    of seawater - 2010: Calculation and use of thermodynamic properties.  
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 
    56,UNESCO (English), 196 pp.  Available from http://www.TEOS-10.org
    See Eqn. (A.30.1) of this TEOS-10 Manual. 

    F. Roquet, G. Madec, T.J. McDougall, P.M. Barker, 2014: Accurate
    polynomial expressions for the density and specifc volume of 
    seawater using the TEOS-10 standard. Ocean Modelling.
    """
    # This line ensures that SA is non-negative.
    SA = np.maximum(SA, 0)
    
    pr2000 = 2000*np.ones_like(SA)

    rho2 = rho(SA,CT,pr2000)

    sigma2_return = rho2 - 1000
    
    return(sigma2_return)


# In[29]:




# In[30]:

@match_args_return
def sigma3(SA,CT):

    """
    gsw.sigma3                  potential density anomaly with reference  
                            sea pressure of 3000 dbar (76-term equation)
    ====================================================================

    USAGE:  
    sigma3 = gsw.sigma3(SA,CT)

    DESCRIPTION:
    Calculates potential density anomaly with reference pressure of 3000 
    dbar, this being this particular potential density minus 1000 
    kg/m^3. Temperature.  This function uses the computationally-
    efficient expression for specific volume in terms of SA, CT and p 
    (Roquet et al., 2014).

    Note that the 76-term equation has been fitted in a restricted range 
    of parameter space, and is most accurate inside the "oceanographic 
    funnel" described in IOC et al. (2010).  The GSW library function 
    "gsw.infunnel(SA,CT,p)" is avaialble to be used if one wants to test 
    if some of one's data lies outside this "funnel".  

    INPUT:
    SA  =  Absolute Salinity                                    [ g/kg ]
    CT  =  Conservative Temperature (ITS-90)                   [ deg C ]

    SA & CT need to have the same dimensions.

    OUTPUT:
    sigma3  =  potential density anomaly with                 [ kg/m^3 ]
               respect to a reference pressure of 3000 dbar,   
               that is, this potential density - 1000 kg/m^3.

    AUTHOR: 
    Paul Barker and Trevor McDougall                [ help@teos-10.org ]

    VERSION NUMBER: 3.04 (10th December, 2013)

    REFERENCES:
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation 
    of seawater - 2010: Calculation and use of thermodynamic properties.  
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 
    56,UNESCO (English), 196 pp.  Available from http://www.TEOS-10.org
    See Eqn. (A.30.1) of this TEOS-10 Manual. 

    F. Roquet, G. Madec, T.J. McDougall, P.M. Barker, 2014: Accurate
    polynomial expressions for the density and specifc volume of 
    seawater using the TEOS-10 standard. Ocean Modelling.
    """
    # This line ensures that SA is non-negative.
    SA = np.maximum(SA, 0)
    
    pr3000 = 3000*np.ones_like(SA)

    rho3 = rho(SA,CT,pr3000)
    
    sigma3_return = rho3 - 1000
    
    return(sigma3_return)


# In[30]:




# In[31]:

@match_args_return
def sigma4(SA,CT):

    """
    gsw.sigma4                  potential density anomaly with reference  
                            sea pressure of 4000 dbar (76-term equation)
    ====================================================================  

    USAGE:  
    sigma4 = gsw.sigma4(SA,CT)

    DESCRIPTION:
    Calculates potential density anomaly with reference pressure of 4000   
    dbar, this being this particular potential density minus 1000 
    kg/m^3. Temperature.  This function uses the computationally-
    efficient expression for specific volume in terms of SA, CT and p 
    (Roquet et al., 2014).

    Note that the 76-term equation has been fitted in a restricted range
    of parameter space, and is most accurate inside the "oceanographic 
    funnel" described in IOC et al. (2010).  The GSW library function 
    "gsw.infunnel(SA,CT,p)" is avaialble to be used if one wants to test
    if some of one's data lies outside this "funnel".  

    INPUT:
    SA  =  Absolute Salinity                                    [ g/kg ]
    CT  =  Conservative Temperature (ITS-90)                   [ deg C ]

    SA & CT need to have the same dimensions.

    OUTPUT:
    sigma4  =  potential density anomaly with                 [ kg/m^3 ]
               respect to a reference pressure of 4000 dbar,   
               that is, this potential density - 1000 kg/m^3.

    AUTHOR: 
    Paul Barker and Trevor McDougall                [ help@teos-10.org ]

    VERSION NUMBER: 3.04 (10th December, 2013)

    REFERENCES:
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation 
    of seawater - 2010: Calculation and use of thermodynamic properties.  
    Intergovernmental Oceanographic Commission, Manuals and Guides No.  
    56,UNESCO (English), 196 pp.  Available from http://www.TEOS-10.org
    See Eqn. (A.30.1) of this TEOS-10 Manual. 

    F. Roquet, G. Madec, T.J. McDougall, P.M. Barker, 2014: Accurate
    polynomial expressions for the density and specifc volume of seawater
    using the TEOS-10 standard. Ocean Modelling.
    """
    # This line ensures that SA is non-negative.
    SA = np.maximum(SA, 0)
    
    pr4000 = 4000*np.ones_like(SA)

    rho4 = rho(SA,CT,pr4000)
    
    sigma4_return = rho4 - 1000
    
    return(sigma4_return)


# In[31]:




# In[32]:

@match_args_return
def sound_speed(SA,CT,p):

    """"
    gsw.sound_speed                       sound speed (76-term equation)
    ====================================================================

    USAGE:  
    sound_speed = gsw.sound_speed(SA,CT,p)

    DESCRIPTION:
    Calculates the speed of sound in seawater.  This function has inputs 
    of Absolute Salinity and Conservative Temperature.  This function 
    uses the computationally-efficient expression for specific volume in
    terms of SA, CT and p (Roquet et al., 2014).

    Note that the 76-term equation has been fitted in a restricted range
    of parameter space, and is most accurate inside the "oceanographic 
    funnel" described in IOC et al. (2010).  The GSW library function 
    "gsw.infunnel(SA,CT,p)" is avaialble to be used if one wants to test
    if some of one's data lies outside this "funnel".  
 
    INPUT:
    SA  =  Absolute Salinity                                    [ g/kg ]
    CT  =  Conservative Temperature (ITS-90)                   [ deg C ]
    p   =  sea pressure                                         [ dbar ]
           ( i.e. absolute pressure - 10.1325 dbar )

    SA & CT need to have the same dimensions.
    p may have dimensions 1x1 or Mx1 or 1xN or MxN, where SA & CT are MxN.

    OUTPUT:
    sound_speed  =  speed of sound in seawater                   [ m/s ]  

    AUTHOR: 
    Paul Barker and Trevor McDougall                [ help@teos-10.org ]   

    VERSION NUMBER: 3.04 (10th December, 2013)

    REFERENCES:
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation 
    of seawater - 2010: Calculation and use of thermodynamic properties.  
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 
    56,UNESCO (English), 196 pp.  Available from http://www.TEOS-10.org
    See Eqn. (2.17.1) of this TEOS-10 Manual. 

    F. Roquet, G. Madec, T.J. McDougall, P.M. Barker, 2014: Accurate
    polynomial expressions for the density and specifc volume of 
    seawater using the TEOS-10 standard. Ocean Modelling.
    """
    # This line ensures that SA is non-negative.
    SA = np.maximum(SA, 0)
    
    # deltaS = 24
    sfac = 0.0248826675584615             # sfac = 1/(40*(35.16504/35)).
    offset = 5.971840214030754e-1                # offset = deltaS*sfac.     

    x2 = sfac*SA
    xs = np.sqrt(x2 + offset)
    ys = CT*0.025
    z = p*1e-4
    
    v = ( v000 + xs*(v010 + xs*(v020 + xs*(v030 + xs*(v040 
        + xs*(v050 + v060*xs))))) + ys*(v100 + xs*(v110 + xs*(v120 
        + xs*(v130 + xs*(v140 + v150*xs)))) + ys*(v200 + xs*(v210 
        + xs*(v220 + xs*(v230 + v240*xs)))  + ys*(v300 + xs*(v310 
        + xs*(v320 + v330*xs)) + ys*(v400 + xs*(v410 + v420*xs) 
        + ys*(v500 + v510*xs + v600*ys))))) + z*(v001 + xs*(v011 
        + xs*(v021 + xs*(v031 + xs*(v041 + v051*xs)))) + ys*(v101 
        + xs*(v111 + xs*(v121 + xs*(v131 + v141*xs))) + ys*(v201 
        + xs*(v211 + xs*(v221 + v231*xs)) + ys*(v301 + xs*(v311 
        + v321*xs) + ys*(v401 + v411*xs + v501*ys)))) + z*(v002
        + xs*(v012 + xs*(v022 + xs*(v032 + v042*xs))) + ys*(v102 
        + xs*(v112 + xs*(v122 + v132*xs)) + ys*(v202 + xs*(v212 
        + v222*xs) + ys*(v302 + v312*xs + v402*ys))) + z*(v003 
        + xs*(v013 + v023*xs) + ys*(v103 + v113*xs + v203*ys) 
        + z*(v004 + v014*xs + v104*ys + z*(v005 + v006*z))))) )
    
    v_p = ( c000 + xs*(c100 + xs*(c200 + xs*(c300 + xs*(c400 
        + c500*xs)))) + ys*(c010 + xs*(c110 + xs*(c210 + xs*(c310 
        + c410*xs))) + ys*(c020 + xs*(c120 + xs*(c220 + c320*xs)) 
        + ys*(c030 + xs*(c130 + c230*xs) + ys*(c040 + c140*xs 
        + c050*ys)))) + z*(c001 + xs*(c101 + xs*(c201 + xs*(c301 
        + c401*xs))) + ys*(c011 + xs*(c111 + xs*(c211 + c311*xs)) 
        + ys*(c021 + xs*(c121 + c221*xs) + ys*(c031 + c131*xs 
        + c041*ys))) + z*( c002 + xs*(c102 + c202*xs) + ys*(c012 
        + c112*xs + c022*ys) + z*(c003 + c103*xs + c013*ys + z*(c004 
        + c005*z)))) )

    sound_speed_return = 10000*np.sqrt(-v**2/v_p)
    
    return(sound_speed_return)


# In[32]:




# In[33]:

@match_args_return
def specvol(SA,CT,p):
    
    """
    gsw_specvol                       specific volume (76-term equation)
    ====================================================================   

    USAGE:  
    specvol = gsw.specvol(SA,CT,p)

    DESCRIPTION:
    Calculates specific volume from Absolute Salinity, Conservative 
    Temperature and pressure, using the computationally-efficient 
    76-term polynomial expression for specific volume 
    (Roquet et al., 2014).

    Note that the 76-term equation has been fitted in a restricted range
    of parameter space, and is most accurate inside the "oceanographic
    funnel" described in McDougall et al. (2011).  The GSW library 
    function "gsw.infunnel(SA,CT,p)" is available to be used if one 
    wants to test if some of one's data lies outside this "funnel".  

    INPUT:
    SA  =  Absolute Salinity                                    [ g/kg ]  
    CT  =  Conservative Temperature (ITS-90)                   [ deg C ]
    p   =  sea pressure                                         [ dbar ]
           ( i.e. absolute pressure - 10.1325 dbar )

    SA & CT need to have the same dimensions.
    p may have dimensions 1x1 or Mx1 or 1xN or MxN, where SA & CT are 
    MxN.

    OUTPUT:
    specvol  =  specific volume                               [ m^3/kg ]  

    AUTHOR: 
    Fabien Roquet

    VERSION NUMBER: 3.05 (27th November, 2015)

    REFERENCES:
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation
    of seawater - 2010: Calculation and use of thermodynamic properties.  
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 
    56, UNESCO (English), 196 pp.  Available from 
    http://www.TEOS-10.org.   
  
    Roquet, F., G. Madec, T.J. McDougall, P.M. Barker, 2014: Accurate
    polynomial expressions for the density and specifc volume of 
    seawater using the TEOS-10 standard. Ocean Modelling.
    """
    #deltaS = 24;
    sfac = 0.0248826675584615           # sfac = 1/(40*(35.16504/35)).
    offset = 5.971840214030754e-1              # offset = deltaS*sfac.

    x2 = sfac*SA
    xs = np.sqrt(x2 + offset)
    ys = CT*0.025
    z = p*1e-4     

    specvol_return = ( v000 + xs*(v010 + xs*(v020 + xs*(v030 + xs*(v040 
        + xs*(v050 + v060*xs))))) + ys*(v100 + xs*(v110 + xs*(v120 
        + xs*(v130 + xs*(v140 + v150*xs)))) + ys*(v200 + xs*(v210 
        + xs*(v220 + xs*(v230 + v240*xs))) + ys*(v300 + xs*(v310 
        + xs*(v320 + v330*xs)) + ys*(v400 + xs*(v410 + v420*xs) 
        + ys*(v500 + v510*xs + v600*ys))))) + z*(v001 + xs*(v011 
        + xs*(v021 + xs*(v031 + xs*(v041 + v051*xs)))) + ys*(v101 
        + xs*(v111 + xs*(v121 + xs*(v131 + v141*xs))) + ys*(v201 
        + xs*(v211 + xs*(v221 + v231*xs)) + ys*(v301 + xs*(v311 
        + v321*xs) + ys*(v401 + v411*xs + v501*ys)))) + z*(v002 
        + xs*(v012 + xs*(v022 + xs*(v032 + v042*xs))) + ys*(v102 
        + xs*(v112 + xs*(v122 + v132*xs)) + ys*(v202 + xs*(v212 
        + v222*xs) + ys*(v302 + v312*xs + v402*ys))) + z*(v003 
        + xs*(v013 + v023*xs) + ys*(v103 + v113*xs + v203*ys) 
        + z*(v004 + v014*xs + v104*ys + z*(v005 + v006*z))))) )

    return(specvol_return) 


# In[33]:




# In[34]:

@match_args_return
def specvol_alpha_beta(SA, CT, p):
    """
    gsw.specvol_alpha_beta   specific volume, thermal expansion & saline   
                              contraction coefficient (76-term equation)  
    ====================================================================

    USAGE:  
    [specvol, alpha, beta] = gsw.specvol_alpha_beta(SA,CT,p)

    DESCRIPTION:
    Calculates specific volume, the appropiate thermal expansion 
    coefficient and the appropriate saline contraction coefficient of
    seawater from Absolute Salinity and Conservative Temperature.  This
    function uses the computationally-efficient expression for specific
    volume in terms of SA, CT and p (Roquet et al., 2014).

    Note that the 76-term equation has been fitted in a restricted range
    of parameter space, and is most accurate inside the "oceanographic 
    funnel" described in IOC et al. (2010).  The GSW library function 
    "gsw.infunnel(SA,CT,p)" is avaialble to be used if one wants to test 
    if some of one's data lies outside this "funnel".  

    INPUT:
    SA  =  Absolute Salinity                                    [ g/kg ]
    CT  =  Conservative Temperature (ITS-90)                   [ deg C ]
    p   =  sea pressure                                         [ dbar ]
           ( i.e. absolute pressure - 10.1325 dbar )

    SA & CT need to have the same dimensions.
    p may have dimensions 1x1 or Mx1 or 1xN or MxN, where SA & CT are 
    MxN.

    OUTPUT:
    specvol =  specific volume                                  [ m/kg ]
    alpha   =  thermal expansion coefficient                     [ 1/K ]
                with respect to Conservative Temperature
    beta    =  saline (i.e. haline) contraction                 [ kg/g ]
                coefficient at constant Conservative Temperature

    AUTHOR: 
    Paul Barker and Trevor McDougall                [ help@teos-10.org ]

    VERSION NUMBER: 3.05 (27th November, 2015)

    REFERENCES:
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation 
    of seawater - 2010: Calculation and use of thermodynamic properties.  
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 
    56, UNESCO (English), 196 pp.  Available from http://www.TEOS-10.org
    See appendix A.20 and appendix K of this TEOS-10 Manual. 

    Roquet, F., G. Madec, T.J. McDougall and P.M. Barker, 2014: Accurate  
    polynomial expressions for the density and specifc volume of 
    seawater using the TEOS-10 standard. Ocean Modelling.
    """
    # This line ensures that SA is non-negative.
    SA = np.maximum(SA, 0)

    # deltaS = 24;
    sfac = 0.0248826675584615             # sfac = 1/(40*(35.16504/35)).
    offset = 5.971840214030754e-1                # offset = deltaS*sfac.  

    x2 = sfac*SA
    xs = np.sqrt(x2 + offset)
    ys = CT*0.025
    z = p*1e-4

    specvol_return = ( v000 + xs*(v010 + xs*(v020 + xs*(v030 + xs*(v040 
        + xs*(v050 + v060*xs))))) + ys*(v100 + xs*(v110 + xs*(v120 
        + xs*(v130 + xs*(v140 + v150*xs)))) + ys*(v200 + xs*(v210 
        + xs*(v220 + xs*(v230 + v240*xs)))  + ys*(v300 + xs*(v310 
        + xs*(v320 + v330*xs)) + ys*(v400 + xs*(v410 + v420*xs) 
        + ys*(v500 + v510*xs + v600*ys))))) + z*(v001 + xs*(v011 
        + xs*(v021 + xs*(v031 + xs*(v041 + v051*xs)))) + ys*(v101 
        + xs*(v111 + xs*(v121 + xs*(v131 + v141*xs))) + ys*(v201 
        + xs*(v211 + xs*(v221 + v231*xs)) + ys*(v301 + xs*(v311 
        + v321*xs) + ys*(v401 + v411*xs + v501*ys)))) + z*(v002
        + xs*(v012 + xs*(v022 + xs*(v032 + v042*xs))) + ys*(v102 
        + xs*(v112 + xs*(v122 + v132*xs)) + ys*(v202 + xs*(v212 
        + v222*xs) + ys*(v302 + v312*xs + v402*ys))) + z*(v003 
        + xs*(v013 + v023*xs) + ys*(v103 + v113*xs + v203*ys) 
        + z*(v004 + v014*xs + v104*ys + z*(v005 + v006*z))))) )
    
    v_CT = ( a000 + xs*(a100 + xs*(a200 + xs*(a300 + xs*(a400 
        + a500*xs)))) + ys*(a010 + xs*(a110 + xs*(a210 + xs*(a310 
        + a410*xs))) + ys*(a020 + xs*(a120 + xs*(a220 + a320*xs)) 
        + ys*(a030 + xs*(a130 + a230*xs) + ys*(a040 + a140*xs 
        + a050*ys )))) + z*(a001 + xs*(a101 + xs*(a201 + xs*(a301 
        + a401*xs))) + ys*(a011 + xs*(a111 + xs*(a211 + a311*xs)) 
        + ys*(a021 + xs*(a121 + a221*xs) + ys*(a031 + a131*xs 
        + a041*ys))) + z*(a002 + xs*(a102 + xs*(a202 + a302*xs)) 
        + ys*(a012 + xs*(a112 + a212*xs) + ys*(a022 + a122*xs 
        + a032*ys)) + z*(a003 + a103*xs + a013*ys + a004*z))) )

    alpha_return = 0.025*v_CT/specvol_return

    v_SA_part = ( b000 + xs*(b100 + xs*(b200 + xs*(b300 + xs*(b400 
        + b500*xs)))) + ys*(b010 + xs*(b110 + xs*(b210 + xs*(b310 
        + b410*xs))) + ys*(b020 + xs*(b120 + xs*(b220 + b320*xs)) 
        + ys*(b030 + xs*(b130 + b230*xs) + ys*(b040 + b140*xs 
        + b050*ys)))) + z*(b001 + xs*(b101 + xs*(b201 + xs*(b301 
        + b401*xs))) + ys*(b011 + xs*(b111 + xs*(b211 + b311*xs)) 
        + ys*(b021 + xs*(b121 + b221*xs) + ys*(b031 + b131*xs 
        + b041*ys))) + z*(b002 + xs*(b102 + xs*(b202 + b302*xs)) 
        + ys*(b012 + xs*(b112 + b212*xs) + ys*(b022 + b122*xs 
        + b032*ys)) + z*(b003 +  b103*xs + b013*ys + b004*z))) )
 
    beta_return = -v_SA_part*0.5*sfac/(specvol_return*xs)
    
    return(specvol_return, alpha_return, beta_return)


# In[34]:




# In[35]:

@match_args_return
def specvol_anom(SA,CT,p):

    """
    gsw.specvol_anom          specific volume anomaly (76-term equation)  
    ====================================================================

    USAGE:  
    specvol_anom = gsw.specvol_anom(SA,CT,p)

    DESCRIPTION:
    Calculates specific volume anomaly from Absolute Salinity, 
    Conservative Temperature and pressure. It uses the computationally-
    efficient expression for specific volume as a function of SA, CT and
    p (Roquet et al., 2014).  The reference value of Absolute Salinity 
    is SSO and the reference value of Conservative Temperature is equal
    to 0 degress C. 

    Note that the 76-term equation has been fitted in a restricted range
    of parameter space, and is most accurate inside the "oceanographic
    funnel" described in IOC et al. (2010).  The GSW library function 
    "gsw.infunnel(SA,CT,p)" is avaialble to be used if one wants to test
    if some of one's data lies outside this "funnel".  

    INPUT:
    SA  =  Absolute Salinity                                    [ g/kg ]  
    CT  =  Conservative Temperature (ITS-90)                   [ deg C ]
    p   =  sea pressure                                         [ dbar ]
           ( i.e. absolute pressure - 10.1325 dbar )

    SA & CT need to have the same dimensions.
    p may have dimensions 1x1 or Mx1 or 1xN or MxN, where SA & CT are 
    MxN.

    OUTPUT:
    specvol_anom  =  specific volume anomaly                  [ m^3/kg ]

    AUTHOR: 
    Paul Barker and Trevor McDougall                [ help@teos-10.org ]

    VERSION NUMBER: 3.04 (10th December, 2013)

    REFERENCES:
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation 
    of seawater - 2010: Calculation and use of thermodynamic properties.  
    Intergovernmental Oceanographic Commission, Manuals and Guides No.
    56,UNESCO (English), 196 pp.  Available from http://www.TEOS-10.org
    See Eqn. (3.7.3) of this TEOS-10 Manual. 

    F. Roquet, G. Madec, T.J. McDougall, P.M. Barker, 2014: Accurate
    polynomial expressions for the density and specifc volume of 
    seawater using the TEOS-10 standard. Ocean Modelling.
    """
    # This line ensures that SA is non-negative.
    SA = np.maximum(SA, 0)

    # deltaS = 24;
    sfac = 0.0248826675584615             # sfac = 1/(40*(35.16504/35)).  
    offset = 5.971840214030754e-1                # offset = deltaS*sfac.

    x2 = sfac*SA
    xs = np.sqrt(x2 + offset)
    ys = CT*0.025
    z = p*1e-4
    
    v = ( v000 + xs*(v010 + xs*(v020 + xs*(v030 + xs*(v040 
        + xs*(v050 + v060*xs))))) + ys*(v100 + xs*(v110 + xs*(v120 
        + xs*(v130 + xs*(v140 + v150*xs)))) + ys*(v200 + xs*(v210 
        + xs*(v220 + xs*(v230 + v240*xs)))  + ys*(v300 + xs*(v310 
        + xs*(v320 + v330*xs)) + ys*(v400 + xs*(v410 + v420*xs) 
        + ys*(v500 + v510*xs + v600*ys))))) + z*(v001 + xs*(v011 
        + xs*(v021 + xs*(v031 + xs*(v041 + v051*xs)))) + ys*(v101 
        + xs*(v111 + xs*(v121 + xs*(v131 + v141*xs))) + ys*(v201 
        + xs*(v211 + xs*(v221 + v231*xs)) + ys*(v301 + xs*(v311 
        + v321*xs) + ys*(v401 + v411*xs + v501*ys)))) + z*(v002
        + xs*(v012 + xs*(v022 + xs*(v032 + v042*xs))) + ys*(v102 
        + xs*(v112 + xs*(v122 + v132*xs)) + ys*(v202 + xs*(v212 
        + v222*xs) + ys*(v302 + v312*xs + v402*ys))) + z*(v003 
        + xs*(v013 + v023*xs) + ys*(v103 + v113*xs + v203*ys) 
        + z*(v004 + v014*xs + v104*ys + z*(v005 + v006*z))))) )
    
    specvol_anom_return = v - specvol_SSO_0_p(p)
    
    return(specvol_anom_return)


# In[35]:




# In[36]:

@match_args_return
def specvol_first_derivatives(SA,CT,p):
    
    """
    gsw.specvol_first_derivatives                first order derivatives
                                   of specific volume (76-term equation)
    ====================================================================

    USAGE:
    [v_SA, v_CT, v_P] = gsw.specvol_first_derivatives(SA,CT,p)

    DESCRIPTION:
    Calculates the following three first-order derivatives of specific
    volume (v),
    (1) v_SA, first-order derivative with respect to Absolute Salinity 
        at constant CT & p.
    (2) v_CT, first-order derivative with respect to CT at 
        constant SA & p. 
    (3) v_P, first-order derivative with respect to P at constant SA 
        and CT. 

    Note that this function uses the using the computationally-efficient
    expression for specific volume (Roquet et al., 2014).  There is an 
    alternative to calling this function, namely 
    gsw.specvol_first_derivatives_CT_exact(SA,CT,p) which uses the full 
    Gibbs function (IOC et al., 2010).   

    INPUT:
    SA  =  Absolute Salinity                                    [ g/kg ]
    CT  =  Conservative Temperature (ITS-90)                   [ deg C ]
    p   =  sea pressure                                         [ dbar ]
           ( i.e. absolute pressure - 10.1325 dbar )

    SA & CT need to have the same dimensions.
    p may have dimensions 1x1 or Mx1 or 1xN or MxN, where SA & CT are 
    MxN.

    OUTPUT:
    v_SA  =  The first derivative of specific volume with respect to 
             Absolute Salinity at constant CT & p.   [ J/(kg (g/kg)^2) ]
    v_CT  =  The first derivative of specific volume with respect to 
             CT at constant SA and p.                 [ J/(kg K(g/kg)) ]
    v_P   =  The first derivative of specific volume with respect to 
             P at constant SA and CT.                     [ J/(kg K^2) ]

    AUTHOR:   
    Trevor McDougall and Paul Barker.               [ help@teos-10.org ]

    VERSION NUMBER: 3.04 (10th December, 2013)

    REFERENCES:
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation 
    of seawater - 2010: Calculation and use of thermodynamic properties.  
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 
    56,UNESCO (English), 196 pp.  Available from http://www.TEOS-10.org.   

    Roquet, F., G. Madec, T.J. McDougall, P.M. Barker, 2014: Accurate
    polynomial expressions for the density and specifc volume of seawater
    using the TEOS-10 standard. Ocean Modelling.
    """
    # This line ensures that SA is non-negative.
    SA = np.maximum(SA, 0)

    # deltaS = 24
    sfac = 0.0248826675584615             # sfac = 1/(40*(35.16504/35)).
    offset = 5.971840214030754e-1                # offset = deltaS*sfac.     

    x2 = sfac*SA
    xs = np.sqrt(x2 + offset)
    ys = CT*0.025
    z = p*1e-4
    
    v_CT_part = ( a000 + xs*(a100 + xs*(a200 + xs*(a300 + xs*(a400 
        + a500*xs)))) + ys*(a010 + xs*(a110 + xs*(a210 + xs*(a310 
        + a410*xs))) + ys*(a020 + xs*(a120 + xs*(a220 + a320*xs)) 
        + ys*(a030 + xs*(a130 + a230*xs) + ys*(a040 + a140*xs 
        + a050*ys )))) + z*(a001 + xs*(a101 + xs*(a201 + xs*(a301 
        + a401*xs))) + ys*(a011 + xs*(a111 + xs*(a211 + a311*xs)) 
        + ys*(a021 + xs*(a121 + a221*xs) + ys*(a031 + a131*xs 
        + a041*ys))) + z*(a002 + xs*(a102 + xs*(a202 + a302*xs)) 
        + ys*(a012 + xs*(a112 + a212*xs) + ys*(a022 + a122*xs 
        + a032*ys)) + z*(a003 + a103*xs + a013*ys + a004*z))) )

    v_CT_return = 0.025*v_CT_part

    v_SA_part = ( b000 + xs*(b100 + xs*(b200 + xs*(b300 + xs*(b400 
        + b500*xs)))) + ys*(b010 + xs*(b110 + xs*(b210 + xs*(b310 
        + b410*xs))) + ys*(b020 + xs*(b120 + xs*(b220 + b320*xs)) 
        + ys*(b030 + xs*(b130 + b230*xs) + ys*(b040 + b140*xs 
        + b050*ys)))) + z*(b001 + xs*(b101 + xs*(b201 + xs*(b301 
        + b401*xs))) + ys*(b011 + xs*(b111 + xs*(b211 + b311*xs)) 
        + ys*(b021 + xs*(b121 + b221*xs) + ys*(b031 + b131*xs 
        + b041*ys))) + z*(b002 + xs*(b102 + xs*(b202 + b302*xs))
        + ys*(b012 + xs*(b112 + b212*xs) + ys*(b022 + b122*xs 
        + b032*ys)) + z*(b003 +  b103*xs + b013*ys + b004*z))) )
 
    v_SA_return = 0.5*sfac*v_SA_part/xs

    v_p_part = ( c000 + xs*(c100 + xs*(c200 + xs*(c300 + xs*(c400 
        + c500*xs))))  + ys*(c010 + xs*(c110 + xs*(c210 + xs*(c310 
        + c410*xs))) + ys*(c020 + xs*(c120 + xs*(c220 + c320*xs)) 
        + ys*(c030 + xs*(c130 + c230*xs) + ys*(c040 + c140*xs 
        + c050*ys)))) + z*(c001 + xs*(c101 + xs*(c201 + xs*(c301 
        + c401*xs))) + ys*(c011 + xs*(c111 + xs*(c211 + c311*xs)) 
        + ys*(c021 + xs*(c121 + c221*xs) + ys*(c031 + c131*xs 
        + c041*ys))) + z*( c002 + xs*(c102 + c202*xs) + ys*(c012 
        + c112*xs + c022*ys) + z*(c003 + c103*xs + c013*ys + z*(c004 
        + c005*z)))) )

    v_P_return = 1e-8*v_p_part

    return(v_SA_return, v_CT_return, v_P_return)


# In[36]:




# In[37]:

@match_args_return
def specvol_second_derivatives(SA,CT,p):
    
    """
    gsw.specvol_second_derivatives              second order derivatives 
                                   of specific volume (76-term equation)
    ====================================================================

    USAGE:
    [v_SA_SA, v_SA_CT, v_CT_CT, v_SA_P, v_CT_P] = ...
                                 gsw.specvol_second_derivatives(SA,CT,p)  

    DESCRIPTION:
    Calculates the following five second-order derivatives of specific
    volume (v),
    (1) v_SA_SA, second-order derivative with respect to Absolute 
        Salinity at constant CT & p.
    (2) v_SA_CT, second-order derivative with respect to SA & CT at 
        constant p. 
    (3) v_CT_CT, second-order derivative with respect to CT at constant 
        SA and p. 
    (4) v_SA_P, second-order derivative with respect to SA & P at 
        constant CT. 
    (5) v_CT_P, second-order derivative with respect to CT & P at 
        constant SA. 

    Note that this function uses the using the computationally-efficient
    expression for specific volume (Roquet et al., 2014).  There is an 
    alternative to calling this function, namely 
    gsw.specvol_second_derivatives_CT_exact(SA,CT,p) which uses the full   
    Gibbs function (IOC et al., 2010).   

    INPUT:
    SA  =  Absolute Salinity                                    [ g/kg ]
    CT  =  Conservative Temperature (ITS-90)                   [ deg C ]
    p   =  sea pressure                                         [ dbar ]
           ( i.e. absolute pressure - 10.1325 dbar )

    SA & CT need to have the same dimensions.
    p may have dimensions 1x1 or Mx1 or 1xN or MxN, where SA & CT are 
    MxN.

    OUTPUT:
    v_SA_SA  =  The second derivative of specific volume with respect to 
                Absolute Salinity at constant CT & p.[ J/(kg (g/kg)^2) ]   
    v_SA_CT  =  The second derivative of specific volume with respect to 
                SA and CT at constant p.              [ J/(kg K(g/kg)) ]
    v_CT_CT  =  The second derivative of specific volume with respect to 
                CT at constant SA and p.                  [ J/(kg K^2) ]
    v_SA_P  =  The second derivative of specific volume with respect to 
               SA and P at constant CT.               [ J/(kg K(g/kg)) ]
    v_CT_P  =  The second derivative of specific volume with respect to 
               CT and P at constant SA.               [ J/(kg K(g/kg)) ]

    AUTHOR:   
    Trevor McDougall and Paul Barker.               [ help@teos-10.org ]

    VERSION NUMBER: 3.04 (31st July, 2014)

    REFERENCES:
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation 
    of seawater - 2010: Calculation and use of thermodynamic properties.  
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 
    56,UNESCO (English), 196 pp.  Available from http://www.TEOS-10.org.  

    Roquet, F., G. Madec, T.J. McDougall, P.M. Barker, 2014: Accurate
    polynomial expressions for the density and specifc volume of seawater
    using the TEOS-10 standard. Ocean Modelling.
    """
    # This line ensures that SA is non-negative.
    SA = np.maximum(SA, 0)

    # deltaS = 24
    sfac = 0.0248826675584615             # sfac = 1/(40*(35.16504/35)).
    offset = 5.971840214030754e-1                # offset = deltaS*sfac.     
    z = p*1e-4
    
    x2 = sfac*SA
    xs2 = x2 + offset
    xs = np.sqrt(x2 + offset)
    ys = CT*0.025
    z = p*1e-4
    
    v_SA_SA_part = ( (-b000 + xs2*(b200 + xs*(2*b300 + xs*(3*b400 
        + 4*b500*xs))) + ys*(-b010 + xs2*(b210 + xs*(2*b310 
        + 3*b410*xs)) + ys*(-b020 + xs2*(b220 + 2*b320*xs) + ys*(-b030 
        + b230*xs2 + ys*(-b040 - b050*ys)))) + z*(-b001 + xs2*(b201 
        + xs*(2*b301 + 3*b401*xs)) + ys*(-b011 + xs2*(b211 + 2*b311*xs) 
        + ys*(-b021 + b221*xs2 + ys*(-b031 - b041*ys))) + z*(-b002 
        + xs2*(b202 + 2*b302*xs) + ys*(-b012 + b212*xs2 
        + ys*(-b022 - b032*ys)) + z*(-b003 - b013*ys - b004*z))))/xs2 )

    v_SA_SA_return = 0.25*sfac*sfac*v_SA_SA_part/xs

    v_SA_CT_part = ( (b010 + xs*(b110 + xs*(b210 + xs*(b310 + b410*xs))) 
        + ys*(2*(b020 + xs*(b120 + xs*(b220 + b320*xs))) + ys*(3*(b030
        + xs*(b130 + b230*xs)) + ys*(4*(b040 + b140*xs) + 5*b050*ys))) 
        + z*(b011 + xs*(b111 + xs*(b211 + b311*xs)) + ys*(2*(b021 
        + xs*(b121 + b221*xs)) + ys*(3*(b031 + b131*xs) + 4*b041*ys)) 
        + z*(b012 + xs*(b112 + b212*xs) + ys*(2*(b022 + b122*xs) 
        + 3*b032*ys) + b013*z)))/xs )

    v_SA_CT_return = 0.025*0.5*sfac*v_SA_CT_part

    v_CT_CT_part = ( a010 + xs*(a110 + xs*(a210 + xs*(a310 + a410*xs))) 
        + ys*(2*(a020 + xs*(a120 + xs*(a220 + a320*xs))) + ys*(3*(a030 
        + xs*(a130 + a230*xs)) + ys*(4*(a040 + a140*xs) + 5*a050*ys))) 
        + z*( a011 + xs*(a111 + xs*(a211 + a311*xs)) + ys*(2*(a021 
        + xs*(a121 + a221*xs)) + ys*(3*(a031 + a131*xs) + 4*a041*ys)) 
        + z*(a012 + xs*(a112 + a212*xs) + ys*(2*(a022 + a122*xs) 
        + 3*a032*ys) + a013*z)) )

    v_CT_CT_return = 0.025*0.025*v_CT_CT_part

    v_SA_P_part = ( b001 + xs*(b101 + xs*(b201 + xs*(b301 + b401*xs))) 
        + ys*(b011 + xs*(b111 + xs*(b211 + b311*xs)) + ys*(b021 
        + xs*(b121 + b221*xs) + ys*(b031 + b131*xs + b041*ys))) 
        + z*(2*(b002 + xs*(b102 + xs*(b202 + b302*xs)) + ys*(b012 
        + xs*(b112 + b212*xs) + ys*(b022 + b122*xs + b032*ys))) 
        + z*(3*(b003 + b103*xs + b013*ys) + 4*b004*z)) )

    v_SA_P_return = 1e-8*0.5*sfac*v_SA_P_part

    v_CT_P_part = ( a001 + xs*(a101 + xs*(a201 + xs*(a301 + a401*xs))) 
        + ys*(a011 + xs*(a111 + xs*(a211 + a311*xs)) + ys*(a021 
        + xs*(a121 + a221*xs) + ys*(a031 + a131*xs + a041*ys))) 
        + z*( 2*(a002 + xs*(a102 + xs*(a202 + a302*xs)) + ys*(a012 
        + xs*(a112 + a212*xs) + ys*(a022 + a122*xs + a032*ys))) 
        + z*(3*(a003 + a103*xs + a013*ys) + 4*a004*z)) )

    v_CT_P_return = 1e-8*0.025*v_CT_P_part

    return(v_SA_SA_return, v_SA_CT_return, v_CT_CT_return, v_SA_P_return, v_CT_P_return)


# In[37]:




# In[38]:

@match_args_return
def specvol_first_derivatives_wrt_enthalpy(SA,CT,p):

    """
    gsw.specvol_first_derivatives_wrt_enthalpy               first order 
                 derivatives of specific volume with respect to enthalpy  
    ====================================================================

    USAGE:
    [v_SA, v_h] = gsw.specvol_first_derivatives_wrt_enthalpy(SA,CT,p)

    DESCRIPTION:
    Calculates the following two first-order derivatives of specific
    volume (v),
    (1) v_SA, first-order derivative with respect to Absolute Salinity 
        at constant h & p.
    (2) v_h, first-order derivative with respect to h at 
        constant SA & p. 

    Note that this function uses the using the computationally-efficient  
    expression for specific volume (Roquet et al., 2014).  There is an 
    alternative to calling this function, namely 
    gsw.specvol_first_derivatives_wrt_enthalpy_CT_exact(SA,CT,p) which 
    uses the full Gibbs function (IOC et al., 2010).   

    INPUT:
    SA  =  Absolute Salinity                                    [ g/kg ]
    CT  =  Conservative Temperature (ITS-90)                   [ deg C ]
    p   =  sea pressure                                         [ dbar ]
           ( i.e. absolute pressure - 10.1325 dbar )

    SA & CT need to have the same dimensions.
    p may have dimensions 1x1 or Mx1 or 1xN or MxN, where SA & CT are 
    MxN.

    OUTPUT:
    v_SA  =  The first derivative of specific volume with respect to 
             Absolute Salinity at constant CT & p.   [ J/(kg (g/kg)^2) ]
    v_h  =  The first derivative of specific volume with respect to 
            SA and CT at constant p.                  [ J/(kg K(g/kg)) ]

    AUTHOR:   
    Trevor McDougall and Paul Barker.               [ help@teos-10.org ]

    VERSION NUMBER: 3.04 (10th December, 2013)

    REFERENCES:
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation 
    of seawater - 2010: Calculation and use of thermodynamic properties.  
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 
    56,UNESCO (English), 196 pp.  Available from http://www.TEOS-10.org.  

    F. Roquet, G. Madec, T.J. McDougall, P.M. Barker, 2014: Accurate
    polynomial expressions for the density and specifc volume of 
    seawater using the TEOS-10 standard. Ocean Modelling.
    """
    # This line ensures that SA is non-negative.
    SA = np.maximum(SA, 0)

    [vCT_SA, vCT_CT, dummy] = specvol_first_derivatives(SA,CT,p)
    [h_SA, h_CT] = enthalpy_first_derivatives(SA,CT,p)

    rec_h_CT = 1/h_CT;

    v_SA_return = vCT_SA - (vCT_CT*h_SA)*rec_h_CT
    v_h_return = vCT_CT*rec_h_CT
    
    return(v_SA_return, v_h_return)


# In[38]:




# In[39]:

@match_args_return
def specvol_second_derivatives_wrt_enthalpy(SA,CT,p):

    """
    gsw.specvol_second_derivatives_wrt_enthalpy             second order 
                 derivatives of volume specific with respect to enthalpy
    ====================================================================

    USAGE:
    [v_SA_SA, v_SA_h, v_h_h] = ...
                  gsw.specvol_second_derivatives_wrt_enthalpy(SA,CT,p)

    DESCRIPTION:
    Calculates the following three first-order derivatives of specific
    volume (v) with respect to enthalpy,
    (1) v_SA_SA, second-order derivative with respect to Absolute 
        Salinity at constant h & p.
    (2) v_SA_h, second-order derivative with respect to SA & h at 
        constant p. 
    (3) v_h_h, second-order derivative with respect to h at 
        constant SA & p. 

    Note that this function uses the using the computationally-efficient
    expression for specific volume (Roquet et al., 2014).  There is an 
    alternative to calling this function, namely 
    gsw.specvol_second_derivatives_wrt_enthalpy_CT_exact(SA,CT,p) which
    uses the full Gibbs function (IOC et al., 2010).   

    INPUT:
    SA  =  Absolute Salinity                                    [ g/kg ]
    CT  =  Conservative Temperature (ITS-90)                   [ deg C ]
    p   =  sea pressure                                         [ dbar ]
           ( i.e. absolute pressure - 10.1325 dbar )

    SA & CT need to have the same dimensions.
    p may have dimensions 1x1 or Mx1 or 1xN or MxN, where SA & CT are 
    MxN.

    OUTPUT:
    v_SA_SA = The second-order derivative of specific volume with 
              respect to Absolute Salinity at constant h & p.  
                                                     [ J/(kg (g/kg)^2) ]  
    v_SA_h  = The second-order derivative of specific volume with 
              respect to SA and h at constant p.      [ J/(kg K(g/kg)) ]
    v_h_h   = The second-order derivative with respect to h at 
              constant SA & p.

    AUTHOR:   
    Trevor McDougall and Paul Barker.               [ help@teos-10.org ]

    VERSION NUMBER: 3.04 (10th December, 2013)

    REFERENCES:
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation 
    of seawater - 2010: Calculation and use of thermodynamic properties.  
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 
    56,UNESCO (English), 196 pp.  Available from http://www.TEOS-10.org.  

    F. Roquet, G. Madec, T.J. McDougall, P.M. Barker, 2014: Accurate
    polynomial expressions for the density and specifc volume of 
    seawater using the TEOS-10 standard. Ocean Modelling.
    """
    # This line ensures that SA is non-negative.
    SA = np.maximum(SA, 0)

    [dummy, v_CT, dummy] = specvol_first_derivatives(SA,CT,p)
    [h_SA, h_CT] = enthalpy_first_derivatives(SA,CT,p)
    [vCT_SA_SA, vCT_SA_CT, vCT_CT_CT, dummy, dummy] = specvol_second_derivatives(SA,CT,p)
    [h_SA_SA, h_SA_CT, h_CT_CT] = enthalpy_second_derivatives(SA,CT,p)

    rec_h_CT = 1/h_CT
    rec_h_CT2 = rec_h_CT**2

    v_h_h_return = (vCT_CT_CT*h_CT - h_CT_CT*v_CT)*(rec_h_CT2*rec_h_CT)
    v_SA_h_return = (vCT_SA_CT*h_CT 
                    - v_CT*h_SA_CT)*rec_h_CT2 - h_SA*v_h_h_return
    v_SA_SA_return = ( vCT_SA_SA - (h_CT*(vCT_SA_CT*h_SA - v_CT*h_SA_SA) 
                    + v_CT*h_SA*h_SA_CT)*rec_h_CT2 - h_SA*v_SA_h_return )

    return(v_SA_SA_return, v_SA_h_return, v_h_h_return)


# In[39]:




# In[40]:

@match_args_return
def specvol_SSO_0_p(p):

    """
    gsw.specvol_SSO_0_p                  specific volume at (SSO,CT=0,p)  
                                                      (76-term equation)
    ====================================================================
    This function calculates specifc volume at the Standard Ocean 
    Salinity, SSO, and at a Conservative Temperature of zero degrees C, 
    as a function of pressure, p, in dbar, using a streamlined version 
    of the 76-term CT version of specific volume, that is, a streamlined
    version of the code "gsw_specvol(SA,CT,p)".

    VERSION NUMBER: 3.04 (10th December, 2013)

    REFERENCES:
    F. Roquet, G. Madec, T.J. McDougall, P.M. Barker, 2014: Accurate
    polynomial expressions for the density and specifc volume of seawater
    using the TEOS-10 standard. Ocean Modelling.
    """
    z = p*1e-4

    v005 = -1.2647261286e-8 
    v006 =  1.9613503930e-9 
        
    specvol_SSO_0_p_return = ( 9.726613854843870e-04 
        + z*(-4.505913211160929e-05 + z*(7.130728965927127e-06 
        + z*(-6.657179479768312e-07 + z*(-2.994054447232880e-08 
        + z*(v005 + v006*z))))) )
    
    return(specvol_SSO_0_p_return)


# In[40]:




# In[41]:

@match_args_return
def spiciness0(SA,CT):
    
    """
    gsw.spiciness0                               spiciness at p = 0 dbar      
                                                      (76-term equation)
    ====================================================================  

    USAGE:  
    spiciness0 = gsw.spiciness0(SA,CT,p)

    DESCRIPTION:
    Calculates spiciness from Absolute Salinity and Conservative 
    Temperature at a pressure of 0 dbar, as described by McDougall 
    and Krzysik (2015).  This routine is based on the computationally-
    efficient expression for specific volume in terms of SA, CT and p 
    (Roquet et al., 2015).

    Note that the 76-term equation has been fitted in a restricted range
    of parameter space, and is most accurate inside the "oceanographic 
    funnel" described in IOC et al. (2010).  The GSW library function 
    "gsw.infunnel(SA,CT,p)" is avaialble to be used if one wants to test
    if some of one's data lies outside this "funnel".  

    INPUT:
    SA  =  Absolute Salinity                                    [ g/kg ] 
    CT  =  Conservative Temperature (ITS-90)                   [ deg C ]

    SA & CT need to have the same dimensions.

    OUTPUT:
    spiciness0  =  spiciness referenced to a pressure of 0 dbar 
                                                              [ kg/m^3 ]   

    AUTHOR: 
    Oliver Krzysik and Trevor McDougall             [ help@teos-10.org ]

    VERSION NUMBER: 3.05 (5th December, 2014)

    REFERENCES:
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation 
    of seawater - 2010: Calculation and use of thermodynamic properties.  
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 
    56, UNESCO (English), 196 pp.  Available from http://www.TEOS-10.org

    McDougall, T.J., and O.A. Krzysik, 2015: Spiciness. To be submitted 
    to the Journal of Marine Research.  

    Roquet, F., G. Madec, T.J. McDougall and P.M. Barker, 2015: Accurate
    polynomial expressions for the density and specifc volume of 
    seawater using the TEOS-10 standard. Ocean Modelling.
    """
    # This line ensures that SA is non-negative.
    SA = np.maximum(SA, 0)

    # deltaS = 24;
    sfac = 0.0248826675584615             # sfac = 1/(40*(35.16504/35)).
    offset = 5.971840214030754e-1                # offset = deltaS*sfac.  

    x2 = sfac*SA
    xs = np.sqrt(x2 + offset)
    ys = CT*0.025
    
    spiciness0_return = ( s001 + ys*(s002 + ys*(s003 + ys*(s004 + ys*(s005 
        + ys*(s006 + s007*ys))))) + xs*(s008 + ys*(s009 + ys*(s010 
        + ys*(s011 + ys*(s012 + ys*(s013 + s014*ys))))) + xs*(s015 
        + ys*(s016 + ys*(s017 + ys*(s018 + ys*(s019 + ys*(s020 
        + s021*ys))))) + xs*(s022 + ys*(s023 + ys*(s024 + ys*(s025 
        + ys*(s026 + ys*(s027 + s028*ys))))) + xs*(s029 + ys*(s030 
        + ys*(s031 + ys*(s032 + ys*(s033 + ys*(s034 + s035*ys))))) 
        + xs*(s036 + ys*(s037 + ys*(s038 + ys*(s039 + ys*(s040 
        + ys*(s041 + s042*ys))))) + xs*(s043 + ys*(s044 + ys*(s045 
        + ys*(s046 + ys*(s047 + ys*(s048 + s049*ys))))))))))) )

    return(spiciness0_return)


# In[41]:




# In[42]:

@match_args_return
def spiciness1(SA,CT):
    
    """
    gsw.spiciness1                            spiciness at p = 1000 dbar      
                                                      (76-term equation)
    ====================================================================  

    USAGE:  
    spiciness1 = gsw.spiciness1(SA,CT,p)

    DESCRIPTION:
    Calculates spiciness from Absolute Salinity and Conservative 
    Temperature at a pressure of 1000 dbar, as described by McDougall 
    and Krzysik (2015).  This routine is based on the computationally-
    efficient expression for specific volume in terms of SA, CT and p 
    (Roquet et al., 2015).

    Note that the 76-term equation has been fitted in a restricted range  
    of parameter space, and is most accurate inside the "oceanographic
    funnel" described in IOC et al. (2010).  The GSW library function 
    "gsw.infunnel(SA,CT,p)" is avaialble to be used if one wants to test
    if some of one's data lies outside this "funnel".  

    INPUT:
    SA  =  Absolute Salinity                                    [ g/kg ] 
    CT  =  Conservative Temperature (ITS-90)                   [ deg C ]

    SA & CT need to have the same dimensions.

    OUTPUT:
    spiciness1  =  spiciness referenced to a pressure of 1000 dbar 
                                                              [ kg/m^3 ]   

    AUTHOR: 
    Oliver Krzysik and Trevor McDougall             [ help@teos-10.org ]

    VERSION NUMBER: 3.05 (5th December, 2014)

    REFERENCES:
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation 
    of seawater - 2010: Calculation and use of thermodynamic properties.  
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 
    56, UNESCO (English), 196 pp.  Available from http://www.TEOS-10.org

    McDougall, T.J., and O.A. Krzysik, 2015: Spiciness. To be submitted 
    to the Journal of Marine Research.  

    Roquet, F., G. Madec, T.J. McDougall and P.M. Barker, 2015: Accurate
    polynomial expressions for the density and specifc volume of 
    seawater using the TEOS-10 standard. Ocean Modelling.
    """
    # This line ensures that SA is non-negative.
    SA = np.maximum(SA, 0)

    # deltaS = 24;
    sfac = 0.0248826675584615             # sfac = 1/(40*(35.16504/35)).
    offset = 5.971840214030754e-1                # offset = deltaS*sfac.  

    x2 = sfac*SA
    xs = np.sqrt(x2 + offset)
    ys = CT*0.025
    
    spiciness1_return = ( s101 + ys*(s102 + ys*(s103 + ys*(s104 + ys*(s105 
        + ys*(s106 + s107*ys))))) + xs*(s108 + ys*(s109 + ys*(s110 
        + ys*(s111 + ys*(s112 + ys*(s113 + s114*ys))))) + xs*(s115 
        + ys*(s116 + ys*(s117 + ys*(s118 + ys*(s119 + ys*(s120 
        + s121*ys))))) + xs*(s122 + ys*(s123 + ys*(s124 + ys*(s125 
        + ys*(s126 + ys*(s127 + s128*ys))))) + xs*(s129 + ys*(s130 
        + ys*(s131 + ys*(s132 + ys*(s133 + ys*(s134 + s135*ys))))) 
        + xs*(s136 + ys*(s137 + ys*(s138 + ys*(s139 + ys*(s140 
        + ys*(s141 + s142*ys))))) + xs*(s143 + ys*(s144 + ys*(s145 
        + ys*(s146 + ys*(s147 + ys*(s148 + s149*ys))))))))))) )

    return(spiciness1_return)


# In[42]:




# In[43]:

@match_args_return
def spiciness2(SA,CT):
    
    """
    gsw.spiciness2                           spiciness at p = 2000 dbar      
                                                      (76-term equation)
    ====================================================================  

    USAGE:  
    spiciness2 = gsw.spiciness2(SA,CT,p)

    DESCRIPTION:
    Calculates spiciness from Absolute Salinity and Conservative 
    Temperature at a pressure of 2000 dbar, as described by McDougall 
    and Krzysik (2015).  This routine is based on the computationally-
    efficient expression for specific volume in terms of SA, CT and p 
    (Roquet et al., 2015).

    Note that the 76-term equation has been fitted in a restricted range
    of parameter space, and is most accurate inside the "oceanographic 
    funnel" described in IOC et al. (2010).  The GSW library function 
    "gsw.infunnel(SA,CT,p)" is avaialble to be used if one wants to test 
    if some of one's data lies outside this "funnel".  

    INPUT:
    SA  =  Absolute Salinity                                    [ g/kg ] 
    CT  =  Conservative Temperature (ITS-90)                   [ deg C ]

    SA & CT need to have the same dimensions.

    OUTPUT:
    spiciness2  =  spiciness referenced to a pressure of 2000 dbar 
                                                              [ kg/m^3 ]   

    AUTHOR: 
    Oliver Krzysik and Trevor McDougall             [ help@teos-10.org ]

    VERSION NUMBER: 3.05 (5th December, 2014)

    REFERENCES:
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation 
    of seawater - 2010: Calculation and use of thermodynamic properties.  
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 
    56, UNESCO (English), 196 pp.  Available from http://www.TEOS-10.org

    McDougall, T.J., and O.A. Krzysik, 2015: Spiciness. To be submitted 
    to the Journal of Marine Research.  

    Roquet, F., G. Madec, T.J. McDougall and P.M. Barker, 2015: Accurate
    polynomial expressions for the density and specifc volume of 
    seawater using the TEOS-10 standard. Ocean Modelling.
    """
    # This line ensures that SA is non-negative.
    SA = np.maximum(SA, 0)

    # deltaS = 24;
    sfac = 0.0248826675584615             # sfac = 1/(40*(35.16504/35)).
    offset = 5.971840214030754e-1                # offset = deltaS*sfac.  

    x2 = sfac*SA
    xs = np.sqrt(x2 + offset)
    ys = CT*0.025
    
    spiciness2_return = ( s201 + ys*(s202 + ys*(s203 + ys*(s204 + ys*(s205 
        + ys*(s206 + s207*ys))))) + xs*(s208 + ys*(s209 + ys*(s210 
        + ys*(s211 + ys*(s212 + ys*(s213 + s214*ys))))) + xs*(s215 
        + ys*(s216 + ys*(s217 + ys*(s218 + ys*(s219 + ys*(s220 
        + s221*ys))))) + xs*(s222 + ys*(s223 + ys*(s224 + ys*(s225 
        + ys*(s226 + ys*(s227 + s228*ys))))) + xs*(s229 + ys*(s230 
        + ys*(s231 + ys*(s232 + ys*(s233 + ys*(s234 + s235*ys))))) 
        + xs*(s236 + ys*(s237 + ys*(s238 + ys*(s239 + ys*(s240 
        + ys*(s241 + s242*ys))))) + xs*(s243 + ys*(s244 + ys*(s245 
        + ys*(s246 + ys*(s247 + ys*(s248 + s249*ys))))))))))) )

    return(spiciness2_return)


# In[43]:




# In[44]:

@match_args_return
def thermobaric(SA,CT,p):

    """
    gsw.thermobaric           thermobaric coefficient (76-term equation)   
    ====================================================================

    USAGE:  
    thermobaric = gsw.thermobaric(SA, CT, p)

    DESCRIPTION:
    Calculates the thermobaric coefficient of seawater with respect to
    Conservative Temperature.  This routine is based on the 
    computationally-efficient expression for specific volume in terms of
    SA, CT and p (Roquet et al., 2014).

    Note that the 76-term equation has been fitted in a restricted range 
    of parameter space, and is most accurate inside the "oceanographic 
    funnel" described in IOC et al. (2010).  The GSW library function 
    "gsw.infunnel(SA,CT,p)" is avaialble to be used if one wants to test 
    if some of one's data lies outside this "funnel".  

    INPUT:
    SA  =  Absolute Salinity                                    [ g/kg ]
    CT  =  Conservative Temperature (ITS-90)                   [ deg C ]
    p   =  sea pressure                                         [ dbar ]
           ( i.e. absolute pressure - 10.1325 dbar )

    SA & CT need to have the same dimensions.
    p may have dimensions 1x1 or Mx1 or 1xN or MxN, where SA & CT are 
    MxN.

    OUTPUT:
    thermobaric  =  thermobaric coefficient with            [ 1/(K Pa) ] 
                    respect to Conservative Temperature.           
    Note. The pressure derivative is taken with respect to
    pressure in Pa not dbar.

    AUTHOR: 
    Trevor McDougall and Paul Barker                [ help@teos-10.org ]

    VERSION NUMBER: 3.04 (10th December, 2013)

    REFERENCES:
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation 
    of seawater - 2010: Calculation and use of thermodynamic properties.  
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 
    56, UNESCO (English), 196 pp.  Available from http://www.TEOS-10.org  
    See Eqns. (3.8.2) and (P.2) of this TEOS-10 manual.

    Roquet, F., G. Madec, T.J. McDougall, P.M. Barker, 2014: Accurate
    polynomial expressions for the density and specifc volume of seawater
    using the TEOS-10 standard. Ocean Modelling.
    """
    # This line ensures that SA is non-negative.
    SA = np.maximum(SA, 0)

    rho_temp = rho(SA,CT,p)

    [v_SA, v_CT, dummy] = specvol_first_derivatives(SA,CT,p)
    [dummy, dummy, dummy, v_SA_P, v_CT_P] = specvol_second_derivatives(SA,CT,p)
    thermobaric_return = rho_temp*(v_CT_P - (v_CT/v_SA)*v_SA_P)
 
    return(thermobaric_return)

