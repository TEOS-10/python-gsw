# -*- coding: utf-8 -*-

"""Unit check for standard profiles for the Gibbs Sea Water python package."""

# Auto generates and perform a set of test methods like:
#
#    def test_funct(self):
#        out = gsw.func(arg1_chck_cast, arg2_chck_cast, ...)
#        maxdiff = np.nanmax(abs(out - cv.func)
#        self.assertTrue(maxdiff < cv.func_ca)
#
# cv is a Dict2Struc instance with all the check values from
# the profile-file
# The func, args are taken from the main dictionary below
# giving a alphabetical table of all functions and their arguments
#
# Extra aliasing attributes are added to cv if the naming
# convention is broken, if the targets or the attributes
# in the check file has "wrong" names
#


# Bjørn Ådlandsvik <bjorn@imr.no>
# 2011-03-03

import os
import unittest
import functools  # Requires python 2.5.
import numpy as np

import gsw
from gsw.utilities import Dict2Struc


# Read data file with check value profiles.
datadir = os.path.join(os.path.dirname(gsw.utilities.__file__), 'data')
cv = Dict2Struc(np.load(os.path.join(datadir, 'gsw_cv_v3_0.npz')))
cf = Dict2Struc(np.load(os.path.join(datadir, 'gsw_cf.npz')))

# Main dictionary of functions with arguments. Could perhaps be auto-generated.
function_arguments = dict(
    # absolute_salinity_sstar_ct.py
    #SA_from_SP=('SP', 'p', 'long', 'lat'), BUG on SAAR
    #Sstar_from_SP  TODO
    CT_from_t=('SA', 't', 'p'),
    #
    # basic_thermodynamic_t.py
    rho_t_exact=('SA', 't', 'p'),
    pot_rho_t_exact=('SA', 't', 'p', 'pr'),
    sigma0_pt0_exact=('SA', 'pt0'),
    alpha_wrt_CT_t_exact=('SA', 't', 'p'),
    alpha_wrt_pt_t_exact=('SA', 't', 'p'),
    alpha_wrt_t_exact=('SA', 't', 'p'),
    beta_const_CT_t_exact=('SA', 't', 'p'),
    beta_const_pt_t_exact=('SA', 't', 'p'),
    beta_const_t_exact=('SA', 't', 'p'),
    specvol_t_exact=('SA', 't', 'p'),
    specvol_anom_t_exact=('SA', 't', 'p'),
    sound_speed_t_exact=('SA', 't', 'p'),
    kappa_t_exact=('SA', 't', 'p'),
    kappa_const_t_exact=('SA', 't', 'p'),
    internal_energy_t_exact=('SA', 't', 'p'),
    enthalpy_t_exact=('SA', 't', 'p'),
    dynamic_enthalpy_t_exact=('SA', 't', 'p'),
    SA_from_rho_t_exact=('rho', 't', 'p'),
    #t_from_rho_exact=('rho', 'SA', 'p'),
    t_maxdensity_exact=('SA', 'p'),
    entropy_t_exact=('SA', 't', 'p'),
    cp_t_exact=('SA', 't', 'p'),
    isochoric_heat_cap_t_exact=('SA', 't', 'p'),
    chem_potential_relative_t_exact=('SA', 't', 'p'),
    chem_potential_water_t_exact=('SA', 't', 'p'),
    chem_potential_salt_t_exact=('SA', 't', 'p'),
    Helmholtz_energy_t_exact=('SA', 't', 'p'),
    adiabatic_lapse_rate_t_exact=('SA', 't', 'p'),
    osmotic_coefficient_t_exact=('SA', 't', 'p'),
    osmotic_pressure_t_exact=('SA', 't', 'p'),
    #
    # conversion.py
    #deltaSA_from_SP  TODO
    #SA_Sstar_from_SP  TODO
    SR_from_SP=('SP',),
    SP_from_SR=('SR',),
    #SP_from_SA=('SA', 'p', 'long', 'lat'),  TODO
    #Sstar_from_SA=('SA', 'p', 'long', 'lat'),  TODO
    #SA_from_Sstar=('Sstar', 'p', 'long', 'lat'), TODO
    #SP_from_Sstar=('Sstar', 'p', 'long', 'lat'),  TODO
    pt_from_CT=('SA', 'CT'),
    t_from_CT=('SA', 'CT', 'p'),
    CT_from_pt=('SA', 'pt'),
    pot_enthalpy_from_pt=('SA', 'pt'),
    pt0_from_t=('SA', 't', 'p'),
    pt_from_t=('SA', 't', 'p', 'pr'),
    t90_from_t48=('t',),
    t90_from_t68=('t',),
    z_from_p=('p', 'lat'),
    p_from_z=('z', 'lat'),
    depth_from_z=('z'),
    z_from_depth=('depth',),
    Abs_Pressure_from_p=('p',),
    p_from_Abs_Pressure=('Abs_Pressure_from_p',),
    entropy_from_CT=('SA', 'CT'),
    CT_from_entropy=('SA', 'entropy'),
    entropy_from_pt=('SA', 'pt'),
    pt_from_entropy=('SA', 'entropy'),
    molality_from_SA=('SA',),
    ionic_strength_from_SA=('SA',),
    #
    # density_enthalpy_48_ct.py
    #rho_CT  TODO
    #alpha_CT  TODO
    #beta_CT  TODO
    #rho_alpha_beta_CT  TODO
    #specvol_CT  TODO
    #specvol_anom_CT  TODO
    #sigma0_CT  TODO
    #sigma1_CT  TODO
    #sigma2_CT  TODO
    #sigma3_CT  TODO
    #sigma4_CT  TODO
    #sound_speed_CT  TODO
    #internal_energy_CT  TODO
    #enthalpy_CT  TODO
    #enthalpy_diff_CT  TODO
    #dynamic_enthalpy_CT  TODO
    #SA_from_rho_CT  TODO
    #CT_from_rho  TODO
    #CT_maxdensity TODO
    #
    # density_enthalpy_48.py NOTE: None are tested on Matlab.
    rho=('SA', 'CT', 'p'),
    alpha=('SA', 'CT', 'p'),
    beta=('SA', 'CT', 'p'),
    rho_alpha_beta=('SA', 'CT', 'p'),
    specvol=('SA', 'CT', 'p'),
    specvol_anom=('SA', 'CT', 'p'),
    sigma0=('SA', 'CT'),
    sigma1=('SA', 'CT'),
    sigma2=('SA', 'CT'),
    sigma3=('SA', 'CT'),
    sigma4=('SA', 'CT'),
    sound_speed=('SA', 'CT', 'p'),
    internal_energy=('SA', 'CT', 'p'),
    enthalpy=('SA', 'CT', 'p'),
    enthalpy_diff=('SA', 'CT', 'p_shallow', 'p_deep'),
    dynamic_enthalpy=('SA', 'CT', 'p'),
    SA_from_rho=('rho_cf', 'CT', 'p'),
    #
    # density_enthalpy_ct_exact.py
    rho_CT_exact=('SA', 'CT', 'p'),
    alpha_CT_exact=('SA', 'CT', 'p'),
    beta_CT_exact=('SA', 'CT', 'p'),
    rho_alpha_beta_CT_exact=('SA', 'CT', 'p'),
    specvol_CT_exact=('SA', 'CT', 'p'),
    specvol_anom_CT_exact=('SA', 'CT', 'p'),
    sigma0_CT_exact=('SA', 'CT'),
    sigma1_CT_exact=('SA', 'CT'),
    sigma2_CT_exact=('SA', 'CT'),
    sigma3_CT_exact=('SA', 'CT'),
    sigma4_CT_exact=('SA', 'CT'),
    sound_speed_CT_exact=('SA', 'CT', 'p'),
    internal_energy_CT_exact=('SA', 'CT', 'p'),
    enthalpy_CT_exact=('SA', 'CT', 'p'),
    enthalpy_diff_CT_exact=('SA', 'CT', 'p_shallow', 'p_deep'),
    dynamic_enthalpy_CT_exact=('SA', 'CT', 'p'),
    SA_from_rho_CT_exact=('rho', 'CT', 'p'),
    # FIXME: NameError: 't_from_rho_exact' not defined
    #CT_from_rho_exact=('rho', 'SA', 'p'),
    CT_maxdensity_exact=('SA', 'p'),
    #
    # derivatives.py
    #CT_first_derivatives=('SA', 'pt'),  FIXME: out should tuple.
    #CT_second_derivatives=('SA', 'pt'),  FIXME: out should tuple.
    #enthalpy_first_derivatives=('SA', 'CT', 'p'),  FIXME: out should tuple.
    #enthalpy_second_derivatives=('SA', 'CT', 'p'),  FIXME: out should tuple.
    #entropy_first_derivatives=('SA', 'CT'),  FIXME: out should tuple.
    #entropy_second_derivatives=('SA', 'CT'),  FIXME: out should tuple.
    #pt_first_derivatives=('SA', 'CT'),  FIXME: out should tuple.
    #pt_second_derivatives=('SA', 'CT'),  FIXME: out should tuple.
    #
    # earth.py
    f=('lat',),
    grav=('lat', 'p'),
    distance=('long', 'lat', 'p'),
    #
    # freezing.py
    # NOTE: The matlab test does not use saturation_fraction=1 which is the
    # default!  It uses saturation_fraction=0.
    CT_freezing=('SA', 'p', 'sat0'),
    t_freezing=('SA', 'p', 'sat0'),
    brineSA_CT=('CT_freezing', 'p', 'sat05'),
    brineSA_t=('t_freezing', 'p', 'sat05'),
    #
    # geostrophic.py
    #geostrophic_velocity  TODO
    #
    # geostrophic_48.py
    #geo_strf_dyn_height  TODO
    #geo_strf_dyn_height_pc  TODO
    #geo_strf_isopycnal  TODO
    #geof_str_isopycnal_pc  TODO
    #geo_strf_Montgomery  TODO
    #geo_strf_Cunningham  TODO
    #
    # isobaric.py
    #latentheat_melting  TODO
    #latentheat_evap_CT  TODO
    #latentheat_evap_t=('SA', 't'),
    #
    # library.py
    #gibbs
    #SAAR  TODO
    #Fdelta  TODO
    #delta_SA_ref=('p', 'long', 'lat'), TODO
    #SA_from_SP_Baltic=('SP', 'long', 'lat'),  NOTE: Not tested on Matlab.
    #SP_from_SA_Baltic=('SA', 'long', 'lat'),  NOTE: Not tested on Matlab.
    #infunnel=('SA', 'CT', 'p'),  NOTE: Not tested on Matlab.
    #entropy_part=('SA', 'CT', 'p'),  NOTE: Not tested on Matlab.
    #entropy_part_zerop=('SA', 'pt0'),  NOTE: Not tested on Matlab.
    #interp_ref_cast=('spycnl', 'gn'),  NOTE: Not tested on Matlab.
    #interp_SA_CT=('SA', 'CT', 'p', 'p_i'),  NOTE: Not tested on Matlab.
    #gibbs_pt0_pt0=('SA', 'pt0'),  NOTE: Not tested on Matlab.
    #specvol_SSO_0_p=('p'),  NOTE: Not tested on Matlab.
    #enthalpy_SSO_0_p=('p',),  FIXME: No enthalpy_SSO_0_p.
    #Hill_ratio_at_SP2= ('t'),  FIXME: No Hill_ratio_at_SP2.
    #
    #neutral_nonlinear_48.py
    #cabbeling,  TODO
    #thermobaric,  TODO
    #isopycnal_slope_ratio,  TODO
    #isopycnal_vs_ntp_CT_ratio,  TODO
    #ntp_pt_vs_CT_ratio  TODO
    #
    # practical_salinity.py
    SP_from_C=('C', 't', 'p'),
    C_from_SP=('SP', 't', 'p'),
    SP_from_R=('R_cf', 't', 'p'),
    R_from_SP=('SP', 't', 'p'),
    SP_salinometer=('Rt', 't'),
    #SP_from_SK=('SK',),  NOTE: Not tested on Matlab.
    #SK_from_SP=('SP',),  NOTE: Not tested on Matlab.
    #
    # steric.py
    #steric_height=TODO
    #
    # water_column_48.py
    Nsquared=('SA', 'CT', 'p', 'lat'),
    Turner_Rsubrho=('SA', 'CT', 'p'),
    IPV_vs_fNsquared_ratio=('SA', 'CT', 'p')
   )


# Make aliases for some values to be used as arguments
cv.entropy_chck_cast = cv.entropy_from_CT
cv.Abs_Pressure_from_p_chck_cast = cv.Abs_Pressure_from_p
cv.depth_chck_cast = cv.depth_from_z
cv.C_chck_cast = cv.C_from_SP
cv.pt_chck_cast = cv.pt_from_t
cv.z_chck_cast = cv.z_from_p
cv.SR_chck_cast = cv.SR_from_SP
cv.pr_chck_cast = cv.pr
cv.p_shallow_chck_cast = cv.p_chck_cast_shallow
cv.p_deep_chck_cast = cv.p_chck_cast_deep
cv.rho_chck_cast = cv.rho_CT_exact
cv.rho_CTrab_exact_ca = cv.rho_CT_exact_rab_ca
cv.CT_freezing_chck_cast = cv.CT_freezing
cv.t_freezing_chck_cast = cv.t_freezing
cv.sat0_chck_cast = 0
cv.sat05_chck_cast = 0.5

# Aliases from computed values.
cv.R_cf_chck_cast = cf.R
cv.rho_cf_chck_cast = cf.rho
cv.pt0_chck_cast = cf.pt0_from_t

# Functions and targets which does not follow the naming convention.
not_match = {
    'CT_first_derivatives': 'CT_SA',
    'CT_second_derivatives': 'CT_SA_SA',
    'enthalpy_first_derivatives': 'h_SA',
    'enthalpy_second_derivatives': 'h_SA_SA',
    'entropy_first_derivatives': 'eta_SA',
    'entropy_second_derivatives': 'eta_SA_SA',
    'pt_first_derivatives': 'pt_SA',
    'pt_second_derivatives': 'pt_SA_SA',
    'Turner_Rsubrho': 'Tu',
    'IPV_vs_fNsquared_ratio': 'IPVfN2',
    'Nsquared': 'n2',
    'chem_potential_relative_t_exact': 'chem_potential_t_exact',
    'rho_alpha_beta_CT_exact': 'rho_CTrab_exact',
    'rho_alpha_beta': 'rho_rab',
    }

# Add target aliases to cv.
for f in not_match:
    setattr(cv, f, getattr(cv, not_match[f]))
    setattr(cv, f + '_ca', getattr(cv, not_match[f] + '_ca'))


# Generic test method.
def generic_test(self, func=None, argnames=None):
    """Generic test function, to be specialized by functools.partial."""
    # Transform argument names to name convention in cv dataset.
    args = [getattr(cv, a + '_chck_cast') for a in argnames]
    # Perform the function call
    out = getattr(gsw, func)(*args)
    # FIXME: Testing just the first output!
    # Check that the maximal error is less than the given tolerance
    if isinstance(out, tuple):
        out = out[0]
        #print("""%s returns a tuple.""" % func)
    maxdiff = np.nanmax(abs(out - getattr(cv, func)))
    try:
        self.assertTrue(maxdiff < getattr(cv, func + '_ca'))
    except AssertionError, e:
        raise AssertionError("Error in %s %s" % (func, e.args))


# Dictionary of functions with corresponding test methods.
function_test = {}
for f in function_arguments:
    function_test[f] = functools.partial(generic_test,
                      func=f, argnames=function_arguments[f])


# Auto-generated TestCase.
class Test_profiles(unittest.TestCase):

    for f in function_test:
        method_def = ("test_" + f +
            " = lambda self: function_test['" + f + "'](self)")
        exec(method_def)


if __name__ == '__main__':
    unittest.main(verbosity=2)
