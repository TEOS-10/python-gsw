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
import sys
import unittest
import functools  # Requires python 2.5.
import numpy as np
import gsw
from gsw.utilities import Dict2Struc


# Read data file with check value profiles
datadir = os.path.join(os.path.dirname(gsw.utilities.__file__), 'data')
fname = 'gsw_cv_v3_0.npz'
cv = Dict2Struc(np.load(os.path.join(datadir, fname)))

# Main dictionary of functions with arguments
# Codes for non-tested functions
#
# TI: Returning a tuple, tested individually
# NV: No test value
# NA: Don't fit the testing scheme
# ERR: Gives error

# Could perhaps be auto-generated
function_arguments = {
    # absolute_salinity_sstar_ct.py
    #SA_from_SP TODO
    #Sstar_from_SP TODO
    'CT_from_t': ('SA', 't', 'p'),
    #
    # basic_thermodynamic_t.py
    'rho_t_exact': ('SA', 't', 'p'),
    'pot_rho_t_exact': ('SA', 't', 'p', 'pr'),
    #'sigma0_pt0_exact'  TODO
    'alpha_wrt_CT_t_exact': ('SA', 't', 'p'),
    'alpha_wrt_pt_t_exact': ('SA', 't', 'p'),
    'alpha_wrt_t_exact': ('SA', 't', 'p'),
    'beta_const_CT_t_exact': ('SA', 't', 'p'),
    'beta_const_pt_t_exact': ('SA', 't', 'p'),
    'beta_const_t_exact': ('SA', 't', 'p'),
    'specvol_t_exact': ('SA', 't', 'p'),
    'specvol_anom_t_exact': ('SA', 't', 'p'),
    'sound_speed_t_exact': ('SA', 't', 'p'),
    'kappa_t_exact': ('SA', 't', 'p'),
    'kappa_const_t_exact': ('SA', 't', 'p'),
    'internal_energy_t_exact': ('SA', 't', 'p'),
    'enthalpy_t_exact': ('SA', 't', 'p'),
    'dynamic_enthalpy_t_exact': ('SA', 't', 'p'),
    #'SA_from_rho_t_exact': ('rho', 't', 'p'), FIXME: No rho_chck_cast.
    #'t_from_rho_exact'  TODO
    't_maxdensity_exact': ('SA', 'p'),
    'entropy_t_exact': ('SA', 't', 'p'),
    'cp_t_exact': ('SA', 't', 'p'),
    'isochoric_heat_cap_t_exact': ('SA', 't', 'p'),
    #'chem_potential_relative_t_exact': ('SA', 't', 'p'), FIXME: No chem_potential_relative_t_exact.
    'chem_potential_water_t_exact': ('SA', 't', 'p'),
    'chem_potential_salt_t_exact': ('SA', 't', 'p'),
    'Helmholtz_energy_t_exact': ('SA', 't', 'p'),
    'adiabatic_lapse_rate_t_exact': ('SA', 't', 'p'),
    'osmotic_coefficient_t_exact': ('SA', 't', 'p'),
    #'osmotic_pressure_t_exact': ('SA', 't', 'pw'), FIXME: No pw_chck_cast.
    #
    # conversion.py
    #'deltaSA_from_SP'
    #'SA_Sstar_from_SP'
    #'SR_from_SP': ('SP'),  FIXME: No S_chck_cast
    #'SP_from_SR': ('SR'),  FIXME: No S_chck_cast
    #'SP_from_SA': ('SA', 'p', 'long', 'lat'),  TODO
    #'Sstar_from_SA': ('SA', 'p', 'long', 'lat'),  TODO
    #'SA_from_Sstar': ('Sstar', 'p', 'long', 'lat'), TODO
    #'SP_from_Sstar': ('Sstar', 'p', 'long', 'lat'),  TODO
    'pt_from_CT': ('SA', 'CT'),
    't_from_CT': ('SA', 'CT', 'p'),
    'CT_from_pt': ('SA', 'pt'),
    #'pot_enthalpy_from_pt': ('pot_enthalpy'), FIXME: No o_chck_cast.
    'pt0_from_t': ('SA', 't', 'p'),
    'pt_from_t': ('SA', 't', 'p', 'pr'),
    't90_from_t48': ('t',),
    't90_from_t68': ('t',),
    'z_from_p': ('p', 'lat'),
    #'p_from_z': ('z', 'lat'), BUG
    'depth_from_z': ('z'),
    #'z_from_depth': ('depth'), FIXME: No d_chck_cast.
    'Abs_Pressure_from_p': ('p'),
    #'p_from_Abs_Pressure': ('Absolute_Pressure'),  FIXME: No A_chck_cast.
    'entropy_from_CT': ('SA', 'CT'),
    #'CT_from_entropy': ('SA', 'entropy'),  FIXME: No entropy_chck_cast.
    'entropy_from_pt': ('SA', 'pt'),
    #'pt_from_entropy' : ('SA', 'entropy'), FIXME: No entropy_chck_cast.
    #'molality_from_SA': ('SA'),  FIXME: No S_chck_cast.
    #'ionic_strength_from_SA': ('SA'),  FIXME: No S_chck_cast
    #
    # density_enthalpy_48_ct.py TODO
    #'rho_CT',  TODO
    #'alpha_CT',  TODO
    #'beta_CT',  TODO
    #'rho_alpha_beta_CT',  TODO
    #'specvol_CT',  TODO
    #'specvol_anom_CT',  TODO
    #'sigma0_CT',  TODO
    #'sigma1_CT',  TODO
    #'sigma2_CT',  TODO
    #'sigma3_CT',  TODO
    #'sigma4_CT',  TODO
    #'sound_speed_CT',  TODO
    #'internal_energy_CT',  TODO
    #'enthalpy_CT',  TODO
    #'enthalpy_diff_CT',  TODO
    #'dynamic_enthalpy_CT',  TODO
    #'SA_from_rho_CT',  TODO
    #'CT_from_rho',  TODO
    #'CT_maxdensity', TODO
    #
    # density_enthalpy_48.py TODO
    'rho': ('SA', 'CT', 'p'),
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
    #
    # density_enthalpy_ct_exact.py
    'rho_CT_exact': ('SA', 'CT', 'p'),
    'alpha_CT_exact': ('SA', 'CT', 'p'),
    'beta_CT_exact': ('SA', 'CT', 'p'),
    #'rho_alpha_beta_CT_exact': ('SA', 'CT', 'p'),  TODO
    'specvol_CT_exact': ('SA', 'CT', 'p'),
    'specvol_anom_CT_exact': ('SA', 'CT', 'p'),
    #'sigma0_CT_exact': ('SA', 'CT'),  FIXME: No sigma0_pt0_exact.
    'sigma1_CT_exact': ('SA', 'CT'),
    'sigma2_CT_exact': ('SA', 'CT'),
    'sigma3_CT_exact': ('SA', 'CT'),
    'sigma4_CT_exact': ('SA', 'CT'),
    'sound_speed_CT_exact': ('SA', 'CT', 'p'),
    'internal_energy_CT_exact': ('SA', 'CT', 'p'),
    'enthalpy_CT_exact': ('SA', 'CT', 'p'),
    #'enthalpy_diff_CT_exact': ('SA', 'CT', 'p_shallow', 'p_deep'),  FIXME: No p_shallow_chck_cast
    'dynamic_enthalpy_CT_exact': ('SA', 'CT', 'p'),
    #'SA_from_rho_CT_exact': ('rho', 'CT', 'p'),  FIXME: No rho_chck_cast
    #'CT_from_rho_exact': ('rho', 'SA', 'p'),  FIXME: No rho_chck_cast
    'CT_maxdensity_exact': ('SA', 'p'),
    #
    # derivatives.py
    #'CT_first_derivatives': ('SA', 'pt'),  #NOTE: TI, FIXME name match
    #'CT_second_derivatives': ('SA', 'pt'),  #NOTE: TI FIXME name match
    #'enthalpy_first_derivatives': ('SA', 'CT', 'p'), FIXME name match
    #'enthalpy_second_derivatives': ('SA', 'CT', 'p'), FIXME name match
    #'entropy_first_derivatives': ('SA', 'CT'),  #NOTE: TI FIXME name match
    #'entropy_second_derivatives': ('SA', 'pt'),  #NOTE: TI FIXME name match
    #'pt_first_derivatives':,  #NOTE: TI FIXME name match
    #'pt_second_derivatives':  #NOTE: TI FIXME name match
    #
    # earth.py
    'f': ('lat',),
    'grav': ('lat', 'p'),
    'distance': ('long', 'lat', 'p'),
    #
    # freezing.py
    #'CT_freezing ',  TODO
    #'t_freezing',  TODO
    #'brineSA_CT',  TODO
    #'brineSA_t'  TODO
    #
    # geostrophic.py
    # 'geostrophic_velocity' TODO
    #
    # geostrophic_48.py
    #'geo_strf_dyn_height',  TODO
    #'geo_strf_dyn_height_pc',  TODO
    #'geo_strf_isopycnal',  TODO
    #'geof_str_isopycnal_pc',  TODO
    #'geo_strf_Montgomery',  TODO
    #'geo_strf_Cunningham'  TODO
    #
    # isobaric.py
    #'latentheat_melting':
    #'latentheat_evap_CT':
    #'latentheat_evap_t': ('SA', 't'),
    #
    # library.py
    #'gibbs'
    #'SAAR'  TODO
    #'Fdelta'  TODO
    #'delta_SA_ref': ('p', 'long', 'lat'), TODO
    #'SA_from_SP_Baltic': ('SP', 'long', 'lat'),  FIXME: No SA_from_SP_Baltic
    #'SP_from_SA_Baltic' : ('SA', 'long', 'lat'),  FIXME: No SP_from_SA_Baltic.
    #'infunnel': ('SA', 'CT', 'p'),  FIXME: No infunnel.
    #'entropy_part': ('SA', 'CT', 'p'),  FIXME: No entropy_part
    #'entropy_part_zerop': ('SA', 'pt0'),  FIXME: No pt0_chck_cast.
    #'interp_ref_cast': ('spycnl', 'gn'),  FIXME: No spycnl_chck_cast.
    #'interp_SA_CT': ('SA', 'CT', 'p', 'p_i'),  FIXME: No p_i_chck_cast.
    #'gibbs_pt0_pt0': ('SA', 'pt0'),  FIXME: No pt0_chck_cast.
    #'specvol_SSO_0_p': ('p'),  FIXME: No specvol_SSO_0_p.
    #'enthalpy_SSO_0_p': ('p',),  FIXME: No enthalpy_SSO_0_p.
    #'Hill_ratio_at_SP2':  ('t'),  FIXME: No Hill_ratio_at_SP2.
    #
    # neutral_nonlinear_48.py
    #'cabbeling',  TODO
    #'thermobaric',  TODO
    #'isopycnal_slope_ratio',  TODO
    #'isopycnal_vs_ntp_CT_ratio',  TODO
    #'ntp_pt_vs_CT_ratio'  TODO
    #
    # TODO below this line.
    #
    # practical_salinity.py
    'SP_from_C': ('C', 't', 'p'),
    #'C_from_SP': ('SP', 't', 'p'),  BUG: line 447 could not be broadcast shapes (45,3) (21)
    #'SP_from_R':  TODO
    #'R_from_SP': TODO
    'SP_salinometer': ('Rt', 't'),
    #'SP_from_SK':  TODO
    #
    # steric.py
    # 'steric_height': TODO
    #
    #water_column_48.py
    'Nsquared': ('SA', 'CT', 'p', 'lat')  #NOTE: Second output is un-tested.
    #'Turner_Rsubrho',  TODO
    #'IPV_vs_fNsquared_ratio'  TODO
    #
   }


# Make aliases for some values to be used as arguments
# cv.SA_from_Sstar != cv.SA_from_SP
# cv.SA_from_Sstar != gsw.SA_from_Sstar(Sstar, p, lon, lat)
# cv.SA_from_Sstar == gsw.SA_from_Sstar(SA, p, lon, lat)
#
# cv.specvol_ST == cv.specvol_ST25
# cv.specvol_ST != cv.specvol
# cv.specvol_ST != 1/cv.rho_ST
# Correction: set cv.specvol_ST = cv.specvol

# Bug work-around
#cv.SA_from_Sstar = cv.SA_from_SP
#cv.specvol_CT = cv.specvol

# Arguments with "wrong" names
cv.C_chck_cast = cv.C_from_SP
#cv.SA_chck_cast      = cv.SA_from_SP
#cv.CT_chck_cast      = cv.CT_from_t
cv.pt_chck_cast      = cv.pt_from_t
#cv.Sstar_chck_cast   = cv.Sstar_from_SA
#cv.entropy_chck_cast = cv.entropy
cv.z_chck_cast       = cv.z_from_p
#cv.rho_chck_cast     = cv.rho
#cv.R_chck_cast       = cv.cndr
cv.pr_chck_cast      = cv.pr
#cv.p0_chck_cast      = cv.p_chck_cast_shallow
#cv.p1_chck_cast      = cv.p_chck_cast_deep

# Functions and targets which does not follow the naming convention.
not_match = {
    #'CT_first_derivatives'        : 'CT_SA',
    #'CT_second_derivatives'       : 'CT_SA_SA',
    #'enthalpy_first_derivatives'  : 'CT_SA_pt',
    #
    #'CT_maxdensity'               : 'CT_maxden',
    #'IPV_vs_fNsquared_ratio_CT25' : 'IPVfN2',
    #'Turner_CT25'                 : 'Tu',
    'Nsquared'               : 'n2',
    #'Rsubrho_CT25'                : 'Rsubrho',
    #'alpha_CT'                    : 'alpha_CTrab',
    #'alpha_CT25'                  : 'alpha_CT25rab',
    #'beta_CT'                     : 'beta_CTrab',
    #'beta_CT25'                   : 'beta_CT25rab',
    #'chem_potential_relative'     : 'chem_potential',
    #'cndr_from_SP'                : 'cndr',
    #'isopycnal_vs_ntp_CT_ratio_CT25' : 'G_CT_CT25',
    #'ntp_pt_vs_CT_ratio_CT25'     : 'ntpptCT_CT25',
    #'pt0_from_t'                  : 'pt0',
    #'pt_from_CT'                  : 'pt',
    #'pt_maxdensity'               : 'pt_maxden',
    #'t_maxdensity'                : 't_maxden',
    }

# Add target aliases to cv
for f in not_match:
    setattr(cv, f, getattr(cv, not_match[f]))
    setattr(cv, f + '_ca', getattr(cv, not_match[f] + '_ca'))


# Generic test method
def generic_test(self, func=None, argnames=None):
    """Generic test function, to be specialized by functools.partial"""
    # Transform argument names to name convention in cv dataset
    args = [getattr(cv, a + '_chck_cast') for a in argnames]
    # Perform the function call
    out = getattr(gsw, func)(*args)
    # FIXME: Testing just the first output!
    if isinstance(out, tuple):
        out = out[0]
    # Check that the maximal error is less than the given tolerance
    maxdiff = np.nanmax(abs(out - getattr(cv, func)))
    #print maxdiff, getattr(cv, func+'_ca')
    self.assertTrue(maxdiff < getattr(cv, func + '_ca'))


# Dictionary of functions with corresponding test methods
function_test = {}
for f in function_arguments:
    function_test[f] = functools.partial(generic_test,
                      func=f, argnames=function_arguments[f])


# Auto-generated TestCase
class Test_profiles(unittest.TestCase):

    for f in function_test:
        method_def = ("test_" + f +
            " = lambda self: function_test['" + f + "'](self)")
        #print method_def
        exec(method_def)


if __name__ == '__main__':
    # A more verbose output
    suite = unittest.TestLoader().loadTestsFromTestCase(Test_profiles)
    a = unittest.TextTestRunner(verbosity=2).run(suite)
    if a.errors or a.failures:
        sys.exit(256)
    #unittest.main()
