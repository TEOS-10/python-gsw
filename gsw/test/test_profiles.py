# -*- coding: utf-8 -*-

"""Unit check for standard profiles for the
Gibbs Sea Water python package.

Functionality similar to matlab_test.py
without the nice output."""

# Autogenerates and perform a set of test methods like:
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
# TI  : Returning a tuple, tested individually
# NI: Not Implemented
# NV: No test value
# NA: Don't fit the testing scheme
# ERR: Gives error

# Could perhaps be auto-generated
function_arguments = {
    #density_enthalpy_ct.py
    'enthalpy_CT_exact': ('SA', 'CT', 'p'),
    #conversion.py
    'pt_from_t': ('SA', 't', 'p', 'pr'),
    't_from_CT': ('SA', 'CT', 'p'),
    #'pt_from_entropy': ('SA', 'entropy'),  #FIXME: ERR
    'CT_from_pt': ('SA', 'pt'),
    #'pot_enthalpy_from_pt': 'pot_enthalpy',   #FIXME: ERR
    'pt0_from_t': ('SA', 't', 'p'),
    'pt_from_CT': ('SA', 'CT'),
    #'SP_from_SA': ('SA', 'p', 'long', 'lat'),  #NOTE: NI
    #'Sstar_from_SA': ('SA', 'p', 'long', 'lat'),  #NOTE: NI
    #'SA_from_Sstar': ('Sstar', 'p', 'long', 'lat'),  #NOTE: NI
    #'SP_from_Sstar': ('Sstar', 'p', 'long', 'lat'),  #NOTE: NI
    'z_from_p': ('p', 'lat'),
    #'p_from_z': ('z', 'lat'),  #FIXME: ERR
    't90_from_t48': ('t',),
    't90_from_t68': ('t',),
    #derivatives.py
    #'CT_first_derivatives': ('SA', 'pt'),  #NOTE: TI
    #'CT_second_derivatives': ('SA', 'pt'),  #NOTE: TI
    #'enthalpy_first_derivatives': ('SA', 'CT', 'p'),
    #'enthalpy_second_derivatives': ('SA', 'CT', 'p'),
    #'entropy_first_derivatives': ('SA', 'CT'),  #NOTE: TI
    #'entropy_second_derivatives': ('SA', 'pt'),  #NOTE: TI
    #'pt_first_derivatives':,  #NOTE: TI
    #'pt_second_derivatives':  #NOTE: TI
    #earth.py
    'f': ('lat',),
    'grav': ('lat', 'p'),
    'distance': ('long', 'lat', 'p'),
    #isobaric.py
    #'latentheat_evap_t': ('SA'. 't')
    #library.py
    #'gibbs':
    #'entropy_part':
    #'gibbs_pt0_pt0':
    #'entropy_part_zerop':
    #'enthalpy_SSO_0_CT25': ('p',),
    #'specvol_SSO_0_CT25': ('p',),
    #'delta_SA': ('p', 'long', 'lat'),
    #'SA_from_SP_Baltic': ('SP', 'long', 'lat'),
    #'SP_from_SA_Baltic' : ('SA', 'long', 'lat'),
    #'interp_SA_CT':  #NOTE: NI
    #practical_salinity.py
    'SP_from_C': ('C', 't', 'p'),
    #C_from_SP
    #SP_from_R
    #R_from_SP
    #SP_salinometer
    #TODO:
    'CT_from_t': ('SA', 't', 'p'),
    #'rho_t_exact': ('SA', 't', 'p'),  #FIXME not in cv
    #'CT_derivative_SA': ('SA', 'pt'),
    #'CT_derivative_SA_SA': ('SA', 'pt'),
    #'CT_derivative_SA_pt': ('SA', 'pt'),
    #'CT_derivative_pt': ('SA', 'pt'),
    #'CT_derivative_pt_pt': ('SA', 'pt'),
    #'CT_from_entropy': ('SA', 'entropy'),
    #'CT_maxdensity': ('SA', 'p'),
    #'Helmholtz_energy_t_exact': ('SA', 't', 'p'),  #FIXME not in cv
    #'IPV_vs_fNsquared_ratio_CT25': ('SA', 'CT', 'p', 'pr'),
    #'Nsquared_CT25': ('SA', 'CT', 'p', 'lat'),
    #'Rsubrho_CT25': ('SA', 'CT', 'p'),
    #'SA_from_SP': ('SP', 'p', 'long', 'lat'),
    #'Sstar_from_SP': ('SP', 'p', 'long', 'lat'),
    #'SA_from_rho': ('rho', 't', 'p'),
    #'SA_Sstar_from_SP': ('SP', 'p', 'long', 'lat'),  #NOTE: TI
    #'Sstar_from_SP': ('SP', 'p', 'long', 'lat'),
    #'Turner_Rsubrho_CT25': ('SA', 'CT', 'p'),  #NOTE: TI
    #'Turner_CT25': ('SA', 'CT', 'p'),
    #'adiabatic_lapse_rate': ('SA', 't', 'p'),
    #'alpha_CT': ('SA', 'CT', 'p'),
    #'alpha_CT25': ('SA', 'CT', 'p'),
    #'alpha_wrt_CT': ('SA', 't', 'p'),
    #'alpha_wrt_pt': ('SA', 't', 'p'),
    #'alpha_wrt_t': ('SA', 't', 'p'),
    #'beta_CT': ('SA', 'CT', 'p'),
    #'beta_CT25': ('SA', 'CT', 'p'),
    #'beta_const_CT': ('SA', 't', 'p'),
    #'beta_const_pt': ('SA', 't', 'p'),
    #'beta_const_t': ('SA', 't', 'p'),
#BUG in profile values on file
    #'cabbeling_CT25': ('SA', 't', 'p'),
    #'chem_potential_relative': ('SA', 't', 'p'),
    #'chem_potential_salt': ('SA', 't', 'p'),
    #'chem_potential_water': ('SA', 't', 'p'),
    #'cndr_from_SP': ('SP', 't', 'p'),
    #'cp': ('SA', 't', 'p'),
    #'enthalpy': ('SA', 't', 'p'),
    #'enthalpy_CT': ('SA', 'CT', 'p'),
    #'enthalpy_CT25': ('SA', 'CT', 'p'),
    #'enthalpy_derivative_CT': ('SA', 'CT', 'p'),
    #'enthalpy_derivative_CT_CT': ('SA', 'CT', 'p'),
    #'enthalpy_derivative_p': ('SA', 'CT', 'p'),
    #'enthalpy_derivative_SA': ('SA', 'CT', 'p'),
    #'enthalpy_derivative_SA_CT': ('SA', 'CT', 'p'),
    #'enthalpy_derivative_SA_SA': ('SA', 'CT', 'p'),
    #'enthalpy_diff_CT': ('SA', 'CT', 'p0', 'p1'),  #NOTE: TI
    #'enthalpy_diff_CT25': ('SA', 'CT', 'p0', 'p1'),  #NOTE: TI
    #'entropy': ('SA', 't', 'p'),
    #'entropy_derivative_SA': ('SA', 'CT'),
    #'entropy_derivative_CT': ('SA', 'CT'),
    #'entropy_from_CT': ('SA', 'CT'),
    #'entropy_from_pt': ('SA', 'pt'),
    #'entropy_derivative_CT_CT': ('SA', 'CT'),
    #'entropy_derivative_SA_CT': ('SA', 'CT'),
    #'entropy_derivative_SA_SA': ('SA', 'CT'),
    #'geo_strf_Cunningham':  #NOTE: NI
    #'geo_strf_McD_Klocker':
    #'geo_strf_McD_Klocker_pc':  #NOTE: NI
    #'geo_strf_Montgomery' :  #NOTE: NI
    #'geo_strf_dyn_height' :  #NOTE: NI
    #'geo_strf_dyn_height_pc':  #NOTE: NI
    #'geostrophic_velocity' :
    #'internal_energy': ('SA', 't', 'p'),
    #'interp_McD_Klocker':  #NOTE: NI
    #'ionic_strength': ('SA',),
    #'isochoric_heat_cap': ('SA', 't', 'p'),
    #'isopycnal_slope_ratio_CT25': ('SA', 'CT', 'p'),
    #'isopycnal_vs_ntp_CT_ratio_CT25': ('SA', 'CT', 'p'),
    #'kappa': ('SA', 't', 'p'),
    #'kappa_const_t': ('SA', 't', 'p'),
    #'molality': ('SA',)
    #'ntp_pt_vs_CT_ratio_CT25': ('SA', 'CT', 'p'),
    #'osmotic_coefficient': ('SA', 't', 'p'),
    #'pot_enthalpy_from_pt': ('SA', 'pt'),
    #'pot_rho': ('SA', 't', 'p', 'pr'),
    #'pt_derivative_CT': ('SA', 'CT'),
    #'pt_derivative_CT_CT': ('SA', 'CT'),
    #'pt_derivative_SA': ('SA', 'CT'),
    #'pt_derivative_SA_CT': ('SA', 'CT'),
    #'pt_derivative_SA_SA': ('SA', 'CT'),
    #'pt_maxdensity': ('SA', 'p'),
    #'rho': ('SA', 't', 'p'),
    #'rho_CT': ('SA', 'CT', 'p'),
    #'rho_CT25': ('SA', 'CT', 'p'),
    #'rho_alpha_beta_CT': ('SA', 'CT', 'p'),  #NOTE: TI
    #'rho_alpha_beta_CT25': ('SA', 'CT', 'p'),  #NOTE: TI
    #'sigma0_CT': ('SA', 'CT'),
    #'sigma0_pt': ('SA', 'pt'),
    #'sigma1_CT': ('SA', 'CT'),
    #'sigma2_CT': ('SA', 'CT'),
    #'sigma3_CT': ('SA', 'CT'),
    #'sigma4_CT': ('SA', 'CT'),
    #'sound_speed': ('SA', 't', 'p'),
    #'specvol': ('SA', 't', 'p'),
    #'specvol_CT': ('SA', 'CT', 'p'),
    #'specvol_CT25': ('SA', 'CT', 'p'),
    #'specvol_anom': ('SA', 't', 'p'),
    #'specvol_anom_CT': ('SA', 'CT', 'p'),
    #'specvol_anom_CT25': ('SA', 'CT', 'p'),
    #'temps_maxdensity': ('SA', 'p'),  #NOTE: TI
    #'t_maxdensity' : ('SA', 'p'),
    #'thermobaric_CT25': ('SA', 'CT', 'p'),
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
    #'CT_derivative_SA'            : 'CT_SA',
    #'CT_derivative_SA_SA'         : 'CT_SA_SA',
    #'CT_derivative_SA_pt'         : 'CT_SA_pt',
    #'CT_derivative_pt'            : 'CT_pt',
    #'CT_derivative_pt_pt'         : 'CT_pt_pt',
    #'CT_derivative_SA'            : 'CT_SA',
    #'CT_maxdensity'               : 'CT_maxden',
    #'IPV_vs_fNsquared_ratio_CT25' : 'IPVfN2',
    #'Turner_CT25'                 : 'Tu',
    #'Nsquared_CT25'               : 'n2',
    #'Rsubrho_CT25'                : 'Rsubrho',
    #'alpha_CT'                    : 'alpha_CTrab',
    #'alpha_CT25'                  : 'alpha_CT25rab',
    #'beta_CT'                     : 'beta_CTrab',
    #'beta_CT25'                   : 'beta_CT25rab',
    #'chem_potential_relative'     : 'chem_potential',
    #'cndr_from_SP'                : 'cndr',
    #'enthalpy_derivative_CT'      : 'h_CT',
    #'enthalpy_derivative_CT_CT'   : 'h_CT_CT',
    #'enthalpy_derivative_p'       : 'h_P',
    #'enthalpy_derivative_SA'      : 'h_SA',
    #'enthalpy_derivative_SA_CT'   : 'h_SA_CT',
    #'enthalpy_derivative_SA_SA'   : 'h_SA_SA',
    #'entropy_derivative_CT'       : 'eta_CT',
    #'entropy_derivative_CT_CT'    : 'eta_CT_CT',
    #'entropy_derivative_SA'       : 'eta_SA',
    #'entropy_derivative_SA_CT'    : 'eta_SA_CT',
    #'entropy_derivative_SA_SA'    : 'eta_SA_SA',
    #'isopycnal_vs_ntp_CT_ratio_CT25' : 'G_CT_CT25',
    #'ntp_pt_vs_CT_ratio_CT25'     : 'ntpptCT_CT25',
    #'pt0_from_t'                  : 'pt0',
    #'pt_derivative_CT'            : 'pt_CT',
    #'pt_derivative_CT_CT'         : 'pt_CT_CT',
    #'pt_derivative_SA'            : 'pt_SA',
    #'pt_derivative_SA_CT'         : 'pt_SA_CT',
    #'pt_derivative_SA_SA'         : 'pt_SA_SA',
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
    """Generic test function, to be spesialized by functools.partial"""
    # Transform argument names to name convention in cv dataset
    args = [getattr(cv, a + '_chck_cast') for a in argnames]
    # Perform the function call
    out = getattr(gsw, func)(*args)
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
