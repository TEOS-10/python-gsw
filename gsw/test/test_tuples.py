# -*- coding: utf-8 -*-

"""Unit check for standard profiles for the Gibbs Sea Water python package."""

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
cf = Dict2Struc(np.load(os.path.join(datadir, 'gsw_cf_.npz')))

# derivatives.py
#CT_first_derivatives=('SA', 'pt'),  # NOTE: TI, BUG
#CT_second_derivatives=('SA', 'pt'),  # NOTE: TI FIXME BUG
#enthalpy_first_derivatives=('SA', 'CT', 'p'), FIXME name match
#enthalpy_second_derivatives=('SA', 'CT', 'p'), FIXME name match
#entropy_first_derivatives=('SA', 'CT'),  #NOTE: TI FIXME name match
#entropy_second_derivatives=('SA', 'pt'),  #NOTE: TI FIXME name match
#pt_first_derivatives':,  #NOTE: TI FIXME name match
#pt_second_derivatives= #NOTE: TI FIXME name match

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
cv.R_cf_chck_cast = cf.R
cv.rho_cf_chck_cast = cf.rho

def generic_test(self, func=None, argnames=None):
    """Generic test function, to be specialized by functools.partial"""
    # Transform argument names to name convention in cv dataset
    args = [getattr(cv, a + '_chck_cast') for a in argnames]
    # Perform the function call
    out = getattr(gsw, func)(*args)
    # FIXME: Testing just the first output!
    # TODO: Create the tuples and compare all together.
    # Check that the maximal error is less than the given tolerance
    if isinstance(out, tuple):
        out = out[0]
    maxdiff = np.nanmax(abs(out - getattr(cv, func)))
    try:
        self.assertTrue(maxdiff < getattr(cv, func + '_ca'))
    except AssertionError, e:
        raise AssertionError("Error in %s %s" % (func, e.args))


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

[gsw_cf.CT_SA, gsw_cf.CT_pt] = gsw_CT_first_derivatives(gsw_cv.SA_chck_cast,gsw_cf.pt);
[gsw_cf.ICT_first_deriv] = find(abs(gsw_cv.CT_SA - gsw_cf.CT_SA) >= gsw_cv.CT_SA_ca | ...
    (gsw_cv.CT_pt - gsw_cf.CT_pt) >= gsw_cv.CT_pt_ca);
if ~isempty(gsw_cf.ICT_first_deriv)
    fprintf(2,'gsw_CT_first_derivatives:   Failed\n');
    gsw_cf.gsw_chks = 0;
end