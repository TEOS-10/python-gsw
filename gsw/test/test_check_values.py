# -*- coding: utf-8 -*-

"""Unit tests using the check values from the version 3 documentation,
http://www.teos-10.org/pubs/gsw/html/gsw_contents.html."""

import sys
import unittest
import numpy as np
import numpy.testing

import gsw

# Standard values for arguments from
# http://www.teos-10.org/pubs/gsw/html/gsw_contents.html
C = [34.5487, 34.7275, 34.8605, 34.6810, 34.5680, 34.5600]
t = [28.7856, 28.4329, 22.8103, 10.2600,  6.8863,  4.4036]
p = [10, 50, 125, 250, 600, 1000]

# Salinities
#SP = [34.5487, 34.7275, 34.8605, 34.6810, 34.5680, 34.5600]
#SA = [34.7118, 34.8915, 35.0256, 34.8472, 34.7366, 34.7324]
#Sstar = [34.7115, 34.8912, 35.0247, 34.8436, 34.7291, 34.7197]

# Temperatures
#t = [28.7856, 28.4329, 22.8103, 10.2600,  6.8863,  4.4036]
#pt = [28.7832, 28.4209, 22.7850, 10.2305,  6.8292,  4.3245]
#CT = [28.8099, 28.4392, 22.7862, 10.2262,  6.8272,  4.3236]
#t48 = [29, 28, 23, 10, 7, 4]
#t68 = [29, 28, 23, 10, 7, 4]

# Other
#p = [10, 50, 125, 250, 600, 1000]
#z = [10, 50, 125, 250, 600, 1000]
#rho = [1021.839, 1022.262, 1024.426, 1027.792, 1029.839, 1032.002]
#entropy = [400.3892, 395.4378, 319.8668, 146.7910,  98.6471,  62.7919]
#lon = 188
#lat = 4


class Test_standard(unittest.TestCase):
#class Test_standard(numpy.testing.TestCase):

    # ----------------------
    # practical_salinity.py
    # ----------------------

    def test_SP_from_C(self):
        """Practical Salinity from Conductivity"""
        output = gsw.SP_from_C(C, t, p)
        check_values = np.array((20.009869599086951,
                                 20.265511864874270,
                                 22.981513062527689,
                                 31.204503263727982,
                                 34.032315787432829,
                                 36.400308494388170))
        numpy.testing.assert_array_equal(output, check_values)

# -----------------------------------------------
if __name__ == '__main__':
    # Verbose output.
    suite = unittest.TestLoader().loadTestsFromTestCase(Test_standard)
    a = unittest.TextTestRunner(verbosity=2).run(suite)
    if a.errors or a.failures:
        sys.exit(1)
    #unittest.main()
    #numpy.testing.test()
