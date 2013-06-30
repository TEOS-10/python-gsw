#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# mat2npz.py
#
# purpose:  Convert matlab file from TEOS-10 group to a npz file
# author:   Filipe P. A. Fernandes
# e-mail:   ocefpaf@gmail
# web:      http://ocefpaf.tiddlyspot.com/
# created:  06-Jun-2011
# modified: Thu 07 Feb 2013 03:46:02 PM BRST
#
# obs:
#

import numpy as np
from pycurrents.file.matfile import loadmatbunch

data_ver = 'v3_0'
gsw_data = loadmatbunch('gsw_data_%s.mat' % data_ver, masked=False)

# Delta SA Atlas.
ref_table = dict()
for k in gsw_data:
    if k == u'gsw_cv' or k == u'#refs#' or k == 'gsw_demo_data':
        pass
    else:
        ref_table[k] = gsw_data[k]
np.savez("gsw_data_%s" % data_ver, **ref_table)

# Save demo data values gsw_demo_data in a separate file.
gsw_demo_data = gsw_data['gsw_demo_data']

np.savez("gsw_demo_data_%s" % data_ver, **gsw_data['gsw_demo_data'])

# Save compare values `gsw_cv` in a separate file.
cv_vars = gsw_data['gsw_cv']

np.savez("gsw_cv_%s" % data_ver, **cv_vars)

# NOTE: This is a saved result of a modified version of `gsw_check_functions.m`
# where the structure variable gsw_cf was saved.  The matlab version relies
# on the result of some of its functions to test others, so we need this file.

# Turned off; I don't have the mat file.  Maybe the existing npz file is OK.
if False:
    gsw_cf = loadmatbunch('gsw_cf.mat', masked=False)
    np.savez("gsw_cf", **gsw_cf)

