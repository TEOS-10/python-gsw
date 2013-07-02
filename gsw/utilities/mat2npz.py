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
# modified: Mon 01 Jul 2013 08:24:06 PM BRT
#
# obs: pycurrents.file.matfile.loadmatbunch is part of CODAS.
# hg clone   http://currents.soest.hawaii.edu/hg/pycurrents
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
np.savez("data/gsw_data_%s" % data_ver, **ref_table)

# Save demo data values gsw_demo_data in a separate file.
gsw_demo_data = gsw_data['gsw_demo_data']

np.savez("data/gsw_demo_data_%s" % data_ver, **gsw_data['gsw_demo_data'])

# Save compare values `gsw_cv` in a separate file.
cv_vars = gsw_data['gsw_cv']

np.savez("data/gsw_cv_%s" % data_ver, **cv_vars)

# NOTE: The matfile gsw_cf.mat is just the structure variable `gsw_cf` from
# gsw_check_functions.m.

if True:
    gsw_cf = loadmatbunch('gsw_cf.mat', masked=False)
    np.savez("data/gsw_cf", **gsw_cf)
