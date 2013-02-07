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

import h5py
import numpy as np


def get_data(data):
    return np.atleast_1d(np.squeeze(data)).T

data_ver = 'v3_0'
gsw_data = h5py.File('gsw_data_%s.mat' % data_ver, mode='r')

# Delta SA Atlas.
ref_table = dict()
for k in gsw_data:
    if k == u'gsw_cv' or k == u'#refs#' or k == 'gsw_demo_data':
        pass
    else:
        if k == 'deltaSA_ref':
            name = 'delta_SA_ref'
        else:
            name = k
        if name == 'version_number' or name == 'version_date':
            var = ''.join([unichr(c) for c in gsw_data[k][:]])
        else:
            var = get_data(gsw_data[k][:])
        ref_table.update({name: var})
np.savez("gsw_data_%s" % data_ver, **ref_table)

# Save demo data values gsw_demo_data` in a separate file.
gsw_demo_data = gsw_data['gsw_demo_data']

demo_vars = dict()
for name in gsw_demo_data:
    var = get_data(gsw_demo_data[name][:])
    demo_vars.update({name: var})
np.savez("gsw_demo_data_%s" % data_ver, **demo_vars)

# Save compare values `gsw_cv` in a separate file.
gsw_cv = gsw_data['gsw_cv']

cv_vars = dict()
for name in gsw_cv:
    var = get_data(gsw_cv[name][:])
    cv_vars.update({name: var})
np.savez("gsw_cv_%s" % data_ver, **cv_vars)

gsw_data.close()

# NOTE: This is a saved result of a modified version of `gsw_check_functions.m`
# where the structure variable gsw_cf was saved.  The matlab version relies
# on the result of some of its functions to test others, so we need this file.
gsw_cf = h5py.File('gsw_cf.mat', mode='r')
gsw_cf['gsw_cf']

cf_vars = dict()
for name in gsw_cf:
    var = get_data(gsw_cf[name][:])
    cf_vars.update({name: var})
np.savez("gsw_cf", **cf_vars)
