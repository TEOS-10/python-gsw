# -*- coding: utf-8 -*-
#
# test_octave.py
#
# purpose:  Quick "compare test" with octave results.
# author:   Filipe P. A. Fernandes
# e-mail:   ocefpaf@gmail
# web:      http://ocefpaf.tiddlyspot.com/
# created:  13-Jun-2013
# modified: Thu 13 Jun 2013 06:36:42 PM BRT
#
# obs:  This is different from `test_check_values.py`, that tests
# against the results in the docs, `test_octave.py` uses same input values to
# run both python and Matlab versions (using Octave).
#
# This is not a thorough test, just an "ad hoc" test to compare when a new
# Matlab version is released.
#

import os
import sys
from collections import OrderedDict

import numpy as np

from oct2py import octave
from oct2py._utils import Oct2PyError

import gsw

#from mlabwrap import MatlabPipe
#mlab = MatlabPipe(matlab_process_path='/home/filipe/bin/matlab',
                    #matlab_version='2013a')
#mlab.open()

try:
    path = sys.argv[1]
except IndexError:
    path = "./matlab_gsw_v3_03"

if not os.path.exists(path):
    raise ValueError("matlab gsw path %s not found" % path)

_ = octave.addpath(octave.genpath(path))

def compare_results(name, function, args):
    args = [values.get(arg) for arg in args]
    print(name)

    try:  # Python.
        res = function(*args)
    except:
        print('%s: python runtime error' % name)
        raise
        return 'no_python'

    nout = 1
    if isinstance(res, tuple):
        nout = len(res)
        res = res[0]

    try:  # Octave.
        val = octave.call('gsw_%s' % name, *args, verbose=True, nout=nout)
        if nout > 1:
            val = val[0]
    except Oct2PyError:
        print('%s: Octave runtime error' % name)
        return 'no_octave'

    val = val.flatten()
    if (val == res).all():
        print('%s: Passed' % name)
        return 'passed'
    else:
        print('%s: Failed' % name)
        print("octave:\n%s" % val)
        print("python:\n%s" % res)
        return 'failed'
    print('')

values = dict(C=np.array([34.5487, 34.7275, 34.8605, 34.6810, 34.568, 34.56]),
              t=np.array([28.7856, 28.4329, 22.8103, 10.2600, 6.8863, 4.4036]),
              p=np.array([10., 50., 125., 250., 600., 1000.]),
              SP=np.array([34.5487, 34.7275, 34.8605, 34.6810, 34.5680,
                           34.5600]),
              SA=np.array([34.7118, 34.8915, 35.0256, 34.8472, 34.7366,
                           34.7324]),
              CT=np.array([28.7856, 28.4329, 22.8103, 10.2600, 6.8863, 4.4036]),
              ps=0,
              pt=0,
              pp=0,
              lat=np.array([4., 4., 4., 4., 4., 4.]),
              lon=np.array([188., 188., 188., 188., 188., 188.]),
              pt0=np.array([28.8099, 28.4392, 22.7862, 10.2262, 6.8272,
                            4.3236]),
              spycnl=np.array([21.8482, 22.2647, 24.4207, 27.7841, 29.8287,
                               31.9916]),
              A='s2', # or anything else, like 'gamma', for gamma_n space
              p_i=500.0,
              # Baltic.
              SAb=np.array([6.6699, 6.7738, 6.9130, 7.3661, 7.5862, 10.3895]),
              SPb=np.array([6.5683, 6.6719, 6.8108, 7.2629, 7.4825, 10.2796]),
              latb=np.array([59., 59., 59., 59., 59., 59.]),
              lonb=np.array([20., 20., 20., 20., 20., 20.])
              )
# Functions.
library = OrderedDict({
    'enthalpy_SSO_0_p': (gsw.library.enthalpy_SSO_0_p, ('p')),
    'entropy_part': (gsw.entropy_part, ('SA', 't', 'p')),
    'entropy_part_zerop': (gsw.entropy_part_zerop, ('SA', 'pt0')),
    'gibbs': (gsw.library.gibbs, ('ps', 'pt', 'pp', 'SA', 't', 'p')),
    'gibbs_pt0_pt0': (gsw.library.gibbs_pt0_pt0, ('SA', 'pt0')),
    'Hill_ratio_at_SP2': (gsw.library.Hill_ratio_at_SP2, ('t')),
    'infunnel': (gsw.library.infunnel, ('SA', 'CT', 'p')),

    # The interp functions fail on octave because of "superiorfloat".
    # There is also a problem with the argument handling in interp_SA_CT.
    'interp_ref_cast': (gsw.library.interp_ref_cast, ('spycnl', 'A')),
    'interp_SA_CT': (gsw.library.interp_SA_CT, ('SA', 'CT', 'p', 'p_i')),

    'SAAR': (gsw.library.SAAR, ('p', 'lon', 'lat')),
    'SA_from_SP_Baltic': (gsw.library.SA_from_SP_Baltic, ('SPb', 'lonb',
                                                          'latb')),
    'specvol_SSO_0_p': (gsw.library.specvol_SSO_0_p, ('p')),
    'SP_from_SA_Baltic': (gsw.library.SP_from_SA_Baltic, ('SAb', 'lonb',
                                                          'latb')),
    })


if __name__ == '__main__':
    outcomes = ['passed', 'no_octave', 'no_python', 'failed']
    results = dict([(k, list()) for k in outcomes])

    for name, (function, args) in library.iteritems():
        ret = compare_results(name=name, function=function, args=args)
        results[ret].append(name)

    print('\nSummary:')
    print('passed:\n  %s' % '\n  '.join(results['passed']))
    print('octave call failed:\n  %s' % '\n  '.join(results['no_octave']))
    print('python call failed:\n  %s' % '\n  '.join(results['no_python']))
    print('results did not match:\n  %s' % '\n  '.join(results['failed']))
    print('')

