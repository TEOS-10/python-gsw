import sys
import os
import glob

import logging

import numpy as np

from pycurrents.system import Bunch

import gsw
from gsw.gibbs import *

log = logging.getLogger()
logging.basicConfig()

try:
    mfiledir = sys.argv[1]
except IndexError:
    mfiledir = "../../../../TEOS-10/matlab_gsw_V3_03"

subdirs = ['./', 'library', 'thermodynamics_from_t']

mfile = os.path.join(mfiledir, "gsw_check_functions.m")
mfilelines = open(mfile, 'rt').readlines()

def function_parts(subdir='./'):
    d = os.path.join(mfiledir, subdir)
    mfilelist = glob.glob(os.path.join(d, '*.m'))
    partslist = []
    for fpath in mfilelist:
        base = os.path.basename(fpath)[:-2]
        if not base.startswith('gsw_') or base == 'gsw_check_functions':
            continue
        topline = open(fpath).readline().strip()
        if not topline.startswith('function'):
            log.warn("path %s, topline is\n%s", fpath, topline)
            continue
        cmd = topline.split(None, 1)[1]

        parts = cmd.split('=')
        if len(parts) == 1:
            out = ''
            func_call = parts[0].strip()
        else:
            out = parts[0].strip()
            if out.startswith('['):
                out = out[1:-1]
            func_call = parts[1].strip()
        out = [a.strip() for a in out.split(',')]

        parts = func_call.split('(')
        if len(parts) == 1:
            argstring = ''
        else:
            argstring = parts[1][:-1]

        args = [a.strip() for a in argstring.split(',')]
        partslist.append((base[4:], tuple(args), tuple(out)))

    return partslist

def arguments(partslist):
    args = set()
    for entry in partslist:
        args.update(entry[1])

    arglist = list(args)
    arglist.sort()
    return arglist

def arglists(partslist):
    args = set()
    for entry in partslist:
        args.update([', '.join(entry[1])])

    arglist = list(args)
    arglist.sort()
    return arglist

def funclist_by_arg(partslist):
    out = dict()
    for entry in partslist:
        if entry[1] not in out:
            out[entry[1]] = []
        out[entry[1]].append(entry[0])
    return out

