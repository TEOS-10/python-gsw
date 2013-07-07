"""
Functions for parsing gsw mfiles.

These might become useful in a scheme for automating docstring
generation.
"""

import sys
import os
import glob
from collections import OrderedDict

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

def funcline_parts(base, topline):
    """
    Parse the m-file function declaration.

    Returns [base, arguments, outputs] where base is the
    function name without the gsw_ prefix, arguments is a
    tuple of strings, and outputs is a tuple of strings.
    """

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

    parts = [base[4:], tuple(args), tuple(out)]
    return parts

help_sections = ['USAGE:',
                 'DESCRIPTION:',
                 'INPUT:',
                 'OPTIONAL:',
                 'OUTPUT:',
                 'AUTHOR:',
                 'VERSION NUMBER:',
                 'REFERENCES:',
                 ]

def helptext_parts(lines):
    """
    Parse the help text from gsw m-files.

    Returns [lines, indexdict], where lines are the lines of
    text with extraneous material removed, and indexdict is
    a dictionary in which the keys are the help sections and
    the values are the indices into lines where the section
    titles are found.
    """
    helplines = []
    parts = OrderedDict()
    for line in lines[2:]:
        line = line.rstrip()
        if line.startswith('%'):
            if line.endswith('=========================='):
                continue
            helplines.append(line[1:])
        else:
            break
    for i, line in enumerate(helplines):
        for section in help_sections:
            if section in line:
                parts[section] = i

    return helplines, parts



def function_parts(subdir='./'):
    """
    Parse the gsw m-files in a given subdirectory.

    Returns a list of parts: function base name, arguments,
    inputs, and the help lines with a section index dictionary.
    """

    d = os.path.join(mfiledir, subdir)
    mfilelist = glob.glob(os.path.join(d, '*.m'))
    partslist = []
    for fpath in mfilelist:
        base = os.path.basename(fpath)[:-2]
        if not base.startswith('gsw_') or base == 'gsw_check_functions':
            continue
        lines = open(fpath).readlines()
        topline = lines[0].strip()
        if not topline.startswith('function'):
            log.warn("path %s, topline is\n%s", fpath, topline)
            continue

        parts = funcline_parts(base, topline)

        parts.append(helptext_parts(lines))

        partslist.append(parts)

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

def help_chunk(helptuple, sect):
    lines, secdict = helptuple
    sections = secdict.keys()
    ind = secdict.values()
    try:
        i0 = sections.index(sect)
    except ValueError:
        return []   # or None?
    # Assuming the sections are always in the expected order, this will
    # work; otherwise we would have to look for the nearest section index
    # after the one found.
    return lines[ind[i0]:ind[i0 + 1]]


def funcs_with_descriptions(partslist):
    fdlist = [(p[0], help_chunk(p[3], 'DESCRIPTION:')) for p in partslist]
    return OrderedDict(fdlist)


