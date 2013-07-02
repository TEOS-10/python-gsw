# -*- coding: utf-8 -*-

from __future__ import division

import os
from functools import wraps

import numpy as np

__all__ = [
           'match_args_return',
           'Dict2Struc',
           'Cache_npz',
           'read_data',
           'strip_mask',
           ]

def repair_npzfile_with_objects(infile, outfile):
    """
    Read an npz file written based on scipy.io.loadmat,
    and write out a new npz file in which arrays have been
    extracted from object arrays.

    This might be needed only during the development process.
    It is motivated by the need to fix gsw_cf.npz so that
    the special object array handling in to_masked is not needed
    for the tests.
    """
    dat = np.load(infile)
    out = dict()
    for k, v in dat.iteritems():
        if v.dtype.kind == 'O':
            v = v.item()
        out[k] = v
    np.savez(outfile, **out)

def to_masked(arg):
    r"""
    Ensure an argument is a floating-point masked array.

    This is a helper for match_args_return.
    """
    if not np.iterable(arg):
        arg = [arg]
    try:
        arg = np.ma.array(arg, copy=False, dtype=float)
    except ValueError:
        # We might not want to keep this here.  It handles the
        # case where reading a matfile with scipy has yielded
        # an object array containing a single object, which is
        # the array one actually wants.
        if arg.dtype.kind == 'O':
            arg = np.ma.array(arg.item(), copy=False, dtype=float)
        else:
            raise
    return np.ma.masked_invalid(arg)


def match_args_return(f):
    """
    Decorator for most functions that operate on profile data.
    """
    @wraps(f)
    def wrapper(*args, **kw):
        p = kw.get('p', None)
        if p is not None:
            args = list(args)
            args.append(p)

        isarray = np.any([hasattr(a, '__iter__') for a in args])
        ismasked = np.any([np.ma.isMaskedArray(a) for a in args])

        def fixup(ret):
            if not ismasked:
                ret = np.ma.filled(ret, np.nan)
            if not isarray:
                ret = ret[0]
            return ret

        newargs = [to_masked(a) for a in args]
        if p is not None:
            kw['p'] = newargs.pop()

        ret = f(*newargs, **kw)

        if isinstance(ret, tuple):
            retlist = [fixup(arg) for arg in ret]
            ret = tuple(retlist)
        else:
            ret = fixup(ret)
        return ret
    wrapper.__wrapped__ = f
    return wrapper


class Dict2Struc(object):
    r"""Open variables from a dictionary in a "matlab-like-structure"."""
    def __init__(self, adict):
        for k in adict.files:
            self.__dict__[k] = adict[k]


class Cache_npz(object):
    def __init__(self):
        self._cache = dict()
        self._default_path = os.path.join(os.path.dirname(__file__), 'data')

    def __call__(self, fname, datadir=None):
        if datadir is None:
            datadir = self._default_path
        fpath = os.path.join(datadir, fname)
        try:
            return self._cache[fpath]
        except KeyError:
            pass
        d = np.load(fpath)
        ret = Dict2Struc(d)
        self._cache[fpath] = ret
        return ret

_npz_cache = Cache_npz()


def read_data(fname, datadir=None):
    r"""Read variables from a numpy '.npz' file into a minimal class providing
    attribute access.  A cache is used to avoid re-reading the same file."""
    return _npz_cache(fname, datadir=datadir)


def strip_mask(*args):
    r"""Process the standard arguments for efficient calculation.

    Return unmasked arguments, plus a mask.

    The first argument, SA, is handled specially so that it can be

    This could be absorbed into a decorator, but it would
    require redefining functions to take the additional
    mask argument or kwarg.
    """
    mask = np.ma.getmaskarray(args[-1])
    SA = args[0]
    if SA.shape:
        SA = np.ma.asarray(SA)
        SA[SA < 0] = np.ma.masked
        for a in args[:-1]:
            mask = np.ma.mask_or(mask, np.ma.getmask(a))
        newargs = [SA.filled(0)]
    elif SA < 0:
        SA = 0
        for a in args[1:-1]:
            mask = np.ma.mask_or(mask, np.ma.getmask(a))
        newargs = [SA]
    newargs.extend([np.ma.filled(a, 0) for a in args[1:]])
    newargs.append(mask)
    return newargs
