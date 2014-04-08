# -*- coding: utf-8 -*-

from __future__ import division

import os
from functools import wraps

import numpy as np
from scipy.io import loadmat

__all__ = ['Bunch',
           'Cache_npz',
           'match_args_return',
           'read_data',
           'strip_mask']


# Based on Robert Kern's Bunch.
class Bunch(dict):
    """
    A dictionary that also provides access via attributes.

    Additional methods update_values and update_None provide
    control over whether new keys are added to the dictionary
    when updating, and whether an attempt to add a new key is
    ignored or raises a KeyError.

    The Bunch also prints differently than a normal
    dictionary, using str() instead of repr() for its
    keys and values, and in key-sorted order.  The printing
    format can be customized by subclassing with a different
    str_ftm class attribute.  Do not assign directly to this
    class attribute, because that would substitute an instance
    attribute which would then become part of the Bunch, and
    would be reported as such by the keys() method.

    To output a string representation with
    a particular format, without subclassing, use the
    formatted() method.
    """

    str_fmt = "{0!s:<{klen}} : {1!s:>{vlen}}\n"

    def __init__(self, *args, **kwargs):
        """
        *args* can be dictionaries, bunches, or sequences of
        key,value tuples.  *kwargs* can be used to initialize
        or add key, value pairs.
        """
        dict.__init__(self)
        self.__dict__ = self
        for arg in args:
            self.update(arg)
        self.update(kwargs)

    def __str__(self):
        return self.formatted()

    def formatted(self, fmt=None):
        """
        Return a string with keys and/or values.

        *fmt* is a format string as used in the str.format() method.

        The str.format() method is called with key, value as positional
        arguments, and klen, vlen as kwargs.  The latter are the maxima
        of the string lengths for the keys and values, respectively,
        up to respective maxima of 20 and 40.
        """
        if fmt is None:
            fmt = self.str_fmt
        klens = [len(str(k)) for k in self.keys()]
        vlens = [len(str(v)) for v in self.values()]
        klen = min(20, max(klens))
        vlen = min(40, max(vlens))
        items = self.items()
        items.sort()
        slist = [fmt.format(k, v, klen=klen, vlen=vlen) for k, v in items]
        return ''.join(slist)

    def update_values(self, *args, **kw):
        """
        arguments are dictionary-like; if present, they act as
        additional sources of kwargs, with the actual kwargs
        taking precedence.

        One reserved optional kwarg is "strict".  If present and
        True, then any attempt to update with keys that are not
        already in the Bunch instance will raise a KeyError.
        """
        strict = kw.pop("strict", False)
        newkw = dict()
        for d in args:
            newkw.update(d)
        newkw.update(kw)
        self._check_strict(strict, newkw)
        dsub = dict([(k, v) for (k, v) in newkw.items() if k in self])
        self.update(dsub)

    def update_None(self, *args, **kw):
        """
        Similar to update_values, except that an existing value
        will be updated only if it is None.
        """
        strict = kw.pop("strict", False)
        newkw = dict()
        for d in args:
            newkw.update(d)
        newkw.update(kw)
        self._check_strict(strict, newkw)
        dsub = dict([(k, v) for (k, v) in newkw.items() if k in self and
                     self[k] is None])
        self.update(dsub)

    def _check_strict(self, strict, kw):
        if strict:
            bad = set(kw.keys()) - set(self.keys())
            if bad:
                bk = list(bad)
                bk.sort()
                ek = self.keys()
                ek.sort()
                raise KeyError(
                    "Update keys %s don't match existing keys %s" % (bk, ek))


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
        ret = Bunch(d)
        self._cache[fpath] = ret
        return ret

_npz_cache = Cache_npz()


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


def read_data(fname, datadir=None):
    """
    Read variables from a numpy '.npz' file into a minimal class providing
    attribute access.  A cache is used to avoid re-reading the same file.
    """
    return _npz_cache(fname, datadir=datadir)


def strip_mask(*args):
    """
    Process the standard arguments for efficient calculation.

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


def _unicode(arr):
    r"""
    loadmat seems to be mishandling strings when there is a difference
    in byte order between the machine that wrote the file and the one
    that is reading the file.  The result is, e.g.,

        u'\U31000000\U34000000'  instead of u'14'

    """
    try:
        return unicode(arr)
    except UnicodeEncodeError:
        dt = arr.dtype.newbyteorder('S')
        return unicode(arr.view(dt))


def crunch(arr, masked=True):
    arr = arr.squeeze()
    if arr.ndim == 0:
        kind = arr.dtype.kind
        if kind == 'f':
            return float(arr)
        if kind in 'ui':
            return int(arr)
        if kind == 'U':
            try:
                return _unicode(arr)
            except UnicodeDecodeError:
                return "Could not decode."
        if kind == 'S':
            return str(arr)
        if kind == 'O':
            return arr
        return arr  # Warn?  Other kinds need to be handled?
    if masked and arr.dtype.kind == 'f':  # Check for complex also.
        arrm = np.ma.masked_invalid(arr)
        if arrm.count() < arrm.size:
            arr = arrm
        else:
            arr = np.array(arr)  # Copy to force a read.
    else:
        arr = np.array(arr)
    return arr


def structured_to_bunch(arr, masked=True):
    if arr.dtype.kind == 'V' and arr.shape == (1, 1):
        b = Bunch()
        x = arr[0, 0]
        for name in x.dtype.names:
            b[name] = structured_to_bunch(x[name], masked=masked)
        return b
    return crunch(arr, masked=masked)


def loadmatbunch(fname, masked=True):
    """
    Wrapper for loadmat that dereferences (1,1) object arrays,
    converts floating point arrays to masked arrays, and uses
    nested Bunch objects in place of the MatlabTM structures.
    """
    out = Bunch()
    fobj = open(fname, 'rb')
    xx = loadmat(fobj)
    keys = [k for k in xx.keys() if not k.startswith("__")]
    for k in keys:
        out[k] = structured_to_bunch(xx[k], masked=masked)
    fobj.close()
    return out
