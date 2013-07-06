import sys
import os
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

mfile = os.path.join(mfiledir, "gsw_check_functions.m")
mfilelines = open(mfile, 'rt').readlines()

def find(x):
    """
    Numpy equivalent to Matlab find.
    """
    return np.nonzero(x.flatten())[0]


first_pass = []

concat = False
for line in mfilelines:
    line = line.strip()
    if concat:
        if line.endswith('...'):
            line = line[:-3]
        first_pass[-1] += line
        if line.endswith(';'):
            concat = False
        continue
    if '=' in line and (line.startswith('gsw_') or line.startswith('[gsw_')):
        if line.endswith('...'):
            line = line[:-3]
            concat = True
        first_pass.append(line)

second_pass = []

for line in first_pass:
    if not '(' in line:
        continue
    if 'which' in line:
        continue
    line = line.replace('gsw_', '')
    if line.startswith('['):
        line = line[1:].replace(']', '')
    if line.endswith(';'):
        line = line[:-1]
    line = line.replace('(I)', '') # for deltaSA_atlas
    second_pass.append(line)

pairs = []

for i in range(len(second_pass)):
    if 'find(' in second_pass[i] and not 'find(' in second_pass[i-1]:
        pairs.extend(second_pass[i-1:i+1])

def group_or(line):
    """
    Numpy wart: using bitwise or as a fake elementwise logical or,
    we need to add parentheses.
    """
    if not ('find(' in line and '|' in line):
        return line
    i0 = line.index('find(') + 5
    head = line[:i0]
    tail = line[i0:]
    parts = tail.replace('|', ') | (')
    new = head + '(' + parts + ')'
    return new

final = [group_or(line) for line in pairs]

class FunctionCheck(object):
    def __init__(self, linepair):
        self.linepair = linepair
        self.runline = linepair[0]
        self.testline = linepair[1]

        # parse the line that runs the function
        head, tail = self.runline.split('=')
        self.outstrings = [s.strip() for s in head.split(',')]
        self.outstr = ','.join(self.outstrings)
        funcstr, argpart = tail.split('(', 1)
        self.name = funcstr.strip()
        self.argstrings = [s.strip() for s in argpart[:-1].split(',')]
        self.argstr = ','.join(self.argstrings)

        # parse the line that checks the results
        head, tail = self.testline.split('=', 1)
        self.resultstr = head.strip()      # cv.I*
        head, tail = tail.split('(',1)
        self.teststr = tail.strip()[:-1]   # argument of "find()"

        # To be set when run() is successful
        self.outlist = None
        self.result = None   # will be a reference to the cv.I* array
        self.passed = None   # will be set to True or False

        # To be set if run() is not successful
        self.exception = None

    def __str__(self):
        return self.runline

    def run(self):
        try:
            exec(self.runline)
            # In Matlab, the number of output arguments varies
            # depending on the LHS of the assignment, but Python
            # always returns the full set.  Here we handle the
            # case where Python is returning 2 (or more) but
            # the LHS is assigning only the first.
            if len(self.outstrings) == 1:
                if isinstance(eval(self.outstr), tuple):
                    exec("%s = %s[0]" % (self.outstr, self.outstr))
            self.outlist = [eval(s) for s in self.outstrings]
            exec(self.testline)
            self.result = eval(self.resultstr)
            self.passed = len(self.result) == 0

        except Exception as e:
            self.exception = e

checks = []
for i in range(0, len(final), 2):
    pair = final[i:i+2]
    checks.append(FunctionCheck(pair))


datadir = os.path.join(os.path.dirname(gsw.utilities.__file__), 'data')
cv = Bunch(np.load(os.path.join(datadir, 'gsw_cv_v3_0.npz')))
cf = Bunch()

for fc in checks:
    fc.run()

passes = [f.name for f in checks if f.passed]
failures = [f.name for f in checks if f.passed is False]

run_problems = [f.name for f in checks if f.exception is not None]

etypes = [NameError, UnboundLocalError, TypeError, AttributeError]
ex_dict = dict()
for exc in etypes:
    elist = [(f.name, f.exception) for f in checks
                if isinstance(f.exception, exc)]
    ex_dict[exc] = elist

print "\n%s tests were translated from gsw_check_functions.m" % len(checks)
print "\n%s tests ran with no error and with correct output" % len(passes)
print "\n%s tests had an output mismatch:" % len(failures)
print " ", "\n  ".join(failures)

print "\n%s exceptions were raised as follows:" % len(run_problems)
for exc in etypes:
    print "  ", exc.__name__
    strings = ["     %s : %s" % e for e in ex_dict[exc]]
    print "\n".join(strings)
    print ""

checkbunch = Bunch([(c.name, c) for c in checks])

def find_arguments():
    argset = set()
    for c in checks:
        argset.update(c.argstrings)
    argsetlist = list(argset)
    argsetlist.sort()
    return argsetlist

def find_arglists():
    alset = set()
    for c in checks:
        alset.update([c.argstr])
    arglists = list(alset)
    arglists.sort()
    return arglists

