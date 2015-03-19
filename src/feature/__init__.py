


#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: __init__.py
# Date: Sat Nov 29 21:42:15 2014 +0800
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import sys
try:
    import BOB as MFCC
except:
    print >> sys.stderr, "Warning: failed to import Bob, use a slower version of MFCC instead."
    import MFCC
import LPC
import numpy as np

def get_extractor(extract_func, **kwargs):
    def f(tup):
        return extract_func(*tup, **kwargs)
    return f

#******************************************************
from scipy import sparse
def _assert_all_finite(X):
    """Like assert_all_finite, but only for ndarray."""
    if (X.dtype.char in np.typecodes['AllFloat'] and not np.isfinite(X.sum())
            and not np.isfinite(X).all()):
        raise ValueError("Array contains NaN or infinity.")


def assert_all_finite(X):
    """Throw a ValueError if X contains NaN or infinity.

    Input MUST be an np.ndarray instance or a scipy.sparse matrix."""

    # First try an O(n) time, O(1) space solution for the common case that
    # there everything is finite; fall back to O(n) space np.isfinite to
    # prevent false positives from overflow in sum method.
    _assert_all_finite(X.data if sparse.issparse(X) else X)
#******************************************************

def mix_feature(tup):
    #print ('mix feature')
    mfcc = MFCC.extract(tup)
    #assert_all_finite(mfcc)
    #print ('mfcc ok')
    #lpc = LPC.extract(tup)
    #assert_all_finite(lpc)
    #print ('lpc ok')
    if len(mfcc) == 0:
        print >> sys.stderr, "ERROR.. failed to extract mfcc feature:", len(tup[1])
    return mfcc
    #return np.concatenate((mfcc, lpc), axis=1)
