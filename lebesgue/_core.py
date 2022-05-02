""" Base module. """
import functools

import numba

try:
    # wishful thinking...
    from numba import jitclass
except ImportError:
    from numba.experimental import jitclass

jit = numba.jit(nopython=True)


try:
    from functools import cache
except ImportError:
    cache = functools.lru_cache(maxsize=None)
