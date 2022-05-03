""" Base module. """
import functools

import numba

try:
    from functools import cache
except ImportError:
    cache = functools.lru_cache(maxsize=None)
