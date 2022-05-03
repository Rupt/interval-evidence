""" Base module. """
import functools

import numba

try:
    from functools import cache
except ImportError:
    cache = functools.lru_cache(maxsize=None)

jit = functools.partial(numba.jit, nopython=True)


def specialize(func, signature_or_list):
    if isinstance(signature_or_list, list):
        sigs = signature_or_list
    else:
        sigs = [signature_or_list]

    for sig in sigs:
        func.compile(sig)

    func.disable_compile()
