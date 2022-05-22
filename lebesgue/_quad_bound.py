"""Bound integrals by numerical quadrature.

Appropriate functions are weakly decreasing on the unit interval, as are prior
masses above likelihood ratios.
"""
from collections.abc import Callable

import numba
import numpy

from ._cache import MutableCache


@MutableCache
def integrator(func: Callable) -> Callable:
    """Return a function to integrate func(args, x) on [0, 1].

    Arguments:
        func: numba.float64(args: Any, x:numba.float64)
            assumed to be non-increasing with x and to return values in [0, 1]

    compilation can be slow

    cache with numba to reduce repeat compilations like so:

    ```
    # at a top level location
    _integrate_func = integrator(func)

    @integrator.put(func)
    @numba.njit(cache=True)
    def integrate_func(args, rtol):
        return _integrate_func(args, rtol)

    ```
    """

    @numba.njit
    def quad_bound(args, rtol):
        # 2 ** -1022 is the smallest positive normal float
        nscan = 1022
        atol = 2.0**-nscan

        fs = numpy.empty(nscan + 1)

        # exponential scan
        zhi = 0.0
        zlo = 0.0
        hi = 1.0

        fs[0] = fhi = func(args, hi)
        for i in range(1, nscan + 1):
            size = lo = 0.5 * hi
            fs[i] = flo = func(args, lo)

            zlo += size * fhi

            # account for mass in the tail slice
            tol = max(atol, rtol * (zlo + flo * size))

            err_tail = (1.0 - flo) * size
            if err_tail * i * 2 < tol:
                break

            hi = lo
            fhi = flo

        nbins = i

        # distribute tolerance to large boxes such that
        # they are large enough to comfortably pull us below tol
        thresh = tol * (0.5 / nbins)

        # refine selection tolderance based on boxes surviving initial thresh
        nlost = 0
        tol_kept = tol - err_tail

        size = 1.0
        fhi = fs[0]
        for flo in fs[1 : nbins + 1]:
            size *= 0.5

            err = flo * size - fhi * size

            if not err > thresh:
                tol_kept -= err
                nlost += 1

            fhi = flo

        nkept = nbins - nlost
        if nkept > 0:
            err_thresh = tol_kept / nkept
        else:
            err_thresh = numpy.nan

        # refine each bin with a linear scan
        zlo = 0.0
        zhi = 0.0

        hi = 1.0
        fhi = fs[0]
        for flo in fs[1 : nbins + 1]:
            size = lo = 0.5 * hi

            err = flo * size - fhi * size

            # doubling steps halves err remaining after scan
            ngrid = _next_pow2(err / err_thresh)
            step = (hi - lo) / max(1, ngrid)

            # accumulate the middle section of the sum before adding on
            # head and tail for lo and hi bounds
            zmid = 0.0
            xlo = hi
            for i in range(1, ngrid):
                # for suitably chosen hi lo and ngrid (powers of 2), repeated
                # subtraction of step can be exact arithmetic
                xlo -= step
                zmid += func(args, xlo)

            # step is constant, so factor it out
            zlo += (fhi + zmid) * step
            zhi += (zmid + flo) * step

            hi = lo
            fhi = flo

        zlo += flo * size
        zhi += 1.0 * size
        return zlo, zhi

    return quad_bound


@numba.njit(numba.int64(numba.float64))
def _next_pow2(x):
    """Return the least non-negative power of 2 >= x."""
    if not x > 0:
        return 0
    x = max(1, x)
    two_inverse_eps = 2.0**53
    return int(numpy.spacing(numpy.nextafter(x, 0)) * two_inverse_eps)


def integrator_signature(args):
    pair_float64 = numba.typeof((0.0, 0.0))
    typeof_args = numba.typeof(args)
    return pair_float64(typeof_args, numba.float64)
