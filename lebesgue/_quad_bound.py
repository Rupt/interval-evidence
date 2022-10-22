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
    # Beginning at x=1, we halve nscan many times in the exponential scan.
    # 2**-1022 is the smallest positive normal float (IEEE 754 binary64)
    nscan = 1022
    atol = 2.0**-nscan

    def quad_bound(args, rtol):
        # Scan exponentially down x=1, x=0.5, ..., x=2**-n
        # to get f(x) array and quadrature tolerance information.
        fs, tol, err_tail = _exp2_scan(args, rtol)

        # Explore the scan results to choose scaling for refinement.
        err_scale = _choose_err_scale(fs, tol, err_tail)

        # Refine with linear quadrature in each boxes
        return _linear_scan(args, fs, err_scale)

    @numba.njit
    def _exp2_scan(args, rtol):
        fs = numpy.empty(nscan + 1)

        # Accumulate a lower bound on the integral down scan depth.
        zlo = 0.0
        xhi = 1.0
        fs[0] = fhi = func(args, xhi)
        for i in range(1, nscan + 1):
            size = xlo = 0.5 * xhi
            fs[i] = flo = func(args, xlo)

            zlo += size * fhi

            # Set relative tolerance from a greatest lower bound given by
            # lower bound at current scan depth plus largest tail area.
            # Combine absolute and relative tolerances.
            tol = max(atol, rtol * (zlo + flo * size))

            err_tail = (1.0 - flo) * size
            if err_tail * i * 2 < tol:
                break

            xhi = xlo
            fhi = flo

        return fs[: i + 1], tol, err_tail

    @numba.njit
    def _linear_scan(args, fs, err_scale):
        # Refine each bin with a linear scan.
        zlo = 0.0
        zhi = 0.0

        hi = 1.0
        fhi = fs[0]
        for flo in fs[1:]:
            size = lo = 0.5 * hi

            err = flo * size - fhi * size

            # Doubling steps halves err remaining after scan.
            ngrid = _next_pow2(err * err_scale)
            step = (hi - lo) / max(1, ngrid)

            # Accumulate the middle section of the sum before adding head and
            # tail for lo and hi bounds.
            zmid = 0.0
            xlo = hi
            for i in range(1, int(ngrid)):
                # For our precisely chosen hi lo and ngrid (powers of 2),
                # repeated subtraction of step is exact arithmetic.
                xlo -= step
                zmid += func(args, xlo)

            # Step is constant, so factors out.
            zlo += (fhi + zmid) * step
            zhi += (zmid + flo) * step

            hi = lo
            fhi = flo

        # Add the tail.
        zlo += flo * size
        zhi += 1.0 * size
        return zlo, zhi

    return numba.njit(quad_bound)


@numba.njit
def _choose_err_scale(fs, tol, err_tail):
    nbins = len(fs) - 1
    # Refining each scan box subtracts from remaining tolerance.
    # Choose a threshold below which to ignore boxes, such that refining the
    # remaining boxes can pull us to the desired precision.
    thresh = tol * (0.5 / nbins)

    err_live = tol - err_tail
    nlive = nbins
    size = 1.0
    fhi = fs[0]
    for flo in fs[1:]:
        size *= 0.5

        err = flo * size - fhi * size

        if not err > thresh:
            err_live -= err
            nlive -= 1

        fhi = flo

    return nlive / err_live


@numba.njit(numba.float64(numba.float64))
def _next_pow2(x):
    """Return the least non-negative power of 2 that is >= x."""
    # max(1, nan) == 1, but numpy maximum propagates the nan.
    x = numpy.maximum(1, x)

    # Use 1 + the exponent of the float before x, which can be extracted
    # from their spacing.
    spacing = x - numpy.nextafter(x, 0)
    two_inverse_eps = 2 / numpy.finfo(numpy.float64).eps
    return spacing * two_inverse_eps


def integrator_signature(args):
    """Return a numba type signature for integrator functions."""
    pair_float64 = numba.typeof((0.0, 0.0))
    typeof_args = numba.typeof(args)
    return pair_float64(typeof_args, numba.float64)
