"""
Bound integrals by numerical quadrature.

Appropriate functions are weakly decreasing on the unit interval, as are prior
masses above likelihood ratios.
"""
from . import _core


@_core.jit
def _quad_bound(model, rtol):
    # 2 ** -1022 is the smallest positive normal float
    nscan = 1022
    atol = 2.0 ** -nscan

    fs = numpy.empty(nscan + 1)

    # exponential scan
    zhi = 0.0
    zlo = 0.0
    hi = 1.0

    fs[0] = fhi = model._mass(hi)
    for i in range(1, nscan + 1):
        size = lo = 0.5 * hi
        fs[i] = flo = model._mass(lo)

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
    err_thresh = tol * (0.5 / nbins)

    nkept = 0.0
    tol_kept = tol - err_tail

    size = 1.0
    fhi = fs[0]
    for i in range(1, nbins + 1):
        size *= 0.5
        flo = fs[i]

        err = flo * size - fhi * size
        keep = err > err_thresh

        nkept += keep
        tol_kept -= (not keep) * err

        fhi = flo

    if nkept > 0.0:
        err_thresh = tol_kept / nkept
    else:
        err_thresh = numpy.nan

    # recursive refinement
    zlo = 0.0
    zhi = 0.0

    hi = 1.0
    fhi = fs[0]
    for i in range(1, nbins + 1):
        size = lo = 0.5 * hi
        flo = fs[i]

        err = flo * size - fhi * size

        if err > err_thresh:
            cut = err - err_thresh
            ilo, ihi = _quad_bound_impl(model, cut, lo, hi, flo, fhi)
        else:
            ilo, ihi = fhi * size, flo * size

        zlo += ilo
        zhi += ihi

        hi = lo
        fhi = flo

    zlo += flo * size
    zhi += 1.0 * size
    return zlo, zhi


@_core.jit
def _quad_bound_recurse(model, cut, lo, hi, flo, fhi):
    size = 0.5 * (hi - lo)
    mid = lo + size
    fmid = model._mass(mid)

    # the box is halved, so err = 2 * err_new
    err_top = fmid * size - fhi * size
    err_bot = flo * size - fmid * size
    err_new = err_top + err_bot

    cut_new = cut - err_new

    cut_top = err_top * (cut_new / err_new)
    cut_bot = err_bot * (cut_new / err_new)

    # prefer lumping together to recursing equally down both sides
    if cut_new < err_bot - cut_top:
        cut_bot = cut_new
        cut_top = 0.0
    elif cut_new < err_top - cut_bot:
        cut_bot = 0.0
        cut_top = cut_new

    if cut_top > 0:
        lo_top, hi_top = _quad_bound_recurse(model, cut_top, mid, hi, fmid, fhi)
    else:
        lo_top, hi_top = fhi * size, fmid * size

    if cut_bot > 0:
        lo_bot, hi_bot = _quad_bound_recurse(model, cut_bot, lo, mid, flo, fmid)
    else:
        lo_bot, hi_bot = fmid * size, flo * size

    return lo_bot + lo_top, hi_bot + hi_top
