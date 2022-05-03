"""
Implement the normal cumulative distirbution fucntion and frields.

Based on cephes in scipy:
https://github.com/scipy/scipy/blob/68a35309d0879466c0253b820001965eecc424af/scipy/special/cephes/ndtr.c

which states:

> /*
>  * Cephes Math Library Release 2.2:  June, 1992
>  * Copyright 1984, 1987, 1988, 1992 by Stephen L. Moshier
>  * Direct inquiries to 30 Frost Street, Cambridge, MA 02140
>  */
"""
import numba
import numpy
from numba import f8

from . import _core

# constants

P = (
    2.46196981473530512524e-10,
    5.64189564831068821977e-1,
    7.46321056442269912687e0,
    4.86371970985681366614e1,
    1.96520832956077098242e2,
    5.26445194995477358631e2,
    9.34528527171957607540e2,
    1.02755188689515710272e3,
    5.57535335369399327526e2,
)

Q = (
    # 1.0
    1.32281951154744992508e1,
    8.67072140885989742329e1,
    3.54937778887819891062e2,
    9.75708501743205489753e2,
    1.82390916687909736289e3,
    2.24633760818710981792e3,
    1.65666309194161350182e3,
    5.57535340817727675546e2,
)

R = (
    5.64189583547755073984e-1,
    1.27536670759978104416e0,
    5.01905042251180477414e0,
    6.16021097993053585195e0,
    7.40974269950448939160e0,
    2.97886665372100240670e0,
)

S = (
    # 1.0
    2.26052863220117276590e0,
    9.39603524938001434673e0,
    1.20489539808096656605e1,
    1.70814450747565897222e1,
    9.60896809063285878198e0,
    3.36907645100081516050e0,
)

T = (
    9.60497373987051638749e0,
    9.00260197203842689217e1,
    2.23200534594684319226e3,
    7.00332514112805075473e3,
    5.55923013010394962768e4,
)

U = (
    # 1.0
    3.35617141647503099647e1,
    5.21357949780152679795e2,
    4.59432382970980127987e3,
    2.26290000613890934246e4,
    4.92673942608635921086e4,
)


UTHRESH = 37.519379347
MAXLOG = 7.09782712893383996732e2
SQRT1_2 = 2**-0.5

# top level functions; all f8(f8); specialized below
@_core.jit(cache=True)
def ndtr(a):
    x = a * SQRT1_2
    z = abs(x)

    if z < SQRT1_2:
        return 0.5 + 0.5 * erf(x)

    y = 0.5 * erfc(z)

    if x > 0:
        return 1.0 - y

    return y


@_core.jit(cache=True)
def erfc(a):
    x = abs(a)

    if x < 1.0:
        return 1.0 - _erf_x_le_1(a)

    return _erfc_x_ge_1(a)


@_core.jit(cache=True)
def erf(a):
    x = abs(a)

    if x > 1.0:
        # yes this passes is x, not a
        r = 1.0 - _erfc_x_ge_1(x)
        return numpy.copysign(r, a)

    return _erf_x_le_1(a)


# specialized cases to remove recursion so we can cache with numba


@_core.jit
def _erf_x_le_1(a):
    r = a * polevl(a * a, T) / p1evl(a * a, U)
    return numpy.copysign(r, a)


@_core.jit
def _erfc_x_ge_1(a):
    x = abs(a)

    z = -a * a

    if z < -MAXLOG:
        return 2.0 * (a < 0)

    z = numpy.exp(z)

    if x < 8.0:
        p = polevl(x, P)
        q = p1evl(x, Q)
    else:
        p = polevl(x, R)
        q = p1evl(x, S)

    y = (z * p) / q

    if a < 0:
        return 2.0 - y

    return y


# utilities


@_core.jit
def polevl(x, coef):
    ans = coef[0]

    for c in coef[1:]:
        ans = ans * x + c

    return ans


@_core.jit
def p1evl(x, coef):
    ans = x + coef[0]

    for c in coef[1:]:
        ans = ans * x + c

    return ans


# specialization

_core.specialize(ndtr, f8(f8))
_core.specialize(erf, f8(f8))
_core.specialize(erfc, f8(f8))
