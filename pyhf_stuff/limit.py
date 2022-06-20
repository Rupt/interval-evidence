"""Tools to assign likelihood limits on signal contributions."""
import scipy

standard_normal_cdf = scipy.special.ndtr


def quantile(prior, q: float, *, x0=1.0, xtol=1e-6):
    """Return an approximate quantile for the given lebesgue prior.

    Works by interval bisection in its cumulative distribution.

    Written for non-negative priors, so begins from a lower limit of 0.

    Arguments:
        prior: lebesgue Prior-like
        q: quantile (target cumulative distribution function value)
        x0: initial estimated upper bound on the quantile
        xtol: passed to scipy.optimize.bisect
    """
    if not 0.0 <= q <= 1.0:
        raise ValueError(q)

    def func(x):
        return prior.between(0.0, x) - q

    # initial expansion to find an upper bound
    hi = x0
    while not func(hi) > 0:
        hi *= 2

    return scipy.optimize.bisect(func, 0.0, hi, xtol=xtol)
