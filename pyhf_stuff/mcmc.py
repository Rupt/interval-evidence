"""Extract statistics with Markov Chain Monte Carlo (MCMC)."""
import jax
import numpy
import scipy.optimize
from tensorflow_probability.substrates import jax as tfp

from .region_properties import region_properties


def generic_chain_hist(
    kernel_func,
    region,
    nbins,
    range_,
    *,
    seed,
    nburnin,
    nsamples,
    nrepeats,
):
    properties = region_properties(region)

    optimum = scipy.optimize.minimize(
        properties.objective_value_and_grad,
        properties.init,
        bounds=properties.bounds,
        jac=True,
        method="L-BFGS-B",
    )

    cov = properties.objective_hess_inv(optimum.x)

    x_of_t, t_of_x = eye_covariance_transform(optimum.x, cov)

    init = jax.numpy.zeros_like(properties.init)

    def logdf(t):
        x = x_of_t(t)
        return properties.logdf(x) + _boundary(x, properties.bounds)

    def observable(t):
        return properties.yield_value(x_of_t(t))

    # mcmc chain sapling
    def chain(key, state):
        _, yields = tfp.mcmc.sample_chain(
            kernel=kernel_func(logdf),
            trace_fn=lambda t, _: observable(t),
            current_state=state,
            num_burnin_steps=nburnin,
            num_results=nsamples,
            seed=key,
        )
        return _histogram(yields, nbins, range_)

    keys = jax.random.split(jax.random.PRNGKey(seed), nrepeats)
    vchain = jax.jit(jax.vmap(chain, in_axes=(0, None)))
    return vchain(keys, init)


def eye_covariance_transform(mean, cov):
    """Return functions x <=> t; t has identity covariance and zero mean.

    Arguments:
        mean: shape (n,)
        cov: shape (n, n), positive definite
    """
    # want x(t) such that:
    #     (x - m).T @ C^-1 @ (x - m) = t @ t    (1)
    # cholesky decompose:
    #     C = A @ A.T, C^-1 = A.T^-1 @ A^-1
    # if x - m = A @ t, then
    # (1) = t.T @ A.T @ A.T^-1 @ A^-1 @ A @ t = t @ t :)
    # x = A @ t + m <=> t = A^-1 @ (x - m)
    chol = jax.numpy.linalg.cholesky(cov)
    inv_chol = jax.numpy.linalg.inv(chol)

    def x_of_t(t):
        return chol @ t + mean

    def t_of_x(x):
        return inv_chol @ (x - mean)

    return x_of_t, t_of_x


# utilities


def _and_minus_infinity(test: bool, *, dtype=float):
    # float cast suppresses warnings
    big_negative = float(numpy.finfo(dtype).min)
    return big_negative * test * 2


def _boundary(x, bounds):
    in_bounds = ((x >= bounds[:, 0]) & (x <= bounds[:, 1])).all()
    return _and_minus_infinity(~in_bounds, dtype=x.dtype)


def _histogram(x, bins, range_, *, dtype=jax.numpy.int32):
    # linearly find bin index
    lo, hi = range_
    bin_per_x = bins / (hi - lo)
    i_float = jax.numpy.floor((x - lo) * bin_per_x)
    i_clip = jax.numpy.clip(i_float, 0, bins - 1)

    # cast to integer for indexing
    index_type = jax.numpy.uint32
    assert bins <= jax.numpy.iinfo(index_type).max
    i = i_clip.astype(index_type)

    # histogramming; only add where in bounds (no under/overflow)
    hist = jax.numpy.zeros(bins, dtype=dtype)
    return hist.at[i].add(i_float == i_clip)


def _summarize_hists(hists, *, axis=0):
    yields = numpy.sum(hists, axis=axis)
    # standard error on mean is std / sqrt(n), so
    # standard error on sum is std / sqrt(n) * n = std * sqrt(n)
    errors = numpy.std(hists, axis=axis) * hists.shape[axis] ** 0.5
    return yields, errors


def n_by_variance(hists):
    """Return an estimate of the number of independent samples in hist bins.

    If poisson-ish, then n in hists are y +- sqrt(y).
    Find y which might reproduce the variance observed in hists.

    Arguments:
        hists: integer counts, shape (nrepeats, nbins)
    """
    mean = numpy.mean(hists, axis=0)
    var = numpy.var(hists, axis=0)
    return _n_by_stats(mean, var)


def n_by_fit(data_class):
    mean = numpy.array(data_class.yields) / data_class.nrepeats
    std = numpy.array(data_class.errors) * data_class.nrepeats**-0.5
    return _n_by_stats(mean, std**2)


def _n_by_stats(mean, var):
    # max with 1 avoids div0 when mean is zero
    return mean**2 / numpy.maximum(var, numpy.maximum(mean, 1))
