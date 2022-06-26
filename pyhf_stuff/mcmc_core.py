"""An MCMC implementation using picklable jax code."""
from functools import partial, wraps

import jax
import numpy

# partial is awesome.
# multiprocessing.Pool requires everything to be pickleable,
# so we use partial to avoid defining tonnes of container classes.


def partial_once(func):
    """Wrap func such that its first call returns a partial wrapper."""

    @wraps(func)
    def partial_func(*args, _partial_once=True, **kwargs):

        if _partial_once:
            return partial(partial_func, *args, _partial_once=False, **kwargs)

        return func(*args, **kwargs)

    return partial_func


# chain execution


@partial_once
def reduce_chain(
    initializer,
    kernel,
    reducer,
    *,
    nburnin,
    nsamples,
):
    kernel_init, kernel_step = kernel()
    reducte_init, reduce_ = reducer()
    init = initializer()

    def reduce_loop(i, state):
        args, reduction = state
        _, x, _ = args = kernel_step(*args)
        reduction = reduce_(reduction, x)
        return args, reduction

    def chain(rng):
        rng, key = jax.random.split(rng)

        x = init(key)
        kernel_state = kernel_init(x)
        reduction = reducte_init(x)

        args = (rng, x, kernel_state)

        # burnin discards its reduction
        args, _ = jax.lax.fori_loop(
            0,
            nburnin,
            reduce_loop,
            (args, reduction),
        )

        # main keeps only the reduction
        _, reduction = jax.lax.fori_loop(
            0,
            nsamples,
            reduce_loop,
            (args, reduction),
        )

        return reduction

    return chain


# transition kernels


@partial_once
def mala(step_size, logdf):

    value_and_grad = jax.value_and_grad(logdf())

    def init(x):
        logf, logf_grad = value_and_grad(x)
        mean = x + 0.5 * step_size * logf_grad
        return logf, mean

    def step(rng, x, state):
        logf, mean = state
        rng, key_noise, key_accept = jax.random.split(rng, 3)

        # propose the next step
        noise = jax.random.normal(key_noise, shape=x.shape, dtype=x.dtype)
        noise_to = step_size**0.5 * noise
        x_to = mean + noise_to
        logf_to, mean_to = state_to = init(x_to)

        # evaluate its acceptance ratio
        norm_to = noise_to.dot(noise_to)
        noise_from = x - mean_to
        norm_from = noise_from.dot(noise_from)

        log_accept = logf_to - logf + (0.5 / step_size) * (norm_to - norm_from)

        x, state = _metropolis(
            key_accept, log_accept, (x_to, state_to), (x, state)
        )

        return rng, x, state

    return init, step


@partial_once
def mix_mala_eye(step_size, prob_eye, logdf):

    value_and_grad = jax.value_and_grad(logdf())

    def init(x):
        logf, logf_grad = value_and_grad(x)
        mean = x + 0.5 * step_size * logf_grad
        return logf, mean

    def step(rng, x, state):
        logf, mean = state
        rng, key_mix, key_noise, key_accept = jax.random.split(rng, 4)

        # identity = eye = standard multivariate normal
        # is a special case of the langevin proposal with mean=0, step_size=1
        do_eye = jax.random.uniform(key_mix) < prob_eye

        # proposal
        noise = jax.random.normal(key_noise, shape=x.shape, dtype=x.dtype)
        noise_to = jax.lax.select(do_eye, noise, step_size**0.5 * noise)
        x_to = jax.lax.select(do_eye, noise, mean + noise_to)
        logf_to, mean_to = state_to = init(x_to)

        # acceptance
        norm_to = noise_to.dot(noise_to)
        noise_from = jax.lax.select(do_eye, x, x - mean_to)
        norm_from = noise_from.dot(noise_from)

        scale = jax.lax.select(do_eye, 0.5, 0.5 / step_size)
        log_accept = logf_to - logf + scale * (norm_to - norm_from)

        x, state = _metropolis(
            key_accept, log_accept, (x_to, state_to), (x, state)
        )

        return rng, x, state

    return init, step


def _metropolis(key, log_accept, state_to, state):
    """Implement the metropolis rule for transitioning states."""
    log_uniform = -jax.random.exponential(key, dtype=log_accept.dtype)
    return _tree_select(log_uniform <= log_accept, state_to, state)


# reductions


@partial_once
def histogram(nbins, range_, observable):

    func = observable()

    def init(_):
        return jax.numpy.zeros(nbins, dtype=jax.numpy.int32)

    def reduce_(state, x):
        hist = state
        return _histogram_reduce(hist, func(x), range_)

    return init, reduce_


# coordinate transforms


def eye_covariance_transform(mean, cov):
    """Return functions x <=> t; t has identity covariance and zero mean.

    Jacobian factors are constant and can be ignored.

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

    x_of_t = _linear_out(chol, mean)
    t_of_x = _linear_in(inv_chol, mean)
    return x_of_t, t_of_x


@partial_once
def _linear_out(weight, bias, t):
    return weight @ t + bias


@partial_once
def _linear_in(weight, bias, x):
    return weight @ (x - bias)


# initializers


@partial_once
def zeros(shape, dtype=None):
    def init(_):
        return jax.numpy.zeros(shape, dtype=dtype)

    return init


# distribution function unilities


def _and_minus_infinity(test: bool, *, dtype=float):
    # float cast suppresses warnings
    big_negative = float(numpy.finfo(dtype).min)
    return big_negative * test * 2


def _boundary(x, bounds):
    in_bounds = ((x >= bounds[:, 0]) & (x <= bounds[:, 1])).all()
    return _and_minus_infinity(~in_bounds, dtype=x.dtype)


# utility


def _tree_select(condition, tree_true, tree_false):
    return jax.tree_util.tree_map(
        lambda x, y: jax.lax.select(condition, x, y),
        tree_true,
        tree_false,
    )


def _histogram(x, nbins, range_, *, dtype=jax.numpy.int32):
    hist = jax.numpy.zeros(nbins, dtype=dtype)
    return _histogram_reduce(hist, x, range_)


def _histogram_reduce(hist, x, range_):
    nbins = len(hist)
    lo, hi = range_

    # linearly find bin index
    bin_per_x = nbins / (hi - lo)
    i_float = jax.numpy.floor((x - lo) * bin_per_x)

    # cast to integer for indexing
    index_type = jax.numpy.int32
    assert nbins <= jax.numpy.iinfo(index_type).max
    i = jax.numpy.clip(i_float, 0, nbins - 1).astype(index_type)

    # histogramming; only add where in bounds (no under/overflow/nan/inf)
    return hist.at[i].add(i == i_float)


def summarize_hists(hists, *, axis=0):
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
    # hedge the variance estimate by expanding it a little
    n = numpy.shape(hists)[0]
    return _n_by_stats(mean, var * n / (n - 1))


def n_by_fit(data_class):
    n = data_class.nrepeats
    mean = numpy.array(data_class.yields) / n
    std = numpy.array(data_class.errors) * n**-0.5
    return _n_by_stats(mean, std**2 * n / (n - 1))


def _n_by_stats(mean, var):
    # max with 1 avoids div0 when mean is zero
    return mean**2 / numpy.maximum(var, numpy.maximum(mean, 1))
