"""An MCMC implementation using picklable jax code."""
from functools import partial
from multiprocessing import get_context

import jax
import scipy

from .mcmc import _boundary
from .region_properties import region_properties

# partial is awesome.
# multiprocessing.Pool requires everything to be pickleable,
# so we use partial to avoid defining tonnes of container classes.


def partial_once(func):
    """Wrap func such that its first call returns a partial wrapper."""
    return partial(partial, func)


# pyhf model specifics


def region_hist_chain(
    kernel_func,
    region,
    nbins,
    range_,
    *,
    seed,
    nburnin,
    nsamples,
    nrepeats,
    nprocesses,
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

    logdf = logdf_template(
        x_of_t,
        properties.model_blind.logpdf,
        properties.data,
        properties.bounds,
    )
    observable = yields_template(
        x_of_t,
        properties.model_blind.expected_actualdata,
        properties.slice_,
    )

    kernel = kernel_func(logdf)
    reducer = histogram(nbins, range_, observable)
    initializer = zeros(properties.init.shape)

    chain = reduce_chain(
        kernel,
        reducer,
        initializer,
        nburnin=nburnin,
        nsamples=nsamples,
    )

    keys = jax.random.split(jax.random.PRNGKey(seed), nrepeats)

    if nprocesses > 1:
        # https://github.com/google/jax/issues/6790
        # "spawn" avoids Mutex warnings, crashes
        with get_context("spawn").Pool(nprocesses) as pool:
            hists = pool.map(call_jit_call(chain), keys)
    else:
        hists = jax.jit(jax.vmap(chain()))(keys)

    return jax.numpy.asarray(hists)


def _logdf_template(x_of_t, logdf_func, data, bounds):
    def logdf(t):
        x = x_of_t(t)
        (logdf,) = logdf_func(x, data)
        return logdf + _boundary(x, bounds)

    return logdf


logdf_template = partial_once(_logdf_template)


def _yields_template(x_of_t, yields_func, slice_):
    def observable(t):
        return yields_func(x_of_t(t))[slice_]

    return observable


yields_template = partial_once(_yields_template)

# transform


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

    x_of_t = partial(_linear_out, chol, mean)
    t_of_x = partial(_linear_in, inv_chol, mean)

    return x_of_t, t_of_x


def _linear_out(weight, bias, t):
    return weight @ t + bias


def _linear_in(weight, bias, x):
    return weight @ (x - bias)


# chain execution


def _reduce_chain(
    kernel,
    reducer,
    initializer,
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

    def chain(key):
        key, key_init = jax.random.split(key)

        x = init(key_init)
        kernel_state = kernel_init(x)
        reduction = reducte_init(x)

        args = (key, x, kernel_state)

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


reduce_chain = partial_once(_reduce_chain)

# transition kernels


def _langevin(step_size, logdf):

    value_and_grad = jax.value_and_grad(logdf())

    def init(x):
        logf, logf_grad = value_and_grad(x)
        mean = x + 0.5 * step_size * logf_grad
        return logf, mean

    def step(key, x, state):
        logf, mean = state

        # propose the next step
        noise_to = step_size**0.5 * jax.random.normal(
            key, shape=x.shape, dtype=x.dtype
        )
        x_to = mean + noise_to
        logf_to, mean_to = state_to = init(x_to)

        # evaluate its acceptance ratio
        norm_to = noise_to.dot(noise_to)

        noise_from = x - mean_to
        norm_from = noise_from.dot(noise_from)

        log_accept = logf_to - logf + (0.5 / step_size) * (norm_to - norm_from)

        return x_to, state_to, log_accept

    return init, step


langevin = partial_once(_langevin)


def _metropolis(kernel):

    init, step_inner = kernel()

    def step(key, x, state):
        key, key_inner, key_accept = jax.random.split(key, 3)

        # inner step
        x_to, state_to, log_accept = step_inner(key_inner, x, state)

        # metropolis rule
        log_uniform = jax.numpy.log(
            jax.random.uniform(key_accept, dtype=log_accept.dtype)
        )
        accept = log_uniform <= log_accept

        # conditional move
        x, state = _tree_select(accept, (x_to, state_to), (x, state))

        return key, x, state

    return init, step


metropolis = partial_once(_metropolis)


def _mala(step_size, logdf):
    return _metropolis(langevin(step_size, logdf))


mala = partial_once(_mala)

# reductions


def _histogram(nbins, range_, observable):

    func = observable()

    def init(_):
        return jax.numpy.zeros(nbins, dtype=jax.numpy.int32)

    def reduce_(state, x):
        hist = state
        return _histogram_reduce(hist, func(x), range_)

    return init, reduce_


histogram = partial_once(_histogram)

# initializers


def _zeros(shape):
    def init(_):
        return jax.numpy.zeros(shape)

    return init


zeros = partial_once(_zeros)

# for multiprocessing


def call_jit_call(func):
    cache = None

    def call(arg):
        nonlocal cache
        if cache is None:
            cache = jax.jit(func())
        return cache(arg)

    return call


# utility


def _histogram_reduce(hist, x, range_):
    # linearly find bin index
    bins = len(hist)
    lo, hi = range_
    bin_per_x = bins / (hi - lo)
    i_float = jax.numpy.floor((x - lo) * bin_per_x)
    i_clip = jax.numpy.clip(i_float, 0, bins - 1)

    # cast to integer for indexing
    index_type = jax.numpy.uint32
    assert bins <= jax.numpy.iinfo(index_type).max
    i = i_clip.astype(index_type)

    # histogramming; only add where in bounds (no under/overflow)
    return hist.at[i].add(i_float == i_clip)


def _tree_select(condition, tree_true, tree_false):
    return jax.tree_util.tree_map(
        lambda x, y: jax.lax.select(condition, x, y),
        tree_true,
        tree_false,
    )
