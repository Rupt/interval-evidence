"""pyhf front to mcmc_core."""
from functools import partial

import jax

from .mcmc_core import (
    eye_covariance_transform,
    histogram,
    partial_once,
    reduce_chain,
    zeros,
)
from .mcmc_tfp import _boundary
from .region_fit import region_fit
from .region_properties import region_properties


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
):
    properties = region_properties(region)
    optimum_x = region_fit(region).x

    cov = properties.objective_hess_inv(optimum_x)
    x_of_t, _ = eye_covariance_transform(optimum_x, cov)

    logdf = logdf_template(region, x_of_t)
    observable = yields_template(region, x_of_t)

    # mcmc stuff
    initializer = zeros(optimum_x.shape)
    kernel = kernel_func(logdf)
    reducer = histogram(nbins, range_, observable)

    chain = reduce_chain(
        initializer,
        kernel,
        reducer,
        nburnin=nburnin,
        nsamples=nsamples,
    )

    keys = jax.random.split(jax.random.PRNGKey(seed), nrepeats)
    key_shape = keys.shape[1:]
    n_processes = min(jax.device_count(), nrepeats)

    # pmap needs a rectangular shape, so split to cover any remainder
    chain_func = jax.pmap(_stable_vmap(chain()))

    n_bulk = nrepeats // n_processes * n_processes
    keys_bulk = keys[:n_bulk].reshape(n_processes, -1, *key_shape)
    out = chain_func(keys_bulk).reshape(n_bulk, -1)

    n_tail = len(keys) - n_bulk
    if n_tail > 0:
        keys_tail = keys[n_bulk:].reshape(n_tail, 1, *key_shape)
        out_tail = chain_func(keys_tail).reshape(n_tail, -1)
        out = jax.numpy.concatenate([out, out_tail])

    return out


def logdf_template(region, x_of_t):
    properties = region_properties(region)
    return _logdf_template_inner(
        properties.init_raw,
        properties.free,
        properties.model_blind.logpdf,
        properties.data,
        properties.bounds,
        x_of_t,
    )


@partial_once
def _logdf_template_inner(init_raw, free, logdf_func, data, bounds, x_of_t):
    def unpack(x):
        return jax.numpy.array(init_raw).at[free].set(x)

    def logdf(t):
        x = x_of_t(t)
        x_raw = unpack(x)
        (logdf,) = logdf_func(x_raw, data)
        return logdf + _boundary(x, bounds)

    return logdf


def yields_template(region, x_of_t):
    properties = region_properties(region)
    return _yields_template_inner(
        properties.init_raw,
        properties.free,
        properties.model_blind.expected_actualdata,
        properties.index,
        x_of_t,
    )


@partial_once
def _yields_template_inner(init_raw, free, yields_func, index, x_of_t):
    def unpack(x):
        return jax.numpy.array(init_raw).at[free].set(x)

    def observable(t):
        x_raw = unpack(x_of_t(t))
        return yields_func(x_raw)[index]

    return observable


# utilities


def _stable_vmap(func):
    """Return func mapped with jax.lax.map.

    jax.vmap appears to change semantics (possibly rounding?)
    """
    return partial(jax.lax.map, func)
