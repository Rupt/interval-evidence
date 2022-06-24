"""pyhf front to mcmc_core."""

from multiprocessing import get_context

import jax
import numpy

from .mcmc_core import (
    CallJitCache,
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
    nprocesses,
):
    properties = region_properties(region)
    optimum_x = region_fit(region).x

    cov = properties.objective_hess_inv(optimum_x)
    x_of_t, _ = eye_covariance_transform(optimum_x, cov)

    initializer = zeros(optimum_x.shape)

    logdf = logdf_template(region, x_of_t)

    observable = yields_template(region, x_of_t)

    # mcmc stuff
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

    if nprocesses <= 1:
        return jax.jit(jax.vmap(chain()))(keys)

    # process spawning is expensive, but reused processe can reused compiled
    # jax code; maximize reuse on by reusing each on maximal chunks
    chunksize = nrepeats // nprocesses + bool(nrepeats % nprocesses)

    # caution to avoid sending jax arrays through multiprocessing
    keys = numpy.array(keys)

    # https://github.com/google/jax/issues/6790
    # "spawn" avoids creashes with Mutex warnings
    with get_context("spawn").Pool(nprocesses) as pool:
        hists = pool.map(CallJitCache(chain), keys, chunksize=chunksize)

    # reference our state after the pool.map to ensure no garbage collection,
    # which may have been causing sporadic errors
    del chain

    return jax.numpy.stack(hists)


def logdf_template(region, x_of_t):
    properties = region_properties(region)
    return _logdf_template_inner(
        properties.init_raw,
        properties.free,
        properties.model_blind.logpdf,
        properties.data,
        properties.bounds_raw,
        x_of_t,
    )


@partial_once
def _logdf_template_inner(init_raw, free, logdf_func, data, bounds, x_of_t):
    def unpack(x):
        return jax.numpy.array(init_raw).at[free].set(x)

    def logdf(t):
        x = unpack(x_of_t(t))
        (logdf,) = logdf_func(x, data)
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
        x = unpack(x_of_t(t))
        return yields_func(x)[index]

    return observable
