"""pyhf front to _mymc."""

from multiprocessing import get_context

import jax
import scipy

from .mcmc import _boundary
from .mymc import (
    CallJitCache,
    eye_covariance_transform,
    histogram,
    partial_once,
    reduce_chain,
    sphere,
)
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
    initializer = clip_to_x_bounds(
        sphere(properties.init.shape),
        x_of_t,
        properties.bounds,
        t_of_x,
    )

    chain = reduce_chain(
        kernel,
        reducer,
        initializer,
        nburnin=nburnin,
        nsamples=nsamples,
    )

    keys = jax.random.split(jax.random.PRNGKey(seed), nrepeats)

    if nprocesses <= 1:
        return jax.jit(jax.vmap(chain()))(keys)

    # process spawning is expensive, but reused processe can reused compiled
    # jax code; maximize reuse on by reusing each on maximal chunks
    chunksize = nrepeats // nprocesses + bool(nrepeats % nprocesses)
    # https://github.com/google/jax/issues/6790
    # "spawn" avoids creashes with Mutex warnings
    with get_context("spawn").Pool(nprocesses) as pool:
        hists = pool.map(CallJitCache(chain), keys, chunksize=chunksize)

    return jax.numpy.stack(hists)


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


def _clip_to_x_bounds(init_func, x_of_t, bounds, t_of_x):

    init_t = init_func()

    def init(key):
        t = init_t(key)
        x_clipped = jax.numpy.clip(x_of_t(t), *bounds.T)
        return t_of_x(x_clipped)

    return init


clip_to_x_bounds = partial_once(_clip_to_x_bounds)