"""Extract statistics with Markov Chain Monte Carlo (MCMC)."""
import jax
from tensorflow_probability.substrates import jax as tfp

from .mcmc_core import _boundary, _histogram
from .region_fit import region_fit
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
    optimum_x = region_fit(region).x

    cov = properties.objective_hess_inv(optimum_x)

    x_of_t, t_of_x = eye_covariance_transform(optimum_x, cov)

    init = jax.numpy.zeros_like(properties.init)

    def logdf(t):
        x = x_of_t(t)
        return properties.logdf(x) + _boundary(x, properties.bounds)

    def observable(t):
        return properties.yield_value(x_of_t(t))

    # mcmc chain sapling
    def chain(key):
        _, yields = tfp.mcmc.sample_chain(
            kernel=kernel_func(logdf),
            trace_fn=lambda t, _: observable(t),
            current_state=init,
            num_burnin_steps=nburnin,
            num_results=nsamples,
            seed=key,
        )
        return _histogram(yields, nbins, range_)

    keys = jax.random.split(jax.random.PRNGKey(seed), nrepeats)
    return jax.jit(jax.vmap(chain))(keys)


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
