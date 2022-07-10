"""Extract statistics with mcmc from tensorflow_probability."""
import jax
from tensorflow_probability.substrates import jax as tfp

from .mcmc import logdf_template, yields_template
from .mcmc_core import _histogram, _stable_pmap, eye_covariance_transform
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

    logdf = logdf_template(region, x_of_t)()
    observable = yields_template(region, x_of_t)()

    init = jax.numpy.zeros_like(optimum_x)

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
    return _stable_pmap(chain, keys)
