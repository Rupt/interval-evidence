"""Efficiently derive properties of Region objects with caching."""
import jax
import numpy

from . import blind, region

# cache to avoid repeating expensive constructions (.model(), compilation)
PROPERTIES = "_properties"


def region_properties(region: region.Region):
    if PROPERTIES in region._cache:
        return region._cache[PROPERTIES]

    result = RegionProperties(region)
    region._cache[PROPERTIES] = result
    return result


class RegionProperties:
    """Parameters and jax-jitted functions for a region."""

    def __init__(self, region):
        model = region.workspace.model()
        # in context we always want to blind all bins of any signal region
        model_blind = blind.Model(model, [region.signal_region_name])
        data = region.workspace.data(model)

        # parameters

        def logdf(x):
            (logdf,) = model_blind.logpdf(x, data)
            return logdf

        # "logpdf" minimization objective
        @jax.value_and_grad
        def objective_value_and_grad(x):
            return -logdf(x)

        def objective_value(x):
            value, _ = objective_value_and_grad(x)
            return value

        def objective_grad(x):
            _, grad = objective_value_and_grad(x)
            return grad

        def objective_hess_inv(x):
            return jax.numpy.linalg.inv(jax.hessian(objective_value)(x))

        # signal region yield; only 1 bin is supported
        slice_ = model_blind.config.channel_slices[region.signal_region_name]
        assert slice_.stop == slice_.start + 1
        assert slice_.step is None
        index = slice_.start

        @jax.value_and_grad
        def yield_value_and_grad(x):
            return model_blind.expected_actualdata(x)[index]

        def yield_value(x):
            value, _ = yield_value_and_grad(x)
            return value

        def yield_grad(x):
            _, grad = yield_value_and_grad(x)
            return grad

        # basic properties
        self.model = model
        self.model_blind = model_blind
        self.data = numpy.array(data)
        self.init = numpy.array(model_blind.config.suggested_init())
        self.bounds = numpy.array(model_blind.config.suggested_bounds())
        self.index = index

        # constraint "logpdf" functions
        self.logdf = logdf
        self.objective_value_and_grad = jax.jit(objective_value_and_grad)
        self.objective_value = jax.jit(objective_value)
        self.objective_grad = jax.jit(objective_grad)
        self.objective_hess_inv = jax.jit(objective_hess_inv)

        # region expectation functions
        self.yield_value_and_grad = jax.jit(yield_value_and_grad)
        self.yield_value = jax.jit(yield_value)
        self.yield_grad = jax.jit(yield_grad)
