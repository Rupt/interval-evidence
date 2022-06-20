"""Efficiently derive properties of Region objects with caching."""
import weakref

import jax
import numpy

from . import blind, region

# cache to avoid repeating expensive constructions (.model(), compilation)
# weakref to clean up if the region object no longer exists elsewhere
_region_properties_cache = weakref.WeakKeyDictionary()


def region_properties(region: region.Region):
    if region in _region_properties_cache:
        return _region_properties_cache[region]

    result = RegionProperties(region)
    _region_properties_cache[region] = result
    return result


class RegionProperties:
    """Parameters and jax-jitted functions for a region."""

    def __init__(self, region):
        model = region.workspace.model()
        model_blind = blind.Model(model, {region.signal_region_name})
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

        # signal region yield
        slice_channel = model_blind.config.channel_slices[
            region.signal_region_name
        ]
        assert slice_channel.step is None
        slice_start = slice_channel.start + region.signal_region_bin
        assert slice_start < slice_channel.stop
        slice_ = slice(slice_start, slice_start + 1)

        @jax.value_and_grad
        def yield_value_and_grad(x):
            (result,) = model_blind.expected_actualdata(x)[slice_]
            return result

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
        self.slice_ = slice_

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
