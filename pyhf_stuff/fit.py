"""Extract staistics from pyhf models."""
import weakref

import cabinetry
import jax

from . import blind

cabinetry  # TODO hides flake warning


def cabinetry_post(model):
    ...


def cabinetry_pre(model):
    ...


def normal(model):
    ...


def interval(model, level):
    ...


def linspace(model, start, stop, num):
    ...


# TODO cache eaach with WeakKeyDict?
_region_functions_cache = weakref.WeakKeyDictionary()


class RegionFunctions:
    """Store various jax-jitted functions to avoid recompilation."""

    def __init__(self, region):
        model, data = model_and_data(region)
        model_blind = blind.Model(model, {region.signal_region_name})

        # "logpdf" minimization objective
        @jax.value_and_grad
        def objective(x):
            (logpdf,) = model_blind.logpdf(x, data)
            return -logpdf

        def objective_value(x):
            value, _ = objective(x)
            return value

        def objective_grad(x):
            _, grad = objective(x)
            return grad

        # signal region yield
        slice_ = model.config.channel_slices[region.signal_region_name]

        @jax.value_and_grad
        def yield_(x):
            (result,) = model.expected_actualdata(x)[slice_]
            return result

        def yield_value(x):
            value, _ = yield_(x)
            return value

        def yield_grad(x):
            _, grad = yield_(x)
            return grad

        self.objective = jax.jit(objective)
        self.objective_value = jax.jit(objective_value)
        self.objective_grad = jax.jit(objective_grad)

        self.yield_ = jax.jit(yield_)
        self.yield_value = jax.jit(yield_value)
        self.yield_grad = jax.jit(yield_grad)


def model_and_data(region):
    model = region.workspace.model()
    data = region.workspace.data(model)
    return model, data
