"""Extract staistics from pyhf models."""
import weakref

import cabinetry
import jax

from . import blind

cabinetry  # TODO hides flake warning


def cabinetry_post(region):
    model = region.workspace.model()
    data = region.workspace.data(model)
    ...


def cabinetry_pre(region):
    data = region.workspace.data(model)
    model = blind.Model(region.workspace.model(), {region.signal_region_name})

    fit_result = cabinetry.fit.fit(model, data)

    prediction = cabinetry.model_utils.prediction(model, fit_result)

    # TODO find index of signal region
    ...


def normal(region):
    funcs = region_functions(region)
    funcs.objective
    ...


def interval(region, level):
    funcs = region_functions(region)
    funcs.objective
    ...


def linspace(region, start, stop, num):
    funcs = region_functions(region)
    funcs.objective
    ...


# region functions

_region_functions_cache = weakref.WeakKeyDictionary()


def region_functions(region):
    if region in _region_functions_cache:
        return _region_functions_cache[region]

    result = RegionFunctions(region)
    _region_functions_cache[region] = result
    return result


class RegionFunctions:
    """Store various jax-jitted functions to avoid recompilation."""

    def __init__(self, region):
        model = region.workspace.model()
        data = region.workspace.data(model)
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
