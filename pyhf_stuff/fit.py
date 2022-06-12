"""Extract staistics from pyhf models."""
import warnings
import weakref
from contextlib import contextmanager

import cabinetry
import jax
import numpy
import scipy
from tensorflow_probability.substrates import jax as tfp

from . import blind

# cabinetry


def cabinetry_post(region):
    model = region.workspace.model()
    data = region.workspace.data(model)
    return _cabinetry_fit(model, data, region.signal_region_name)


def cabinetry_pre(region):
    model = blind.Model(region.workspace.model(), {region.signal_region_name})
    data = region.workspace.data(model)
    return _cabinetry_fit(model, data, region.signal_region_name)


def _cabinetry_fit(model, data, signal_region_name):
    prediction = cabinetry.model_utils.prediction(
        model, fit_results=cabinetry.fit.fit(model, data)
    )

    index = model.config.channels.index(signal_region_name)
    yield_ = numpy.sum(prediction.model_yields[index])
    error = prediction.total_stdev_model_channels[index]
    return {
        "yield": yield_,
        "error": error,
    }


# expansion for Gaussian approximation


def normal(region):
    state = region_state(region)

    optimum = scipy.optimize.minimize(
        state.objective_value_and_grad,
        state.init,
        bounds=state.bounds,
        jac=True,
        method="L-BFGS-B",
    )

    # if gaussian, then covariance is inverse hessian
    cov = state.objective_hess_inv(optimum.x)

    # approximate signal region yield (log) linearly from gradients
    yield_value, yield_grad = state.yield_value_and_grad(optimum.x)

    yield_std = _quadratic_form(cov, yield_grad) ** 0.5

    # d/dx log(f(x)) = f'(x) / f(x)
    yield_log_std = _quadratic_form(cov, yield_grad / yield_value) ** 0.5

    # values are 0-dim jax arrays; cast to floats
    return {
        "linear": {
            "yield": float(yield_value),
            "error": float(yield_std),
        },
        "log": {
            "yield": float(numpy.log(yield_value)),
            "error": float(yield_log_std),
        },
    }


def _quadratic_form(matrix, vector):
    return matrix.dot(vector).dot(vector)


# levels of fixed values


def interval(region, *, levels=(0.5, 2, 4.5)):
    state = region_state(region)

    optimum = scipy.optimize.minimize(
        state.objective_value_and_grad,
        state.init,
        bounds=state.bounds,
        jac=True,
        method="L-BFGS-B",
    )

    def minmax_given_level(level):
        constaint = scipy.optimize.NonlinearConstraint(
            state.objective_value,
            optimum.fun + level,
            optimum.fun + level,
            jac=state.objective_grad,
        )

        with _suppress_bounds_warning():
            minimum = scipy.optimize.minimize(
                state.yield_value_and_grad,
                state.init,
                bounds=state.bounds,
                jac=True,
                method="SLSQP",
                constraints=constaint,
            )

        def negative_yield_value_and_grad(x):
            value, grad = state.yield_value_and_grad(x)
            return -value, -grad

        with _suppress_bounds_warning():
            maximum = scipy.optimize.minimize(
                negative_yield_value_and_grad,
                state.init,
                bounds=state.bounds,
                jac=True,
                method="SLSQP",
                constraints=constaint,
            )

        return minimum.fun, -maximum.fun

    levels = list(levels)

    intervals = [minmax_given_level(level) for level in levels]

    return {
        "levels": levels,
        "intervals": intervals,
    }


@contextmanager
def _suppress_bounds_warning():
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            (
                "Values in x were outside bounds during a "
                "minimize step, clipping to bounds"
            ),
            category=RuntimeWarning,
        )
        yield


# linear scan, like numpy.linspace


def linspace(region, start, stop, num):
    state = region_state(region)

    def optimum_given_yield(yield_):
        constaint = scipy.optimize.NonlinearConstraint(
            state.yield_value,
            yield_,
            yield_,
            jac=state.yield_grad,
        )

        optimum = scipy.optimize.minimize(
            state.objective_value_and_grad,
            state.init,
            bounds=state.bounds,
            jac=True,
            method="SLSQP",
            constraints=constaint,
        )

        return optimum.fun

    levels = [
        optimum_given_yield(yield_)
        for yield_ in numpy.linspace(start, stop, num)
    ]

    return {
        "start": start,
        "stop": stop,
        "levels": levels,
    }


# MCMC


def mcmc_nuts(
    region, nsamples=100, *, seed=0, step_size=1e-3, burnin=500, thin=100
):
    state = region_state(region)

    # TODO IMPLEMENT BOUNDS
    # exclude samples? set logpdf to 0?

    # sample
    @jax.jit
    def chain(key, chain_state):
        nuts = tfp.mcmc.NoUTurnSampler(state.logpdf, step_size)
        nuts = tfp.mcmc.SimpleStepSizeAdaptation(nuts, int(burnin * 0.8))
        states, trace = tfp.mcmc.sample_chain(
            nsamples,
            current_state=chain_state,
            kernel=nuts,
            trace_fn=lambda x, _: state.yield_value(x),
            num_burnin_steps=burnin,
            num_steps_between_results=thin,
            seed=key,
        )
        return jax.numpy.array(trace)

    yields = chain(jax.random.PRNGKey(seed), state.init)

    print(tfp.mcmc.effective_sample_size(yields))
    print(yields.mean(), yields.std())

    return {
        "seed": seed,
        "step_size": step_size,
        "burnin": burnin,
        "thin": thin,
        "yields": yields.tolist(),
    }


def mcmc_hmc(
    region, nsamples=100, *, seed=0, step_size=5e-3, burnin=100, thin=1
):
    state = region_state(region)

    # TODO IMPLEMENT BOUNDS
    # exclude samples? set logpdf to 0?

    # sample
    @jax.jit
    def chain(key, chain_state):
        adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
            tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn=state.logpdf,
                num_leapfrog_steps=3,
                step_size=step_size,
            ),
            num_adaptation_steps=int(burnin * 0.8),
        )

        states, trace = tfp.mcmc.sample_chain(
            nsamples,
            current_state=chain_state,
            kernel=adaptive_hmc,
            # trace_fn=lambda x, _: state.yield_value(x),
            trace_fn=lambda x, pkr: (
                state.yield_value(x),
                pkr.inner_results.log_accept_ratio,
            ),
            num_burnin_steps=burnin,
            num_steps_between_results=thin,
            seed=key,
        )
        return states, jax.numpy.array(trace)

    states, (yields, is_accepted) = chain(jax.random.PRNGKey(seed), state.init)
    print(states.shape)
    print(yields.shape)
    print(is_accepted)
    # print((yields[:-1] == yields[1:]).astype(float))

    print(tfp.mcmc.effective_sample_size(yields))
    print(yields.mean(), yields.std())

    return {
        "seed": seed,
        "step_size": step_size,
        "burnin": burnin,
        "thin": thin,
        "yields": yields.tolist(),
    }


# region functions

# cache to avoid recompilation, weakref to clean up if the region object no
# longer exists elsewhere
# TODO make this a lazy property on region object
_region_functions_cache = weakref.WeakKeyDictionary()


def region_state(region):
    if region in _region_functions_cache:
        return _region_functions_cache[region]

    result = RegionState(region)
    _region_functions_cache[region] = result
    return result


class RegionState:
    """Parameters and jax-jitted functions for a region."""

    def __init__(self, region):
        model = region.workspace.model()
        data = region.workspace.data(model)
        model_blind = blind.Model(model, {region.signal_region_name})

        # parameters
        self.init = numpy.array(model.config.suggested_init())
        self.bounds = numpy.array(model.config.suggested_bounds())

        def logpdf(x):
            (logpdf,) = model_blind.logpdf(x, data)
            return logpdf

        self.logpdf = logpdf

        # "logpdf" minimization objective
        @jax.value_and_grad
        def objective_value_and_grad(x):
            return -logpdf(x)

        def objective_value(x):
            value, _ = objective_value_and_grad(x)
            return value

        def objective_grad(x):
            _, grad = objective_value_and_grad(x)
            return grad

        def objective_hess_inv(x):
            return jax.numpy.linalg.inv(jax.hessian(objective_value)(x))

        # signal region yield
        slice_ = model.config.channel_slices[region.signal_region_name]

        @jax.value_and_grad
        def yield_value_and_grad(x):
            (result,) = model.expected_actualdata(x)[slice_]
            return result

        def yield_value(x):
            value, _ = yield_value_and_grad(x)
            return value

        def yield_grad(x):
            _, grad = yield_value_and_grad(x)
            return grad

        self.objective_value_and_grad = jax.jit(objective_value_and_grad)
        self.objective_value = jax.jit(objective_value)
        self.objective_grad = jax.jit(objective_grad)
        self.objective_hess_inv = jax.jit(objective_hess_inv)

        self.yield_value_and_grad = jax.jit(yield_value_and_grad)
        self.yield_value = jax.jit(yield_value)
        self.yield_grad = jax.jit(yield_grad)


# utilities


def filename(func):
    return func.__name__ + ".json"
