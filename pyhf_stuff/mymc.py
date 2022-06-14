"""An MCMC implementation using picklable jax code."""
from functools import partial

import jax

# partial is awesome


def partial_once(func):
    """Wrap func such that its first call returns a partial wrapper."""
    return partial(partial, func)


# chain execution


def _reduce_chain(
    kernel,
    reducer,
    initializer,
    *,
    nburnin,
    nsamples,
):
    kernel_init, kernel_step = kernel()
    reducte_init, reduce_ = reducer()
    init = initializer()

    def reduce_loop(i, state):
        args, reduction = state
        _, x, _ = args = kernel_step(*args)
        reduction = reduce_(reduction, x)
        return args, reduction

    def chain(key):
        key, key_init = jax.random.split(key)

        x = init(key_init)
        kernel_state = kernel_init(x)
        reduction = reducte_init(x)

        args = (key, x, kernel_state)

        # burnin discards its reduction
        args, _ = jax.lax.fori_loop(
            0,
            nburnin,
            reduce_loop,
            (args, reduction),
        )

        # main keeps only the reduction
        _, reduction = jax.lax.fori_loop(
            0,
            nsamples,
            reduce_loop,
            (args, reduction),
        )

        return reduction

    return chain


reduce_chain = partial_once(_reduce_chain)

# transition kernels


def _mala(step_size, logdf):
    def init(x):
        logf, logf_grad = jax.value_and_grad(logdf)(x)
        mean = x + 0.5 * step_size * logf_grad
        return logf, mean

    def step(key, x, state):
        logf, mean = state

        # propose the next step
        noise_to = step_size**0.5 * jax.random.normal(
            key, shape=x.shape, dtype=x.dtype
        )
        x_to = mean + noise_to
        logf_to, mean_to = state_to = init(x_to)

        # evaluate its acceptance ratio
        norm_to = noise_to.dot(noise_to)

        noise_from = x - mean_to
        norm_from = noise_from.dot(noise_from)

        log_accept = logf_to - logf + (0.5 / step_size) * (norm_to - norm_from)

        return x_to, state_to, log_accept

    return init, step


mala = partial_once(_mala)


def _metropolis(kernel):

    init, step_inner = kernel()

    def step(key, x, state):
        key, key_inner, key_accept = jax.random.split(key, 3)

        # inner step
        x_to, state_to, log_accept = step_inner(key_inner, x, state)

        # metropolis rule
        log_uniform = jax.numpy.log(
            jax.random.uniform(key_accept, dtype=log_accept.dtype)
        )
        accept = log_uniform <= log_accept

        # conditional move
        x, state = _tree_select(accept, (x_to, state_to), (x, state))

        return key, x, state

    return init, step


metropolis = partial_once(_metropolis)

# reductions


def _histogram(nbins, range_, func):
    def init(_):
        return jax.numpy.zeros(nbins, dtype=jax.numpy.int32)

    def reduce_(state, x):
        hist = state
        return _histogram_reduce(hist, func(x), range_)

    return init, reduce_


histogram = partial_once(_histogram)

# initializers


def _zeros(shape):
    def init(_):
        return jax.numpy.zeros(shape)

    return init


zeros = partial_once(_zeros)

# for multiprocessing


def _call_jit_call(func, arg):
    return jax.jit(func())(arg)


call_jit_call = partial_once(_call_jit_call)

# utility


def _histogram_reduce(hist, x, range_):
    # linearly find bin index
    bins = len(hist)
    lo, hi = range_
    bin_per_x = bins / (hi - lo)
    i_float = jax.numpy.floor((x - lo) * bin_per_x)
    i_clip = jax.numpy.clip(i_float, 0, bins - 1)

    # cast to integer for indexing
    index_type = jax.numpy.uint32
    assert bins <= jax.numpy.iinfo(index_type).max
    i = i_clip.astype(index_type)

    # histogramming; only add where in bounds (no under/overflow)
    return hist.at[i].add(i_float == i_clip)


def _tree_select(condition, tree_true, tree_false):
    return jax.tree_util.tree_map(
        lambda x, y: jax.lax.select(condition, x, y),
        tree_true,
        tree_false,
    )
