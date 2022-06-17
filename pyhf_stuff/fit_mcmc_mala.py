import os
from dataclasses import asdict, dataclass
from functools import partial
from typing import List

import numpy

from . import mcmc, mcmc_jax, mcmc_tfp, serial

FILENAME = "mcmc_mala.json"
DEFAULT_NPROCESSES = os.cpu_count() // 2


def fit(
    region,
    nbins,
    range_,
    *,
    seed,
    nburnin=100,
    nsamples=20_000,
    nrepeats=10,
    step_size=0.5,
    nprocesses=DEFAULT_NPROCESSES,
):
    range_ = numpy.array(range_, dtype=float).tolist()

    kernel_func = partial(mcmc_jax.mala, step_size)

    hists = mcmc.region_hist_chain(
        kernel_func,
        region,
        nbins,
        range_,
        seed=seed,
        nburnin=nburnin,
        nsamples=nsamples,
        nrepeats=nrepeats,
        nprocesses=nprocesses,
    )

    hists = numpy.array(hists)

    yields, errors = mcmc_tfp._summarize_hists(hists)

    return FitMala2(
        # histogram arguments
        nbins=nbins,
        range_=range_,
        # generic arguments
        nburnin=nburnin,
        nsamples=nsamples,
        nrepeats=nrepeats,
        seed=seed,
        # special arguments
        step_size=step_size,
        # results
        yields=yields.tolist(),
        errors=errors.tolist(),
    )


# serialization


@dataclass(frozen=True)
class FitMala2:
    # histogram arguments
    nbins: int
    range_: List[float]
    # generic arguments
    nburnin: int
    nsamples: int
    nrepeats: int
    seed: int
    # special arguments
    step_size: float
    # results
    yields: List[int]
    errors: List[float]


def dump(fit: FitMala2, path):
    os.makedirs(path, exist_ok=True)
    serial.dump_json_human(asdict(fit), os.path.join(path, FILENAME))


def load(path) -> FitMala2:
    obj_json = serial.load_json(os.path.join(path, FILENAME))
    return FitMala2(**obj_json)
