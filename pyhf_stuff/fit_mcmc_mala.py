import os
from dataclasses import asdict, dataclass
from functools import partial
from typing import List

import numpy

from . import mcmc, mcmc_core, serial

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

    kernel_func = partial(mcmc_core.mala, step_size)

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

    yields, errors = mcmc_core.summarize_hists(hists)

    return FitMcmcMala(
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
class FitMcmcMala:
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

    def dump(self, path):
        os.makedirs(path, exist_ok=True)
        serial.dump_json_human(asdict(self), os.path.join(path, FILENAME))

    @classmethod
    def load(cls, path):
        obj_json = serial.load_json(os.path.join(path, FILENAME))
        return cls(**obj_json)
