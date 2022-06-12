import os
from dataclasses import asdict, dataclass
from typing import List

from tensorflow_probability.substrates import jax as tfp

from . import mcmc, serial

FILENAME = "mala.json"


def fit(
    region,
    nbins,
    range_,
    *,
    seed,
    nburnin=500,
    nsamples=100_000,
    nrepeats=100,
    step_size=0.5,
):
    def kernel_func(logdf):
        return tfp.mcmc.MetropolisAdjustedLangevinAlgorithm(
            logdf,
            step_size,
        )

    yields = mcmc.generic_fit(
        kernel_func,
        region,
        nbins,
        range_,
        seed=seed,
        nburnin=nburnin,
        nsamples=nsamples,
        nrepeats=nrepeats,
    )

    return FitMala(
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
    )


# serialization


@dataclass(frozen=True)
class FitMala:
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
    yields: List[List[int]]


def dump(fit: FitMala, path):
    os.makedirs(path, exist_ok=True)
    serial.dump_json_human(asdict(fit), os.path.join(path, FILENAME))


def load(path) -> FitMala:
    obj_json = serial.load_json(os.path.join(path, FILENAME))
    return FitMala(**obj_json)
