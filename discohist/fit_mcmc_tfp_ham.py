import os
from dataclasses import asdict, dataclass

import numpy
from tensorflow_probability.substrates import jax as tfp

from . import mcmc_core, mcmc_tfp, serial


def fit(
    region,
    nbins,
    range_,
    *,
    seed,
    nburnin=100,
    nsamples=20_000,
    nrepeats=100,
    step_size=0.5,
    num_leapfrog_steps=3,
):
    range_ = numpy.array(range_, dtype=float).tolist()

    def kernel_func(logdf):
        return tfp.mcmc.HamiltonianMonteCarlo(
            logdf,
            step_size,
            num_leapfrog_steps,
        )

    hists = mcmc_tfp.generic_chain_hist(
        kernel_func,
        region,
        nbins,
        range_,
        seed=seed,
        nburnin=nburnin,
        nsamples=nsamples,
        nrepeats=nrepeats,
    )

    yields, errors = mcmc_core.summarize_hists(hists)

    return FitMcmcTfpHam(
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
        num_leapfrog_steps=num_leapfrog_steps,
        # results
        yields=yields.tolist(),
        errors=errors.tolist(),
    )


# serialization


@dataclass(frozen=True)
class FitMcmcTfpHam:
    # histogram arguments
    nbins: int
    range_: list[float]
    # generic arguments
    nburnin: int
    nsamples: int
    nrepeats: int
    seed: int
    # special arguments
    step_size: float
    num_leapfrog_steps: int
    # results
    yields: list[int]
    errors: list[float]

    filename = "mcmc_tfp_ham"

    def dump(self, path, *, suffix=""):
        os.makedirs(path, exist_ok=True)
        filename = self.filename + suffix + ".json"
        serial.dump_json_human(asdict(self), os.path.join(path, filename))

    @classmethod
    def load(cls, path, *, suffix=""):
        filename = cls.filename + suffix + ".json"
        obj_json = serial.load_json(os.path.join(path, filename))
        return cls(**obj_json)
