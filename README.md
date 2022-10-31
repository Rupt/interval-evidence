# ¡Research in progress!

# Likelihood intervals for ATLAS discovery regions

Where is my likelihood above this threshold?<br/>
_This project answers that question for likelihoods on signal yields
after integrating out background uncertainties._

Re-interpreting ATLAS results for single-bin signal-plus-background "discovery"
regions.
ATLAS amazingly publishes many of their complete models for these regions
in pyhf format through HEPData.

We use those models to assign uncertain predictions (priors)
in their single signal region bins.
To compute likelihoods on signal yields alone
(independent of background variations)
we marginalize the background uncertainty.

Our marginalization integrals average the Poisson likelihood on observed
data.
These integrals are computed by the `lebesgue` module, which bounds integrals
of general unimodal (likelihood) functions over general (prior) measures.

The `discohisto` module interprets the ATLAS models and uses `lebesgue` to
extract results.

## to do
- [x] Implement efficient integration.
- [x] Extract models from HEPData.
- [x] Uncertainty assignments: best fit
- [x] Uncertainty assignments: maximum likelihood profile
- [x] Uncertainty assignments: MCMC
- [x] Assign limits.
- [ ] Produce summary figures.
- [ ] Write a paper.
- [ ] Document usage scripts.


# Setup to use this software

Prepare or activate a virtual environment:

```bash
source setup.sh

```

## Development

Auto-format code:
```bash
make fmt

```

Run tests:
```bash
make test

```

Clean up:

```bash
make clean

```
