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


# Use this software

Setup: prepare and activate our virtual environment:

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

## Reproduce results

This software goes to some length to unsure that the results are fully
reproducible on hardware.

However, the compilation phases are liable to generate semantically different
floating-point calculations when optimizing for different machines.

Different numerical errors do not change the conclusions, but do
accumulate to produce different Markov chains.

### Prepare `pyhf` models and extract likelihood intervals:
<details>
<summary>Click to expand shell script</summary>

```sh
SEARCHES='
atlas_susy_1Lbb_2020
atlas_susy_1Ljets_2021
atlas_susy_2hadtau_2020
atlas_susy_2L0J_2019
atlas_susy_2Ljets_2022
atlas_susy_3L_2021
atlas_susy_3Lresonance_2020
atlas_susy_3LRJmimic_2020
atlas_susy_3Lss_2019
atlas_susy_4L_2021
atlas_susy_compressed_2020
atlas_susy_DVmuon_2020
atlas_susy_hb_2019
atlas_susy_jets_2021
'

# Download and prepare HEPData serialized workspaces.
for search in ${SEARCHES}
do
    ./searches/${search}/make_workspaces.sh
done

# Extract single-region workspaces for our discovery regions.
# Outputs searches/*/*/region.json.gz
for search in ${SEARCHES}
do
    python searches/${search}/dump_regions.py
done

# Fits extract properties of the pyhf models from which we later assign
# background predictions with uncertainty.

# Extract fits with best fitting strategies (cabinetry, normal, profile)
# Outputs searches/*/*/fit/{normal,cabinetry,cabinetry_post,linspace}.json
for search in ${SEARCHES}
do
    { time python searches/${search}/dump_fits.py ; } |& tee log/dump_fits_${search}.log
done

# Extract maxima with constant added signal yields.
# (This is more similar to current practice, but not so relevant.)
# Outputs searches/*/*/fit/signal.json
for search in ${SEARCHES}
do
    { time python searches/${search}/dump_fit_signal.py ; } |& tee log/dump_fit_signal_${search}.log
done

# Extract histograms with MCMC (Markov Chain Monte Carlo) sampling. (SLOW!)
# We use up to ${NPROCESSES} CPU threads with Jax pmap parallelism.
# Outputs searches/*/*/fit/mcmc_*.json
# where * depends on the MCMC strategy employed.
export NPROCESSES=15
for search in ${SEARCHES}
do
    { time python searches/${search}/dump_fit_mcmc.py ; } |& tee log/dump_fit_mcmc_${search}.log
done

# Assign probabilities from the fits and data in models and scan marginal
# likelihoods and assign upper limits at various levels.
# Outputs searches/*/*/fit/limit/scan_*_{central,up,down,observed}.json
# where * names the fit used and {central,up,down,observed} are variations on
# the data. Observed is integer Poisson, others are real interpolated Poisson
# using lebesgue.likelihood.gamma1.
for search in ${SEARCHES}
do
    { time python searches/${search}/dump_limits.py ; } |& tee log/dump_limits_${search}.log
done

```
</details>
