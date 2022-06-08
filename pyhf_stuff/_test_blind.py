from test import raises

import numpy
import pyhf

from .blind import Model, model_logpdf_blind


def test_simple_model_blind():
    model = pyhf.simplemodels.uncorrelated_background(
        signal=[1.0], bkg=[55.0], bkg_uncertainty=[7.0]
    )

    nobs = 54

    data = numpy.concatenate([[nobs], model.config.auxdata])
    parameters = model.config.suggested_init()
    channel_name = model.config.channels[0]

    def logf():
        (x,) = model.logpdf(parameters, data)
        return x

    def logf_blind(blind_bins):
        (x,) = model_logpdf_blind(model, parameters, data, blind_bins)
        return x

    # check we recover the same result with nothing blinded
    assert logf() == logf_blind([])

    # check that constraint + loglikelihood recovers total
    expected_data = model.make_pdf(numpy.array(parameters)).expected_data()
    slice_ = model.config.channel_slices[channel_name]
    mu = expected_data[slice_]

    loglikelihood = pyhf.probability.Poisson(mu).log_prob(nobs)

    assert logf() == logf_blind({channel_name}) + loglikelihood

    # check (channel, bin) form
    assert logf() == logf_blind({(channel_name, 0)}) + loglikelihood

    assert raises(lambda: logf_blind({(channel_name, 1)}), IndexError)
    assert raises(lambda: logf_blind({("foo", 0)}), KeyError)
    assert raises(lambda: logf_blind({"foo"}), KeyError)


def test_model():
    model = pyhf.simplemodels.uncorrelated_background(
        signal=[0.0], bkg=[3.0], bkg_uncertainty=[1.0]
    )
    nobs = 4

    bins_blind = {model.config.channels[0]}
    model_blind = Model(model, bins_blind)

    data = numpy.concatenate([[nobs], model.config.auxdata])
    parameters = model.config.suggested_init()

    assert model_blind.logpdf(parameters, data) == model_logpdf_blind(
        model,
        parameters,
        data,
        bins_blind,
    )

    # I don't know why some of these aren't "is"
    assert model_blind.batch_size is model.batch_size
    assert model_blind.config is model.config
    assert model_blind.constraint_logpdf == model.constraint_logpdf
    assert model_blind.constraint_model == model.constraint_model
    assert model_blind.expected_actualdata == model.expected_actualdata
    assert model_blind.expected_auxdata == model.expected_auxdata
    assert model_blind.expected_data == model.expected_data
    assert model_blind.fullpdf_tv is model.fullpdf_tv
    assert model_blind.main_model is model.main_model
    assert model_blind.mainlogpdf == model.mainlogpdf
    assert model_blind.make_pdf == model.make_pdf
    assert model_blind.nominal_rates is model.nominal_rates
    assert model_blind.pdf == model.pdf
    assert model_blind.schema is model.schema
    assert model_blind.spec is model.spec
    assert model_blind.version is model.version
