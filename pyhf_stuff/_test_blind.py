from test import raises

import numpy
import pyhf

from .blind import Model, _make_mask, model_logpdf_blind


def test_simple_model_blind():
    model = pyhf.simplemodels.uncorrelated_background(
        signal=[1.0], bkg=[55.0], bkg_uncertainty=[7.0]
    )

    nobs = 54

    parameters = model.config.suggested_init()
    data = numpy.concatenate([[nobs], model.expected_data(parameters)[1:]])
    channel_name = model.config.channels[0]

    def f():
        return model.logpdf(parameters, data)

    def f_blind(blind_bins):
        return model_logpdf_blind(model, blind_bins, parameters, data)

    # check we recover the same result with nothing blinded
    assert f() == f_blind([])

    # check that constraint + loglikelihood recovers total
    expected_data = model.make_pdf(
        pyhf.tensorlib.astensor(parameters)
    ).expected_data()
    slice_ = model.config.channel_slices[channel_name]
    mu = expected_data[slice_]

    loglikelihood = pyhf.probability.Poisson(mu).log_prob(nobs)
    assert f() == f_blind({channel_name}) + loglikelihood

    # check (channel, bin) form
    assert f() == f_blind({(channel_name, 0)}) + loglikelihood


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
        bins_blind,
        parameters,
        data,
    )

    assert isinstance(model_blind, pyhf.Model)
    assert model_blind.batch_size == model.batch_size
    assert model_blind.schema == model.schema
    assert model_blind.spec == model.spec
    assert model_blind.version == model.version

    channel_name = model.config.channels[0]
    assert raises(lambda: _make_mask(model, {(channel_name, 1)}), IndexError)
    assert raises(lambda: _make_mask(model, {("foo", 0)}), KeyError)
    assert raises(lambda: _make_mask(model, {"foo"}), KeyError)
