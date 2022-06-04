from test import raises

import numpy
import pyhf

from .blind import model_logpdf_blind


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
