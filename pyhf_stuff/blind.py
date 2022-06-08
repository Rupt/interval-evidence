"""Blind data in pyhf models."""
import numpy
import pyhf


def model_logpdf_blind(model, pars, data, blind_bins):
    """Return a "logpdf" value with blinded channel-bins.

    Args:
        model: pyhf.pdf.Model-like
        pars (:obj:`tensor`): The parameter values
        data (:obj:`tensor`): The measurement data
        blinded_channelbins: Sequence of either
            pair (channel_name, bin_index)
            or str channel_name.
            str channel_name requires that the channel has one bin only.

    Returns:
        Tensor: The "log density" value

    """
    mask = _make_mask(model, blind_bins)
    return _model_logpdf_masked(model, pars, data, mask)


class Model:
    """Wrapper around pyhf.Model with regions blinded."""

    def __init__(self, model: pyhf.Model, blind_bins):
        self.model = model
        self.blind_bins = blind_bins

    def logpdf(self, pars, data):
        return model_logpdf_blind(self.model, pars, data, self.blind_bins)

    # forward failed attributes to the pyhf.Model
    def __getattr__(self, name):
        return self.model.__getattribute__(name)


def _make_mask(model, blind_bins):
    """Return a mask to blind data in specified blind_bins."""
    channel_to_slice = model.config.channel_slices

    # the last slice is the number of channelbins
    ntot = next(reversed(channel_to_slice.values())).stop
    mask = numpy.ones(ntot, dtype=bool)

    for channelbin in blind_bins:
        str_form = isinstance(channelbin, str)
        if str_form:
            channelbin = (channelbin, 0)

        channel, bin_ = channelbin

        slice_ = channel_to_slice[channel]
        assert slice_.step is None
        slice_range = range(slice_.start, slice_.stop)

        nbins = len(slice_range)
        if str_form and nbins != 1:
            raise ValueError(
                f"bin index is needed for channel {channel} with {nbins=}"
            )

        i = slice_range[bin_]
        mask[i] = False

    return mask


def _model_logpdf_masked(model, pars, data, mask):
    # modified from pyhf.pdf.Model.logpdf
    tensorlib, _ = pyhf.get_backend()
    pars, data = tensorlib.astensor(pars), tensorlib.astensor(data)

    # Verify parameter and data shapes
    if pars.shape[-1] != model.config.npars:
        raise ValueError(
            f"pars has len {pars.shape[-1]} but "
            f"{model.config.npars} was expected"
        )

    len_actualdata = model.nominal_rates.shape[-1]
    len_auxdata = len(model.config.auxdata)
    if data.shape[-1] != len_actualdata + len_auxdata:
        raise ValueError(
            f"data has len {data.shape[-1]} but "
            f"{model.config.nmaindata + model.config.nauxdata} was expected"
        )

    pdf = _model_make_pdf_masked(model, pars, mask)

    actualdata = data[:len_actualdata]
    auxdata = data[len_actualdata:]

    actualdata_masked = tensorlib.where(mask, actualdata, 0)
    data_masked = tensorlib.concatenate([actualdata_masked, auxdata])

    result = pdf.log_prob(data_masked)

    if model.batch_size:
        return result

    return tensorlib.reshape(result, (1,))


def _model_make_pdf_masked(model, pars, mask):
    tensorlib, _ = pyhf.get_backend()

    pdfobjs = []

    mainpdf = _main_model_make_pdf_masked(model.main_model, pars, mask)
    if mainpdf:
        pdfobjs.append(mainpdf)

    constraintpdf = model.constraint_model.make_pdf(pars)
    if constraintpdf:
        pdfobjs.append(constraintpdf)

    return pyhf.probability.Simultaneous(
        pdfobjs, model.fullpdf_tv, model.batch_size
    )


def _main_model_make_pdf_masked(main_model, pars, mask):
    tensorlib, _ = pyhf.get_backend()

    lambdas_data = main_model.expected_data(pars)

    # pyhf nans for poisson(0 | 0.0), so settle for a tiny mean.
    # This makes a constant addition to logpdf, so makes no difference
    # to derivative-based or maximum-relative fits.
    # Furthermore, only very small values will be changed by adding tiny, so
    # in typical cases its impact is exactly zero.
    tiny = numpy.finfo(lambdas_data.dtype).tiny
    lambdas_blinded = tensorlib.where(mask, lambdas_data, tiny)

    return pyhf.probability.Independent(
        pyhf.probability.Poisson(lambdas_blinded)
    )
