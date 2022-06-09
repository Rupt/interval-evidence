"""Blind data in pyhf models."""
import numpy
import pyhf


def model_logpdf_blind(model, blind_bins, pars, data):
    """Return a "logpdf" value with blinded channel-bins.

    Args:
        model: pyhf.pdf.Model-like
        blind_bins: dataclass of either
            pair (channel_name, bin_index)
            or str channel_name.
            str channel_name requires that the channel has one bin only.
        pars (:obj:`tensor`): The parameter values
        data (:obj:`tensor`): The measurement data

    Returns:
        Tensor: The "log density" value

    """
    return Model(model, blind_bins).logpdf(pars, data)


class Model:
    """Wrapper around pyhf.Model with selected channel bins blinded."""

    def __init__(self, model, blind_bins):
        self.model = model
        self.blind_bins = blind_bins

    def make_pdf(self, pars):
        pdf = self.model.make_pdf(pars)
        # pdf ~ Simultaneous([Independent(Poisson), constraint])
        main, constraint = pdf

        mask = _make_mask(self.model, self.blind_bins)
        poisson_masked = PoissonMasked(main.expected_data(), mask)

        return pyhf.probability.Simultaneous(
            [
                pyhf.probability.Independent(poisson_masked),
                constraint,
            ],
            tensorview=pdf.tv,
            batch_size=pdf.batch_size,
        )

    logpdf = pyhf.Model.logpdf

    # forward missing attributes to the pyhf.Model
    def __getattr__(self, name):
        return self.model.__getattribute__(name)


def _make_mask(model, blind_bins):
    """Return a mask to blind data in specified blind_bins."""
    channel_to_slice = model.config.channel_slices

    # the last slice ends at the number of channelbins
    ntot = next(reversed(channel_to_slice.values())).stop
    mask = numpy.ones(ntot, dtype=bool)

    for channelbin in blind_bins:
        str_form = isinstance(channelbin, str)
        if str_form:
            channelbin = (channelbin, 0)

        channel, bin_ = channelbin

        mask_slice = mask[channel_to_slice[channel]]

        nbins = len(mask_slice)
        if str_form and nbins != 1:
            raise ValueError(f"missing bin index: {channel=} with {nbins=}")

        mask_slice[bin_] = False

    return mask


class PoissonMasked(pyhf.probability.Poisson):
    def __init__(self, rate, mask):
        super().__init__(rate)
        self.mask = mask

    def log_prob(self, value):
        return super().log_prob(value)[self.mask]
