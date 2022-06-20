"""Blind data in pyhf models."""
import numpy
import pyhf


class Model(pyhf.Model):
    """Wrapper around pyhf.Model with selected channel bins blinded."""

    def __init__(self, model, blind_channels):
        """
        Arguments:
            model: pyhf.pdf.Model-like
            blind_channels: sequence str channel_name
                (all bins are blinded)
        """
        self.batch_size = model.batch_size
        self.spec = model.spec
        self.schema = model.schema
        self.version = model.version
        self.config = model.config
        self.main_model = model.main_model
        self.constraint_model = model.constraint_model
        self.fullpdf_tv = model.fullpdf_tv

        self.blind_channels = set(blind_channels)

    def make_pdf(self, pars):
        # pdf ~ Simultaneous([Independent(Poisson), constraint])
        pdf = super().make_pdf(pars)
        main, constraint = pdf

        # Poisson rates are _private, it that they are the expected_data
        poisson_masked = PoissonMasked(
            main.expected_data(), _make_mask(self, self.blind_channels)
        )

        return pyhf.probability.Simultaneous(
            [
                pyhf.probability.Independent(poisson_masked),
                constraint,
            ],
            tensorview=pdf.tv,
            batch_size=pdf.batch_size,
        )


def _make_mask(model, blind_channels):
    """Return a mask to blind data in specified blind_channels."""
    channel_to_slice = model.config.channel_slices

    # the last slice ends at the number of channelbins
    ntot = next(reversed(channel_to_slice.values())).stop
    mask = numpy.ones(ntot, dtype=bool)

    for channel in blind_channels:
        slice_ = channel_to_slice[channel]
        mask[slice_] = False

    return mask


class PoissonMasked(pyhf.probability.Poisson):
    def __init__(self, rate, mask):
        super().__init__(rate)
        self.mask = mask

    def log_prob(self, value):
        return super().log_prob(value)[self.mask]
