"""Derive fit results for Region objects with caching."""
import numpy
import pyhf

from . import region
from .region_properties import region_properties

FIT_PRE = "_fit_pre"


def region_fit(region: region.Region):
    if FIT_PRE in region._cache:
        return region._cache[FIT_PRE]

    result = RegionFit(region)
    region._cache[FIT_PRE] = result
    return result


class RegionFit:
    def __init__(self, region, *, post=False):
        """Use pyhf to fit with different strategies, and keep the best."""
        properties = region_properties(region)

        data = properties.data
        if post:
            model = properties.model
        else:
            model = properties.model_blind

        # other code (cabinetry...) modify the backend state;
        # save it to restore later
        backend, optimizer_old = pyhf.get_backend()

        pyhf.set_backend(backend, custom_optimizer="minuit")
        x_raw_minuit = pyhf.infer.mle.fit(data, model)

        pyhf.set_backend(backend, custom_optimizer="scipy")
        x_raw_scipy = pyhf.infer.mle.fit(data, model)

        # we are now later; restore
        pyhf.set_backend(backend, optimizer_old)

        # trim to non-fixed values
        free = properties.free
        x_minuit = x_raw_minuit[free]
        x_scipy = x_raw_scipy[free]

        # pick the best
        fun_minuit = properties.objective_value(x_minuit)
        fun_scipy = properties.objective_value(x_scipy)

        if fun_minuit < fun_scipy:
            self.x = numpy.array(x_minuit)
            self.fun = float(fun_minuit)
        else:
            self.x = numpy.array(x_scipy)
            self.fun = float(fun_scipy)
