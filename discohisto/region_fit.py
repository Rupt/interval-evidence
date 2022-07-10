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

        free = properties.free

        # other code (cabinetry...) modify the backend state;
        # save it to restore later
        backend, optimizer_old = pyhf.get_backend()

        pyhf.set_backend(backend, custom_optimizer="minuit")
        try:
            x_minuit = pyhf.infer.mle.fit(data, model)[free]
            fun_minuit = properties.objective_value(x_minuit)
        except pyhf.exceptions.FailedMinimization:
            x_minuit = None
            fun_minuit = -numpy.inf

        pyhf.set_backend(backend, custom_optimizer="scipy")
        try:
            x_scipy = pyhf.infer.mle.fit(data, model)[free]
            fun_minuit = properties.objective_value(x_minuit)
        except pyhf.exceptions.FailedMinimization:
            x_minuit = None
            fun_minuit = -numpy.inf

        # we are now later; restore
        pyhf.set_backend(backend, optimizer_old)

        # pick the best
        fun_minuit = properties.objective_value(x_minuit)
        fun_scipy = properties.objective_value(x_scipy)

        if fun_minuit < fun_scipy:
            self.x = numpy.array(x_minuit)
            self.fun = float(fun_minuit)
        else:
            self.x = numpy.array(x_scipy)
            self.fun = float(fun_scipy)
