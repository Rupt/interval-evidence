from . import _cache, likelihood, prior
from ._bayes import Likelihood, Model, Prior

# importing _cache prepares some compiled code; discard its reference
del _cache
