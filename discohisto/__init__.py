import logging
import os

import jax
import pyhf
from pyhf.optimize.mixins import log as _pyhf_optimize_mixins_log

# https://github.com/google/jax/issues/6790
# avoid internal threading to give the user control over multiprocessing
# in experiments this seems to make no performance difference for our use case
# https://github.com/google/jax/issues/5506#issuecomment-766998022
# allow multiprocessing with pmap
_NPROCESSES = int(os.environ.get("NPROCESSES", "1"))
os.environ["XLA_FLAGS"] = " ".join(
    [
        "--xla_force_host_platform_device_count=%d" % _NPROCESSES,
        "--xla_cpu_multi_thread_eigen=false",
        "intra_op_parallelism_threads=1",
    ]
)

# GPU might be powerful, but I want CPU portability
jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

# pyhf

pyhf.set_backend("jax")


def _no_validate_no_network(*args, **kwargs):
    # pyhf validation downloads schema files from the interweb every time
    # objects of certain classes are __init__ialized.
    # Not all of those classes offer options to disable that validation.
    # This has been a source of sporadic crashes, and I want to work without an
    # internet connection.
    ...


pyhf.schema.validate = _no_validate_no_network


# pyhf screams into this logger when its fits fail

_pyhf_optimize_mixins_log.setLevel(logging.CRITICAL)
