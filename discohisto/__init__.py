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

# pyhf screams into this logger when its fits fail
_pyhf_optimize_mixins_log.setLevel(logging.CRITICAL)
