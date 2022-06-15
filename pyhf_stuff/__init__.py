import os

import jax
import pyhf

# GPU might be powerful, but I want CPU portability
jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

pyhf.set_backend("jax")

# https://github.com/google/jax/issues/6790
# avoid internal threading to give the user control over multiprocessing
# in experiments this seems to make no performance difference for our use case
os.environ["XLA_FLAGS"] = " ".join(
    "--xla_cpu_multi_thread_eigen=false", "intra_op_parallelism_threads=1"
)
