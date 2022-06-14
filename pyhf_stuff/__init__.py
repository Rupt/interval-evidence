import os

import jax
import pyhf

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)
pyhf.set_backend("jax")

# https://github.com/google/jax/issues/6790
# avoid fighting between processes
os.environ[
    "XLA_FLAGS"
] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
