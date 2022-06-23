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
    ["--xla_cpu_multi_thread_eigen=false", "intra_op_parallelism_threads=1"]
)


# pyhf validation downloads schema files from the interweb every time objects
# of certain classes are __init__ialized.
# Not all of those classes offer options to disable that validation.
# This has been a source of sporadic crashes, and I want to work without an
# internet connection.
def no_validate_no_network_bullsh_t(*args, **kwargs):
    ...


pyhf.schema.validate = no_validate_no_network_bullsh_t
