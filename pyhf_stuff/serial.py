"""Work with serialized data."""
import gzip
import json
import os

SIGNAL_REGION_NAME = "__signal_region__"
REGION_FILENAME = "region.json.gz"

# regions


def dump_region(path, workspace):
    os.makedirs(path, exist_ok=True)

    outpath = os.path.join(path, REGION_FILENAME)

    # gzip adds a timestamp which makes git think identical files have changed
    # Avoid that by setting that timestamp to a constant value
    outfile = gzip.GzipFile(outpath, "w", mtime=0)
    outfile.write(json.dumps(workspace).encode())
    outfile.close()


def load_region(path):
    inpath = os.path.join(path, REGION_FILENAME)
    with gzip.open(inpath, "r") as infile:
        workspace = json.loads(infile.read().decode())
    return workspace


# fits to regions
# TODO

# likelihoods from those fits
# TODO


def _json_dump_human(obj, fp, **kwargs):
    kwargs["sort_keys"] = False
    kwargs["indent"] = 4
    json.dump(obj, fp, **kwargs)
