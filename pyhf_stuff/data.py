"""Work with serialized data."""
import gzip
import json
import os

REGION_FILENAME = "region.json.gz"

# regions


def dump_region(path, signal_region_name, model, data):
    region_json = {
        "signal_region_name": signal_region_name,
        "model": model,
        "data": data,
    }
    os.makedirs(path, exist_ok=True)
    with gzip.open(os.path.join(path, REGION_FILENAME), "w") as file_:
        _json_dump_human(region_json, file_)


def load_region(path):
    with gzip.open(os.path.join(path, REGION_FILENAME), "w") as file_:
        region_json = json.load(file_)

    signal_region_name = region_json["signal_region_name"]
    model = region_json["model"]
    data = region_json["data"]
    return signal_region_name, model, data


# fits to regions
# TODO

# likelihoods from those fits
# TODO


def _json_dump_human(obj, fp, **kwargs):
    kwargs["sort_keys"] = False
    kwargs["indent"] = 4
    json.dump(obj, fp, **kwargs)
