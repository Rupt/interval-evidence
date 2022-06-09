"""Work with serialized data."""
import gzip
import json


def dump_json_human(obj, path):
    with open(path, "w") as file_:
        json.dump(obj, file_, indent=4)


def dump_json_gz(obj, path):
    # gzip adds a timestamp which makes git think identical files have changed
    # Avoid that by setting that timestamp to a constant value
    file_ = gzip.GzipFile(path, "w", mtime=0)
    file_.write(json.dumps(obj).encode())
    file_.close()


def load_json_gz(path):
    with gzip.open(path, "r") as file_:
        spec = json.loads(file_.read().decode())
    return spec
