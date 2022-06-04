"""Serialize and load json files."""
import gzip
import json


def dump(inpath, outpath):
    spec = json.load(open(inpath))
    with gzip.open(outpath, "w") as outfile:
        outfile.write(json.dumps(spec).encode())


def load(inpath):
    with gzip.open(inpath, "r") as infile:
        spec = json.loads(infile.read().decode())
    return spec


if __name__ == "__main__":
    import sys

    # example: python jsongz.py ins1852821_bkg.json ins1852821_bkg.json.gz
    dump(*sys.argv[1:])
