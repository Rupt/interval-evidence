"""Serialize and load pyhf json spec files."""
import json
import sys

import serial


def main():
    assert (
        len(sys.argv) == 3
    ), "example: python jsongz.py ins1852821_bkg.json ins1852821_bkg.json.gz"

    inpath, outpath = sys.argv[1:]
    spec = json.load(open(inpath))
    serial.dump_json_gz(spec, outpath)


if __name__ == "__main__":
    main()
