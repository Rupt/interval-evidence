import os

import pyhf

from pyhf_stuff import region, serial

BASEPATH = os.path.dirname(__file__)
HEPDATA = "ins1852821"


def main():
    spec = serial.load_json_gz(
        os.path.join(BASEPATH, HEPDATA + "_bkg.json.gz")
    )
    workspace = pyhf.workspace.Workspace(region.clear_poi(spec))

    channels = workspace.channel_slices.keys()

    # find control and signals region strings
    control_regions = set()
    signal_regions = set()
    for name in channels:
        if name.startswith("CR"):
            control_regions.add(name)
            continue

        # ins1852821 modifies its control regions to exclude 5-lepton events
        # but we don't have specs for those control regions
        if name.startswith("SR5L"):
            continue

        assert name.startswith("SR")
        signal_regions.add(name)

    # serialize regions for each
    for name in sorted(signal_regions):
        signal_region_name = region.strip_cuts(name)
        region.dump(
            signal_region_name,
            region.prune(workspace, name, *control_regions),
            os.path.join(BASEPATH, signal_region_name),
        )


if __name__ == "__main__":
    main()
