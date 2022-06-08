import os

import jsongz
import pyhf
import serial

BASEPATH = os.path.dirname(__file__)
PAPER = "ins1852821"

# TODO standardize standardizable parts


def main():
    spec = jsongz.load(os.path.join(BASEPATH, PAPER + "_bkg.json.gz"))
    workspace = pyhf.workspace.Workspace(clear_poi(spec))

    channel_names = workspace.channel_slices.keys()

    # find control and signals region strings
    control_region_names = set()
    signal_region_names = set()
    for channel_name in channel_names:
        if channel_name.startswith("CR"):
            control_region_names.add(channel_name)
            continue

        # ins1852821 modifies its control regions to exclude 5-lepton events
        # but we don't have specs for those control regions
        if channel_name.startswith("SR5L"):
            continue

        assert channel_name.startswith("SR")
        signal_region_names.add(channel_name)

    # serialize regions for each
    for signal_region_name in sorted(signal_region_names):
        serial.dump_region(
            os.path.join(BASEPATH, PAPER, strip_cuts(signal_region_name)),
            regionize(workspace, control_region_names, signal_region_name),
        )


# TODO import from elsewhere


def strip_cuts(string, *, cuts="_cuts"):
    if string.endswith(cuts):
        return string[: -len(cuts)]
    return string


def clear_poi(spec):
    """Set all measurement poi in spec to the empty string.

    This avoids exceptions thrown by pyhf workspace stuff.
    """
    for measurement in spec["measurements"]:
        measurement["config"]["poi"] = ""
    return spec


def regionize(workspace, control_region_names, signal_region_name):

    channels_prune = (
        workspace.channel_slices.keys()
        - control_region_names
        - {signal_region_name}
    )

    return pyhf.Workspace.sorted(
        workspace.prune(channels=channels_prune).rename(
            channels={signal_region_name: serial.SIGNAL_REGION_NAME}
        )
    )


if __name__ == "__main__":
    main()
