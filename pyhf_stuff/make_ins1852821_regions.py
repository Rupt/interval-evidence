import os

import jsongz
import pyhf
import serial

BASEPATH = os.path.dirname(__file__)
PAPER = "ins1852821"

# TODO standardize standardizable parts

def main():
    spec = jsongz.load(os.path.join(BASEPATH, PAPER + "_bkg.json.gz"))

    # find control and signals region strings
    control_region_names = set()
    signal_region_names = set()
    for channel in spec["channels"]:
        name = channel["name"]

        if name.startswith("CR"):
            control_region_names.add(name)
            continue

        # ins1852821 modifies its control regions to exclude 5-lepton events
        # but we don't have specs for those control regions
        if name.startswith("SR5L"):
            continue

        assert name.startswith("SR")
        signal_region_names.add(name)

    # serialize regions for each
    spec = clear_poi(spec)
    workspace = pyhf.workspace.Workspace(spec)

    channels_prune_cr = workspace.channel_slices.keys() - control_region_names

    for signal_region_name in sorted(signal_region_names):
        channels_prune = channels_prune_cr - {signal_region_name}

        assert signal_region_name.endswith("_cuts")
        label = signal_region_name[: -len("_cuts")]
        path = os.path.join(BASEPATH, PAPER, label)

        serial.dump_region(
            path,
            pyhf.Workspace.sorted(
                workspace.prune(channels=channels_prune).rename(
                    channels={signal_region_name: serial.SIGNAL_REGION_NAME}
                )
            ),
        )


# TODO import from elsewhere


def clear_poi(spec):
    """Set all measurement poi in spec to the empty string.

    This avoids exceptions thrown by pyhf workspace stuff.
    """
    for measurement in spec["measurements"]:
        measurement["config"]["poi"] = ""
    return spec


if __name__ == "__main__":
    main()
