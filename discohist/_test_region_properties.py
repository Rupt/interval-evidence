import pyhf

from .region import Region
from .region_properties import region_properties


def test_cache():
    # modified from
    # https://scikit-hep.org/pyhf/likelihood.html#toy-example
    spec = {
        "channels": [
            {
                "name": "singlechannel",
                "samples": [
                    {
                        "name": "signal",
                        "data": [5.0],
                        "modifiers": [
                            {"name": "mu", "type": "normfactor", "data": None}
                        ],
                    },
                    {
                        "name": "background",
                        "data": [50.0],
                        "modifiers": [
                            {
                                "name": "uncorr_bkguncrt",
                                "type": "shapesys",
                                "data": [5.0],
                            }
                        ],
                    },
                ],
            }
        ],
        "observations": [{"name": "singlechannel", "data": [50.0]}],
        "measurements": [
            {"name": "Measurement", "config": {"poi": "mu", "parameters": []}}
        ],
        "version": "1.0.0",
    }

    region = Region("singlechannel", pyhf.Workspace(spec))

    props = region_properties(region)
    assert props is region_properties(region)
