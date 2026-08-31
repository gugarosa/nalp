from importlib.metadata import version

import nalp


def test_package_version_matches_metadata():
    assert nalp.__version__ == version("nalp")
