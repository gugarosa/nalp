import tomllib
from importlib.metadata import version
from pathlib import Path

import nalp


def test_package_version_matches_metadata():
    project = Path(__file__).resolve().parents[1].joinpath("pyproject.toml")
    metadata = tomllib.loads(project.read_text(encoding="utf-8"))

    assert nalp.__version__ == version("nalp") == metadata["project"]["version"]
