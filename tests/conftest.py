"""Seam 1 harness: run the CLI against a project root built from the fixture data folder."""

import zipfile
from pathlib import Path

import pytest

from recurring_family.cli import main

FIXTURE_DATA = Path(__file__).parent / "fixtures" / "data"


def make_zip(src: Path, dest: Path) -> Path:
    with zipfile.ZipFile(dest, "w") as zf:
        for f in sorted(src.iterdir()):
            zf.write(f, arcname=f.name)
    return dest


class Project:
    def __init__(self, root: Path):
        self.root = root
        self.zip = make_zip(FIXTURE_DATA, root / "dataset.zip")

    def run(self, *args: str) -> int:
        return main([*args, "--root", str(self.root)])

    @property
    def raw(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def log(self) -> Path:
        return self.root / "experiments" / "log.csv"


@pytest.fixture
def project(tmp_path) -> Project:
    return Project(tmp_path)


@pytest.fixture
def fetched(project) -> Project:
    assert project.run("fetch-data", "--zip", str(project.zip)) == 0
    return project
