import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="session")
def synthetic_dng(tmp_path_factory):
    """A synthetic noisy capture written as a DNG, plus its clean mosaic."""
    from fixtures import write_synthetic_dng

    d = tmp_path_factory.mktemp("raw")
    path, clean, profile = write_synthetic_dng(str(d / "capture.dng"), h=256, w=256)
    return path, clean, profile
