import pytest
from pathlib import Path


@pytest.fixture
def sample_cvat_video_xml() -> Path:
    return Path(__file__).parent / "res" / "sample_video.xml"
