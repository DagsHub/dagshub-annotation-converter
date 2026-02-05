"""Pytest fixtures for CVAT Video format tests."""
import pytest
from pathlib import Path


@pytest.fixture
def sample_cvat_video_xml() -> Path:
    """Path to sample CVAT video XML file."""
    return Path(__file__).parent / "res" / "sample_video.xml"
