"""Pytest fixtures for Label Studio Video format tests."""
import pytest
from pathlib import Path
import json


@pytest.fixture
def sample_ls_video_task() -> Path:
    """Path to sample Label Studio video task JSON file."""
    return Path(__file__).parent / "res" / "sample_task.json"


@pytest.fixture
def sample_ls_video_task_data(sample_ls_video_task) -> dict:
    """Loaded sample Label Studio video task data."""
    with open(sample_ls_video_task) as f:
        return json.load(f)


# Re-export fixture for use in roundtrip tests
@pytest.fixture
def mot_sample_file() -> Path:
    """Path to MOT sample file for cross-format tests."""
    return Path(__file__).parent.parent / "mot" / "res" / "sample.txt"
