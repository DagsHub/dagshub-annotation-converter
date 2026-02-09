import pytest
from pathlib import Path
import json


@pytest.fixture
def sample_ls_video_task() -> Path:
    return Path(__file__).parent / "res" / "sample_task.json"


@pytest.fixture
def sample_ls_video_task_data(sample_ls_video_task) -> dict:
    with open(sample_ls_video_task) as f:
        return json.load(f)
