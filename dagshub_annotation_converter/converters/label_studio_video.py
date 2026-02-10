import logging
from collections import defaultdict
from typing import Dict, List

from dagshub_annotation_converter.formats.label_studio.videorectangle import (
    VideoRectangleAnnotation,
)
from dagshub_annotation_converter.formats.label_studio.task import (
    LabelStudioTask,
    AnnotationsContainer,
    task_lookup,
)
from dagshub_annotation_converter.ir.video import IRVideoBBoxAnnotation

logger = logging.getLogger(__name__)

# Register VideoRectangleAnnotation in task_lookup for proper parsing
task_lookup["videorectangle"] = VideoRectangleAnnotation


def video_ir_to_ls_video_tasks(
    annotations: List[IRVideoBBoxAnnotation],
    video_path: str = "/data/video.mp4",
) -> List[LabelStudioTask]:
    """
    Convert Video IR annotations to Label Studio Video tasks.

    Groups annotations by track_id and creates one VideoRectangle per track.
    All annotations are combined into a single task.
    """
    if not annotations:
        return []

    tracks: Dict[int, List[IRVideoBBoxAnnotation]] = defaultdict(list)
    for ann in annotations:
        tracks[ann.track_id].append(ann)

    video_rectangles: List[VideoRectangleAnnotation] = []
    for track_id, track_anns in sorted(tracks.items()):
        video_rect = VideoRectangleAnnotation.from_ir_annotations(track_anns)
        video_rectangles.append(video_rect)

    # Use model_construct to bypass the BeforeValidator on result
    task = LabelStudioTask(
        data={"video": video_path},
    )
    container = AnnotationsContainer.model_construct(
        completed_by=None,
        result=video_rectangles,
        ground_truth=False,
    )
    task.annotations = [container]

    return [task]


def ls_video_task_to_video_ir(task: LabelStudioTask) -> List[IRVideoBBoxAnnotation]:
    """Convert a Label Studio Video task to Video IR annotations."""
    annotations = []

    for container in task.annotations:
        for result in container.result:
            if isinstance(result, VideoRectangleAnnotation):
                annotations.extend(result.to_ir_annotations())
            elif hasattr(result, 'type') and result.type == 'videorectangle':
                video_rect = VideoRectangleAnnotation.model_validate(result.model_dump())
                annotations.extend(video_rect.to_ir_annotations())

    return annotations


def video_ir_to_ls_video_json(
    annotations: List[IRVideoBBoxAnnotation],
    video_path: str = "/data/video.mp4",
) -> str:
    tasks = video_ir_to_ls_video_tasks(annotations, video_path)
    if not tasks:
        return "[]"
    return tasks[0].model_dump_json(indent=2)


def ls_video_json_to_video_ir(json_str: str) -> List[IRVideoBBoxAnnotation]:
    """Convert Label Studio Video JSON to Video IR annotations."""
    task = LabelStudioTask.model_validate_json(json_str)
    return ls_video_task_to_video_ir(task)
