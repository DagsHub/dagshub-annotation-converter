from typing import List, Optional

from dagshub_annotation_converter.formats.label_studio.videorectangle import (
    VideoRectangleAnnotation,
)
from dagshub_annotation_converter.formats.label_studio.task import (
    LabelStudioTask,
    AnnotationsContainer,
)
from dagshub_annotation_converter.ir.video import IRVideoSequence


def _resolve_video_path_for_export(sequence: IRVideoSequence, video_path: Optional[str]) -> str:
    if sequence.filename:
        return sequence.filename
    if video_path:
        return video_path
    raise ValueError(
        "Cannot determine video path for Label Studio video export. "
        "Provide video_path explicitly or set sequence.filename."
    )


def video_ir_to_ls_video_tasks(
    sequence: IRVideoSequence,
    video_path: Optional[str] = None,
) -> List[LabelStudioTask]:
    """
    Convert Video IR annotations to Label Studio Video tasks.

    Creates one VideoRectangle per track and combines them into a single task.
    """
    if not sequence.tracks:
        return []

    video_rectangles = [
        VideoRectangleAnnotation.from_ir_track(track, frames_count=sequence.sequence_length)
        for track in sequence.tracks
    ]

    resolved_video_path = _resolve_video_path_for_export(sequence, video_path)

    task = LabelStudioTask(
        data={"video": resolved_video_path},
    )
    container = AnnotationsContainer.model_construct(
        completed_by=None,
        result=video_rectangles,
        ground_truth=False,
    )
    task.annotations = [container]

    return [task]


def ls_video_task_to_video_ir(task: LabelStudioTask) -> IRVideoSequence:
    """Convert a Label Studio Video task to a Video IR sequence."""
    tracks = []
    sequence_length = None

    for container in task.annotations:
        for result in container.result:
            if isinstance(result, VideoRectangleAnnotation):
                track = result.to_ir_track()
            elif hasattr(result, "type") and result.type == "videorectangle":
                video_rect = VideoRectangleAnnotation.model_validate(result.model_dump())
                track = video_rect.to_ir_track()
            else:
                continue

            if result.value.framesCount is not None and result.value.framesCount > 0:
                if sequence_length is None:
                    sequence_length = result.value.framesCount
                else:
                    sequence_length = max(sequence_length, result.value.framesCount)
            tracks.append(track)

    video_path = task.data.get("video")
    return IRVideoSequence(
        tracks=tracks,
        filename=video_path,
        sequence_length=sequence_length,
    )


def video_ir_to_ls_video_json(
    sequence: IRVideoSequence,
    video_path: Optional[str] = None,
) -> str:
    tasks = video_ir_to_ls_video_tasks(sequence, video_path)
    if not tasks:
        return "[]"
    return tasks[0].model_dump_json(indent=2)


def ls_video_json_to_video_ir(json_str: str) -> IRVideoSequence:
    """Convert Label Studio Video JSON to a Video IR sequence."""
    task = LabelStudioTask.model_validate_json(json_str)
    return ls_video_task_to_video_ir(task)
