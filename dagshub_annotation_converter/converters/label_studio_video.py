"""
Label Studio Video format converter.

Provides bidirectional conversion between Video IR and Label Studio Video format.
"""

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
    
    Args:
        annotations: List of video bbox annotations
        video_path: Path to video file for task data
        
    Returns:
        List containing a single LabelStudioTask with all tracks
    """
    if not annotations:
        return []
    
    # Group annotations by track_id
    tracks: Dict[int, List[IRVideoBBoxAnnotation]] = defaultdict(list)
    for ann in annotations:
        tracks[ann.track_id].append(ann)
    
    # Create VideoRectangle for each track
    video_rectangles: List[VideoRectangleAnnotation] = []
    for track_id, track_anns in sorted(tracks.items()):
        video_rect = VideoRectangleAnnotation.from_ir_annotations(track_anns)
        video_rectangles.append(video_rect)
    
    # Create task - use model_construct to bypass validation on annotations
    # then manually set the result list
    task = LabelStudioTask(
        data={"video": video_path},
    )
    
    # Use model_construct to bypass the BeforeValidator on result
    container = AnnotationsContainer.model_construct(
        completed_by=None,
        result=video_rectangles,  # Pass VideoRectangleAnnotation objects directly
        ground_truth=False,
    )
    task.annotations = [container]
    
    return [task]


def ls_video_task_to_video_ir(task: LabelStudioTask) -> List[IRVideoBBoxAnnotation]:
    """
    Convert a Label Studio Video task to Video IR annotations.
    
    Extracts all VideoRectangle annotations and converts them to IR format.
    
    Args:
        task: Label Studio task containing video annotations
        
    Returns:
        List of video bbox annotations
    """
    annotations = []
    
    for container in task.annotations:
        for result in container.result:
            if isinstance(result, VideoRectangleAnnotation):
                annotations.extend(result.to_ir_annotations())
            elif hasattr(result, 'type') and result.type == 'videorectangle':
                # Parse as VideoRectangle if it's a dict-like object
                video_rect = VideoRectangleAnnotation.model_validate(result.model_dump())
                annotations.extend(video_rect.to_ir_annotations())
    
    return annotations


def video_ir_to_ls_video_json(
    annotations: List[IRVideoBBoxAnnotation],
    video_path: str = "/data/video.mp4",
) -> str:
    """
    Convert Video IR annotations to Label Studio Video JSON format.
    
    Args:
        annotations: List of video bbox annotations
        video_path: Path to video file for task data
        
    Returns:
        JSON string of Label Studio task
    """
    tasks = video_ir_to_ls_video_tasks(annotations, video_path)
    if not tasks:
        return "[]"
    
    return tasks[0].model_dump_json(indent=2)


def ls_video_json_to_video_ir(json_str: str) -> List[IRVideoBBoxAnnotation]:
    """
    Convert Label Studio Video JSON to Video IR annotations.
    
    Args:
        json_str: JSON string of Label Studio task
        
    Returns:
        List of video bbox annotations
    """
    # Need to register videorectangle type for parsing
    from dagshub_annotation_converter.formats.label_studio.task import task_lookup
    task_lookup["videorectangle"] = VideoRectangleAnnotation
    
    task = LabelStudioTask.model_validate_json(json_str)
    return ls_video_task_to_video_ir(task)
