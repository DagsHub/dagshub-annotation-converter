import configparser
from pathlib import Path
from typing import Optional

from pydantic import Field

from dagshub_annotation_converter.formats.categories import Categories
from dagshub_annotation_converter.util.pydantic_util import ParentModel


class MOTContext(ParentModel):
    """
    Context for MOT format import/export.

    CVAT MOT 1.1 format (9 columns):
    ``frame_id, track_id, x, y, w, h, "not ignored", class_id, visibility``

    Categories are loaded from labels.txt (one class per line, 1-indexed).
    """

    frame_rate: float = 30.0
    frame_subdirectory: str = "img1"
    frame_extension: str = ".jpg"
    video_width: Optional[int] = None
    video_height: Optional[int] = None
    sequence_name: Optional[str] = None
    """Name of the video sequence"""
    sequence_length: Optional[int] = None
    """Length of the video sequence in frames"""
    categories: Categories = Field(default_factory=lambda: Categories(start_index=1))
    """Mapping of class_id (1-indexed) to category name."""
    default_category: str = "object"

    @staticmethod
    def from_seqinfo_string(content: str) -> "MOTContext":
        """
        Load context from seqinfo.ini file's content.

        Example seqinfo.ini::

            [Sequence]
            name=test_sequence
            imDir=img1
            frameRate=30
            seqLength=100
            imWidth=1920
            imHeight=1080
            imExt=.jpg
        """

        config = configparser.ConfigParser()
        config.read_string(content)
        ctx = MOTContext()
        if config.has_section("Sequence"):
            seq = config["Sequence"]
            ctx.sequence_name = seq.get("name")
            ctx.frame_rate = float(seq.get("frameRate", "30.0"))
            ctx.frame_extension = seq.get("imExt", ".jpg")
            ctx.frame_subdirectory = seq.get("imDir", "img1")
            ctx.sequence_length = int(seq.get("seqLength", "0")) or None
            ctx.video_width = int(seq.get("imWidth", "0")) or None
            ctx.video_height = int(seq.get("imHeight", "0")) or None
        return ctx

    @staticmethod
    def load_labels(labels_path: Path) -> Categories:
        """
        Load category mapping from labels.txt (one class per line).
        """
        return MOTContext.load_labels_from_string(labels_path.read_text(encoding="utf-8"))

    @staticmethod
    def load_labels_from_string(content: str) -> Categories:
        categories = Categories(start_index=1)
        for idx, line in enumerate(content.split(), start=1):
            name = line.strip()
            if name:
                categories.add(name, idx)
        return categories

    def write_labels(self, labels_path: Path):
        sorted_categories = sorted(self.categories, key=lambda x: x.id)
        with open(labels_path, "w") as f:
            for cat in sorted_categories:
                f.write(f"{cat.name}\n")

    def write_seqinfo(self, seqinfo_path: Path):
        config = configparser.ConfigParser()
        # Preserve case for all values on writing
        config.optionxform = str  # type: ignore
        config["Sequence"] = {}
        seq = config["Sequence"]

        if self.sequence_name:
            seq["name"] = self.sequence_name
        seq["frameRate"] = str(int(round(self.frame_rate)))
        if self.sequence_length:
            seq["seqLength"] = str(self.sequence_length)
        if self.video_width:
            seq["imWidth"] = str(self.video_width)
        if self.video_height:
            seq["imHeight"] = str(self.video_height)
        seq["imDir"] = self.frame_subdirectory
        seq["imExt"] = self.frame_extension

        with open(seqinfo_path, "w") as f:
            config.write(f)
