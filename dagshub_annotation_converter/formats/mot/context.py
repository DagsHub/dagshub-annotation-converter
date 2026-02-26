import io
from pathlib import Path
from typing import Dict, Optional
import configparser

from dagshub_annotation_converter.util.pydantic_util import ParentModel


class MOTContext(ParentModel):
    """
    Context for MOT format import/export.

    CVAT MOT 1.1 format (9 columns):
    ``frame_id, track_id, x, y, w, h, "not ignored", class_id, visibility``

    Categories are loaded from labels.txt (one class per line, 1-indexed).
    """

    frame_rate: float = 30.0
    image_width: Optional[int] = None
    image_height: Optional[int] = None
    seq_name: Optional[str] = None
    seq_length: Optional[int] = None
    categories: Dict[int, str] = {}
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
            ctx.seq_name = seq.get("name")
            ctx.frame_rate = float(seq.get("frameRate", "30.0"))
            ctx.seq_length = int(seq.get("seqLength", "0")) or None
            ctx.image_width = int(seq.get("imWidth", "0")) or None
            ctx.image_height = int(seq.get("imHeight", "0")) or None
        return ctx

    @staticmethod
    def load_labels(labels_path: Path) -> Dict[int, str]:
        """
        Load category mapping from labels.txt (one class per line).

        Returns dict mapping class_id (1-indexed) to name.
        """
        categories = {}
        with open(labels_path, "r") as f:
            for idx, line in enumerate(f, start=1):
                name = line.strip()
                if name:
                    categories[idx] = name
        return categories

    @staticmethod
    def load_labels_from_string(content: str) -> Dict[int, str]:
        categories = {}
        for idx, line in enumerate(io.StringIO(content), start=1):
            name = line.strip()
            if name:
                categories[idx] = name
        return categories

    def get_category_name(self, class_id: int) -> str:
        return self.categories.get(class_id, self.default_category)

    def get_class_id(self, category_name: str) -> int:
        for class_id, name in self.categories.items():
            if name == category_name:
                return class_id
        new_id = max(self.categories.keys(), default=0) + 1
        self.categories[new_id] = category_name
        return new_id

    def write_labels(self, labels_path: Path):
        sorted_cats = sorted(self.categories.items(), key=lambda x: x[0])
        with open(labels_path, "w") as f:
            for _, name in sorted_cats:
                f.write(f"{name}\n")

    def write_seqinfo(self, seqinfo_path: Path):
        config = configparser.ConfigParser()
        # Preserve case for all values on writing
        config.optionxform = str  # type: ignore
        config["Sequence"] = {}
        seq = config["Sequence"]

        if self.seq_name:
            seq["name"] = self.seq_name
        seq["frameRate"] = str(int(round(self.frame_rate)))
        if self.seq_length:
            seq["seqLength"] = str(self.seq_length)
        if self.image_width:
            seq["imWidth"] = str(self.image_width)
        if self.image_height:
            seq["imHeight"] = str(self.image_height)
        seq["imDir"] = "img1"
        seq["imExt"] = ".jpg"

        with open(seqinfo_path, "w") as f:
            config.write(f)
