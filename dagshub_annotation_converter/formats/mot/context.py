"""MOT format context and configuration."""

from pathlib import Path
from typing import Dict, Optional
import configparser

from dagshub_annotation_converter.util.pydantic_util import ParentModel


class MOTContext(ParentModel):
    """
    Context for MOT format import/export.
    
    CVAT MOT 1.1 format (9 columns):
    frame_id, track_id, x, y, w, h, "not ignored", class_id, visibility
    
    Categories are loaded from labels.txt (one class per line, 1-indexed).
    """

    frame_rate: float = 30.0
    """Frame rate of the video sequence."""
    
    image_width: Optional[int] = None
    """Width of video frames in pixels."""
    
    image_height: Optional[int] = None
    """Height of video frames in pixels."""
    
    seq_name: Optional[str] = None
    """Name of the sequence."""
    
    seq_length: Optional[int] = None
    """Number of frames in the sequence."""
    
    categories: Dict[int, str] = {}
    """Mapping of class_id (1-indexed) to category name."""
    
    default_category: str = "object"
    """Default category name when class_id is not in categories map."""

    @staticmethod
    def from_seqinfo(seqinfo_path: Path) -> "MOTContext":
        """
        Load context from seqinfo.ini file.
        
        Example seqinfo.ini:
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
        config.read(seqinfo_path)
        
        ctx = MOTContext()
        
        if config.has_section("Sequence"):
            seq = config["Sequence"]
            ctx.seq_name = seq.get("name")
            ctx.frame_rate = float(seq.get("frameRate", 30.0))
            ctx.seq_length = int(seq.get("seqLength", 0)) or None
            ctx.image_width = int(seq.get("imWidth", 0)) or None
            ctx.image_height = int(seq.get("imHeight", 0)) or None
        
        return ctx

    @staticmethod
    def load_labels(labels_path: Path) -> Dict[int, str]:
        """
        Load category mapping from labels.txt file.
        
        labels.txt format (one class per line):
        person
        car
        bicycle
        
        Returns dict mapping class_id (1-indexed) to name.
        """
        categories = {}
        with open(labels_path, "r") as f:
            for idx, line in enumerate(f, start=1):
                name = line.strip()
                if name:
                    categories[idx] = name
        return categories

    def get_category_name(self, class_id: int) -> str:
        """Get category name for a class_id, or default if not found."""
        return self.categories.get(class_id, self.default_category)

    def get_class_id(self, category_name: str) -> int:
        """Get class_id for a category name, or 1 if not found."""
        for class_id, name in self.categories.items():
            if name == category_name:
                return class_id
        # If not found, add it and return the new ID
        new_id = max(self.categories.keys(), default=0) + 1
        self.categories[new_id] = category_name
        return new_id

    def write_labels(self, labels_path: Path):
        """Write categories to labels.txt file."""
        # Sort by class_id to maintain order
        sorted_cats = sorted(self.categories.items(), key=lambda x: x[0])
        with open(labels_path, "w") as f:
            for _, name in sorted_cats:
                f.write(f"{name}\n")

    def write_seqinfo(self, seqinfo_path: Path):
        """Write context to seqinfo.ini file."""
        config = configparser.ConfigParser()
        config["Sequence"] = {}
        seq = config["Sequence"]
        
        if self.seq_name:
            seq["name"] = self.seq_name
        seq["frameRate"] = str(int(self.frame_rate))
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
