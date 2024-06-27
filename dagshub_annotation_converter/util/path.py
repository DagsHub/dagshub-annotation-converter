from pathlib import Path
from typing import Optional


def get_extension(path: Path) -> str:
    name = path.name
    ext_dot_index = name.rfind(".")
    ext = name[ext_dot_index:]
    return ext


def replace_folder(in_path: Path, to_replace: str, replace_with: str, label_extension: Optional[str] = ".txt") -> Path:
    """Swaps LAST occurrence of to_replace with replace_with in the path."""
    new_parts = list(in_path.parts)

    # Replace last occurrence
    for i, part in enumerate(reversed(in_path.parts)):
        if part == to_replace:
            new_parts[len(new_parts) - i - 1] = replace_with

    if label_extension is not None:
        # Replace the extension
        new_parts[-1] = new_parts[-1].replace(get_extension(in_path), label_extension)
    return Path(*new_parts)
