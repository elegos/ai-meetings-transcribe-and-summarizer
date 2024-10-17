from pathlib import Path


def unlink(file_path: Path):
    if file_path.exists():
        file_path.unlink()